"""One-rank GenEvolve Route-SFT. Original objective plus Good-only full-mass type KL."""
from contextlib import nullcontext
import gc
import random
import shutil
from statistics import fmean

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import read_json,write_json
from .distributed import (initialize,latest,load_checkpoint,make_sharded_actor,load_model,
                          save_checkpoint,export_policy,restore_optimizer,restore_rng)
from .policy import ActorScorer
from .route_data import (MARKER_ID,load_pack,pack_manifest,parent_tokenizer,recipe,register_marker,
    require,route_identity,source_checkpoint_lock,verify_artifact_identity,verify_parent,
    verify_parent_identity,verify_tokenizer)
from .sft_export import sft_manifest
from .route_type_kl import TYPE_IDS,THREEWAY_VERSION,good_threeway_kl,prepare_reference_cache,score_threeway,verify_type_ids


def route_bce(logits, marker_id, is_bad):
    logits = logits.float()
    other = torch.logsumexp(torch.cat((logits[:marker_id], logits[marker_id + 1:])), dim=0)
    binary = logits[marker_id] - other
    return F.binary_cross_entropy_with_logits(binary, binary.new_tensor(float(is_bad))), binary.sigmoid()


class RouteScorer(ActorScorer):
    def __init__(self, model, projection_chunk=128, type_ids=TYPE_IDS):
        super().__init__(model, projection_chunk)
        self.type_ids = type_ids

    def forward(self, ids, start, is_bad, marker_id=MARKER_ID, route_lambda=0.1, *, branch="suffix", reference=None):
        if branch == "kl":
            require(not is_bad and reference is not None, "Only Good rows receive type KL.")
            student = score_threeway(self.model, ids, start, self.type_ids)
            return good_threeway_kl(student, torch.as_tensor(reference, dtype=torch.float32, device=ids.device))
        require(branch == "suffix", "Unknown training branch.")
        h=self.model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0,start-1:-1]
        t=ids[0,start:]
        def part(x,y):return F.cross_entropy(self.model.lm_head(x).float(),y,reduction='sum')
        values=[checkpoint(part,x,y,use_reentrant=False) if torch.is_grad_enabled() else part(x,y)
                for x,y in zip(h.split(self.projection_chunk),t.split(self.projection_chunk))]
        ce=torch.stack(values).sum()/len(t)
        z=self.model.lm_head(h[0]).float()
        bce,prob=route_bce(z,marker_id,is_bad)
        return ce+route_lambda*bce,ce.detach(),bce.detach(),prob.detach()

def verify_route_export(path, cfg=None, expected_contract=None):
    cfg = cfg or recipe()
    total = cfg["batches_per_epoch"] * cfg["epochs"]
    done = read_json(path / "completed.json") if (path / "completed.json").is_file() else {}
    identity = done.get("contract") or {}
    if (done.get("algorithm") != "route-sft" or done.get("step") != total
            or identity.get("config") != cfg
            or identity.get("route_data_source_sha256") != source_checkpoint_lock()):
        raise ValueError("Stage 2 must start from the complete configured Route-SFT HF export.")
    verify_artifact_identity(identity.get("manifest_sha256"), identity.get("artifact_sha256"), cfg)
    verify_parent_identity(identity.get("parent"), cfg)
    if expected_contract is not None and identity != expected_contract:
        raise ValueError("Route-SFT export does not match the supplied 2B parent and v4 pack; use a new WORK.")
    receipt = read_json(path / "export_sha256.json") if (path / "export_sha256.json").is_file() else {}
    manifest = sft_manifest(path)
    if not receipt or manifest != receipt:
        raise ValueError("Final Route-SFT export SHA-256 does not match its receipt.")
    return {"completed": done, **manifest}


def hardware_suggest(torch,device):
    free,total=torch.cuda.mem_get_info(device)
    require(torch.cuda.is_bf16_supported(),'SFT requires BF16 CUDA')
    require(total>=29*2**30 and free>=min(total*0.85,58*(2**30)), 'Insufficient available GPU RAM for this single-rank profile')
    # CPU offload is an implementation choice, not a different optimizer or quantization.
    return {'world_size':1,'cpu_offload':total<65*2**30,'gpu_name':torch.cuda.get_device_name(device)}

def update(actor,opt,rows,bb,cfg,device,references,marker_id=MARKER_ID):
    """One update: eight suffix/BCE gradients and six independently averaged Good KL gradients."""
    opt.zero_grad(set_to_none=True)
    jobs=[(idx,branch) for idx in bb for branch in (('suffix','kl_branch')
          if rows[idx]['route_label']==0 else ('suffix',))]
    sums=[];ces=[];bces=[];pp={0:[],1:[]};kls=[]
    for ii,(idx,branch) in enumerate(jobs):
        r=rows[idx];bad=r['route_label'];startpos=r['target_start']
        # Separate CE and Good-only KL backward: bound peak RAM, then a single common optimizer step.
        ids=torch.tensor([r['input_ids'] if branch=='suffix' else r['input_ids'][:startpos]],dtype=torch.long,device=device)
        with actor.no_sync() if ii+1<len(jobs) and hasattr(actor, "no_sync") else nullcontext():
            if branch=='suffix':
                raw,ce,bce,prob=actor(ids,startpos,bad,marker_id,cfg['route_lambda'])
                loss=raw/len(bb)
                ces.append(ce.detach().item());bces.append(bce.detach().item());pp[bad].append(prob.item())
            else:
                kl=actor(ids,startpos,bad,branch="kl",reference=references[idx])
                loss=cfg['good_type_kl_lambda']*kl/cfg['good_per_batch'];kls.append(kl.detach().item())
            if not torch.isfinite(loss).item():raise FloatingPointError('Non-finite route loss')
            sums.append(loss.detach().item());loss.backward()
    norm=(actor.clip_grad_norm_(cfg['clip_grad']) if hasattr(actor, 'clip_grad_norm_') else
          torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg['clip_grad']))
    if not torch.isfinite(norm).item():raise FloatingPointError('Non-finite route gradient')
    opt.step()
    metrics=dict(loss=sum(sums),suffix_ce=fmean(ces),route_bce=fmean(bces),
                 good_p_rethink=fmean(pp[0]),bad_p_rethink=fmean(pp[1]),grad_norm=float(norm))
    if kls:metrics.update(good_threeway_kl=fmean(kls),weighted_kl=cfg['good_type_kl_lambda']*fmean(kls))
    return metrics


def worker(work, parent, data):
    from transformers import get_cosine_schedule_with_warmup, AutoProcessor
    cfg=recipe()
    require((cfg['epochs'],cfg['batches_per_epoch'],cfg['world_size'],cfg['good_per_batch'],cfg['bad_per_batch'],
             cfg['micro_batch'],cfg['good_type_kl_lambda'],cfg['good_type_kl_version']) ==
            (2,281,1,6,2,1,0.1,THREEWAY_VERSION),'Incompatible two-epoch Good-type-KL recipe.')
    m=pack_manifest(data,cfg)
    rows,batches=load_pack(data,cfg,m)
    c=route_identity(data,m,cfg,verify_parent(parent,cfg))
    tok=parent_tokenizer(parent)
    verify_type_ids(tok)
    verify_tokenizer(rows,tok)
    del tok
    rank,size,device=initialize(world_size=1)
    try:
        prof=hardware_suggest(torch,device)
        out=work.resolve()/'route_sft';out.mkdir(parents=True,exist_ok=True)
        hardware={'world_size':1,'cpu_offload':prof['cpu_offload']}
        if (out/'hardware.json').exists():require(hardware==read_json(out/'hardware.json'), 'Resume hardware profile mismatch.')
        write_json(out/'hardware.json',hardware)
        known={out/'checkpoint-00281', out/'checkpoint-00562'}
        require(set(out.glob('checkpoint-*'))<=known,'Unrelated checkpoint in output folder; refusing to overwrite or prune')
        cp=latest(out)
        if cp:require((cp/'state_sha256.json').is_file(),'Unaudited prior training checkpoint')
        if (out/'contract.json').exists():require(c==read_json(out/'contract.json'),'Run directory contract differs.')
        if not cp:require(not any(out.glob('metrics/step-*.json')), 'Partial prior training found without a resumable checkpoint; use a fresh output root')
        write_json(out/'contract.json',c)
        cfg=c['config']; total=len(batches)*cfg['epochs']
        require(total==562 and len(batches)==281,'Wrong epoch mapping')
        need={None:58,'checkpoint-00281':35,'checkpoint-00562':9}[cp.name if cp else None]
        require(shutil.disk_usage(out).free >= need*2**30,f'Not enough disk; need >= {need} GiB for outstanding saves.')
        saved=load_checkpoint(cp) if cp else None;start=saved['step'] if saved else 0
        require(not saved or (saved['contract']==c and start in (281,total) and cp.name==f'checkpoint-{start:05d}'),'Invalid saved state')
        references=None
        if start<total:
            references=prepare_reference_cache(out,parent,rows,c,rank,size,device)
        random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        actor=make_sharded_actor(parent,device,saved['actor'] if saved else None,scorer_type=RouteScorer,
                                 cpu_offload=prof['cpu_offload'])
        opt=torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad),lr=cfg['lr'],betas=cfg['betas'],
                              weight_decay=cfg['weight_decay'],foreach=False)
        sched=get_cosine_schedule_with_warmup(opt,int(cfg['warmup_ratio']*total),total)
        if saved:
            restore_optimizer(actor,opt,sched,saved,rank);restore_rng(saved['rng'][rank],device)
        del saved
        actor.train()
        for j in range(start,total):
            epoch0,batch0=divmod(j,len(batches))
            bb=batches[batch0]
            metrics=update(actor,opt,rows,bb,cfg,device,references)
            sched.step(); n=j+1
            metrics.update(step=n,epoch=epoch0+1,batch_in_epoch=batch0+1,lr=sched.get_last_lr()[0])
            write_json(out/'metrics'/f'step-{n:05d}.json',metrics)
            if n==1 or n%5==0 or n==total:print(metrics,flush=True)
            if n%len(batches)==0:save_checkpoint(out,actor,opt,sched,n,c,rank,device,keep=2,verify=True)
        del opt,sched;gc.collect();torch.cuda.empty_cache()
        hf=out/f'hf-step-{total:05d}'
        if not hf.exists():
            rep=load_model(parent,torch.bfloat16,attention='sdpa')
            proc=AutoProcessor.from_pretrained(parent,local_files_only=True);register_marker(proc.tokenizer)
            export_policy(hf,actor,rep,proc,total,rank,metadata={'algorithm':'route-sft','contract':c})
        require(read_json(hf/'completed.json')=={'step':total,'algorithm':'route-sft','contract':c},'Wrong HF export')
        receipt=sft_manifest(hf)
        if (hf/'export_sha256.json').is_file():require(receipt==read_json(hf/'export_sha256.json'),'HF digest mismatch')
        else:write_json(hf/'export_sha256.json',receipt)
        print('COMPLETE',hf,flush=True)
    finally:
        dist.destroy_process_group()
