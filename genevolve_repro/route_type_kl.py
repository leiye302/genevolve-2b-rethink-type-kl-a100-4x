"""Native Tool / Answer / Other probabilities for the Good-only SFT-550 forward KL."""
import hashlib
import gc
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch
from torch.nn import functional as F

TYPE_IDS = (151657, 27, 9217)
THREEWAY_VERSION = "parent550-good-forward-kl-full-tool-answer-other-mass-v1"
def verify_type_ids(tokenizer):
    if (tokenizer.encode("<tool_call>", add_special_tokens=False) != [TYPE_IDS[0]] or
            tokenizer.encode("<answer", add_special_tokens=False) != list(TYPE_IDS[1:])):
        raise ValueError("Native tokenizer does not have the expected action openers.")


def make_candidate_prefix(input_ids, start, type_ids=TYPE_IDS):
    """H/R followed only by hypothetical '<'; works with prefix-only parent input."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 0 < start <= input_ids.shape[1]:
        raise ValueError("Expected one sequence and a valid boundary.")
    return torch.cat((input_ids[:, :start], input_ids.new_tensor([[type_ids[1]]])), 1)


def _lse_except(z, skip):
    # The raw complement '1-exp(log_t)-exp(log_a)' is unreliable if O is tiny.
    ids = sorted(skip)
    pieces = [z[lo:hi] for lo,hi in zip([0] + [s+1 for s in ids], ids+[len(z)]) if hi > lo]
    if not pieces:
        raise ValueError("No remaining vocabulary items.")
    return torch.logsumexp(torch.cat(pieces), 0)


def threeway_log_probs(first_logits, after_open_logits, type_ids=TYPE_IDS):
    """Exact log probabilities of disjoint cylinder events under the full vocabulary.

    log O combines first-token-other and '<' followed by something else. O also
    includes participating vocabulary such as EOS and RETHINK; do not discard it.
    T/A are opener events, not probabilities of an entire valid JSON program.
    """
    if (first_logits.ndim != 1 or after_open_logits.shape != first_logits.shape or
            len(set(type_ids)) != 3 or min(type_ids) < 0 or max(type_ids) >= first_logits.numel()):
        raise ValueError("Vocabulary/shape mismatch.")
    t, opening, answer = type_ids
    first, after = first_logits.float(), after_open_logits.float()
    ft, fo, fr = first[t], first[opening], _lse_except(first, (t, opening))
    a, r = after[answer], _lse_except(after, (answer,))
    # Renormalize after subtracting the maximum, for stability even if a branch
    # dominates and the other two carry tiny but nonzero mass.
    second = F.log_softmax(torch.stack((a, r)), 0)
    scores = torch.stack((ft, fo + second[0], torch.logaddexp(fr, fo + second[1])))
    return F.log_softmax(scores, 0)


def score_threeway(model, input_ids, start, type_ids=TYPE_IDS):
    """Parent under no_grad or differentiable student; no label/action/future read."""
    ids = make_candidate_prefix(input_ids, start, type_ids)
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    first, after_open = model.lm_head(hidden[0,-2:]).unbind(0)
    return threeway_log_probs(first, after_open, type_ids)


def good_threeway_kl(student_logp, reference_logp):
    """Per-Good-row forward KL. Caller must omit this operation on every Bad."""
    if student_logp.shape != (3,) or reference_logp.shape != (3,):
        raise ValueError("Expected three log probabilities; old 2-class caches are incompatible.")
    return F.kl_div(student_logp.float(), reference_logp.detach().float(),
                    reduction="sum", log_target=True)

def _read(p):
    return json.loads(p.read_bytes())


def _write(p, value):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=p.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, separators=(",",":"), allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def _file_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _validate(entries, rows, wanted, complete=False):
    if not isinstance(entries, dict) or not set(entries).issubset(wanted) or (complete and set(entries)!=set(wanted)):
        raise ValueError("Unexpected or missing Good reference cache entries.")
    for k,v in entries.items():
        if not isinstance(v,dict) or v.get("id")!=rows[int(k)]["id"]:
            raise ValueError("Reference row identity mismatch.")
        lp=v.get("logp")
        if (not isinstance(lp,list) or len(lp)!=3 or
                not all(isinstance(x,(float,int)) and math.isfinite(x) and x<=1e-6 for x in lp) or
                abs(sum(math.exp(x) for x in lp)-1.)>2e-5):
            raise ValueError("Invalid three-way reference log probabilities.")


@torch.no_grad()
def prepare_reference_cache(out,parent,rows,contract,rank,size,device,*,loader=None,
                            attention="flash_attention_2",flush_every=16,type_ids=TYPE_IDS):
    """Score frozen parent once, unload before training. Returns only 3 floats/Good."""
    if size not in (1,4) or not 0 <= rank < size or flush_every<1:
        raise ValueError("Use one or four ranks and a positive cache flush interval.")
    if not contract or not isinstance(contract,dict):
        raise ValueError("Missing bound parent/data/config contract.")
    device=torch.device(device)
    config={"version":THREEWAY_VERSION, "contract":contract, "dtype":"bfloat16",
            "attention":attention,"type_ids":list(type_ids),"projection":"full-native-vocabulary"}
    if size != 1:
        config["shard"] = {"rank":rank, "world_size":size, "partition":"row_index_mod_world_size"}
    folder=Path(out)/"good_threeway_reference"
    folder.mkdir(parents=True,exist_ok=True)
    f=folder/f"rank-{rank}.json"
    receipt=folder/f"rank-{rank}.sha256.json"
    all_good={str(j) for j,r in enumerate(rows) if r["route_label"]==0}
    expected={j for j in all_good if int(j) % size == rank}
    saved=_read(f) if f.is_file() else {"identity":config,"entries":{}}
    if saved.get("identity")!=config:
        raise ValueError("Three-way cache is for a different model/data/config.")
    entries=saved["entries"]
    sealed=receipt.is_file()
    if sealed and (not f.is_file() or _read(receipt)!={f.name:_file_sha(f)}):
        raise ValueError("Reference cache seal hash mismatch.")
    _validate(entries,rows,expected,complete=sealed)
    missing=sorted((int(k) for k in expected-set(entries)))
    if missing:
        if loader is None:
            from .distributed import load_model as loader
        model=loader(parent,torch.bfloat16,attention=attention).to(device).eval().requires_grad_(False)
        try:
            for pos,k in enumerate(missing,1):
                r=rows[k]
                start=r["target_start"]
                # Do not even send future tokens or the true Good action to the parent.
                ids=torch.tensor([r["input_ids"][:start]],device=device,dtype=torch.long)
                lp=score_threeway(model,ids,start,type_ids).detach().cpu().tolist()
                entries[str(k)]={"id":r["id"],"logp":lp}
                if pos%flush_every==0 or pos==len(missing):
                    _validate(entries,rows,expected)
                    _write(f,saved)
                    print(f"frozen SFT-550 Good reference: {len(entries)}/{len(expected)}", flush=True)
        finally:
            del model
            gc.collect()
            if device.type=="cuda":
                torch.cuda.empty_cache()
    _validate(entries,rows,expected,complete=True)
    if not f.is_file():
        _write(f,saved)
    if not sealed:
        _write(receipt,{f.name:_file_sha(f)})
    result={int(k):v["logp"] for k,v in entries.items()}
    if size == 1:
        return result
    import torch.distributed as dist
    if not dist.is_initialized() or (rank,size)!=(dist.get_rank(),dist.get_world_size()):
        raise ValueError("Reference shards require a matching process group.")
    parts=[None]*size
    dist.all_gather_object(parts,result)
    merged={}
    for owner,part in enumerate(parts):
        if set(part)!={int(j) for j in all_good if int(j)%size==owner}:
            raise ValueError("Missing, duplicated or misassigned frozen Good reference rows.")
        merged.update(part)
    return merged
