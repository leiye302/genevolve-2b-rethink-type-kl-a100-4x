"""Four-rank Qwen3-VL-2B Route-SFT on the reviewed v3 examples."""

from contextlib import nullcontext
import random
from statistics import fmean

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .common import model_ready, read_json, write_json
from .distributed import (
    barrier, export_policy, gather, initialize, latest, load_checkpoint, load_model,
    make_sharded_actor, restore_optimizer, restore_rng, save_checkpoint,
)
from .policy import ActorScorer
from .route_data import (
    MARKER_ID, load_pack, pack_manifest, parent_tokenizer, recipe, register_marker,
    route_identity, source_checkpoint_lock, verify_artifact_identity, verify_parent,
    verify_parent_identity, verify_tokenizer,
)
from .sft_export import sft_manifest
from .route_type_kl import (
    TYPE_IDS, candidate_prefix, prepare_reference_cache, reference_to_student_kl,
    type_log_probs, validate_config, verify_type_tokenizer,
)


def route_bce(logits, marker_id, is_bad):
    logits = logits.float()
    other = torch.logsumexp(torch.cat((logits[:marker_id], logits[marker_id + 1:])), dim=0)
    binary = logits[marker_id] - other
    return F.binary_cross_entropy_with_logits(binary, binary.new_tensor(float(is_bad))), binary.sigmoid()


class RouteScorer(ActorScorer):
    def __init__(self, model, projection_chunk=128, type_ids=TYPE_IDS):
        super().__init__(model, projection_chunk)
        self.type_ids = type_ids

    def forward(self, input_ids, start, is_bad, marker_id, route_lambda,
                type_kl_weight=0.0, reference_logp=None):
        hidden = self.model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        # Only the supervised token predictors are projected. H/R remain in
        # the attention context; no CE is imposed on their token positions.
        states = hidden.last_hidden_state[0, start - 1:-1]
        targets = input_ids[0, start:]

        def ce_sum(state, ids):
            return F.cross_entropy(self.model.lm_head(state).float(), ids, reduction="sum")

        parts = [checkpoint(ce_sum, chunk, ids, use_reentrant=False) if torch.is_grad_enabled()
                 else ce_sum(chunk, ids)
                 for chunk, ids in zip(states.split(self.projection_chunk), targets.split(self.projection_chunk))]
        ce = torch.stack(parts).sum() / len(targets)
        first_logits = self.model.lm_head(states[0])
        bce, probability = route_bce(first_logits, marker_id, is_bad)
        loss = ce + route_lambda * bce
        if not type_kl_weight:
            # Exactly the original loss and forward graph when lambda=0.
            return loss, ce.detach(), bce.detach(), probability.detach()
        if reference_logp is None:
            raise ValueError("Type KL requires a frozen reference distribution.")
        # Run this graph even for Bad: conditional FSDP submodule calls on
        # different ranks could otherwise deadlock. Bad contributes zero KL.
        candidate = candidate_prefix(input_ids, start, self.type_ids)
        alternate = self.model.model(input_ids=candidate, use_cache=False, return_dict=True)
        after_open = self.model.lm_head(alternate.last_hidden_state[0, -1])
        student_logp = type_log_probs(first_logits, after_open, self.type_ids)
        kl = reference_to_student_kl(student_logp, reference_logp) * (1.0 - float(is_bad))
        loss = loss + type_kl_weight * kl
        return (loss, ce.detach(), bce.detach(), probability.detach(),
                kl.detach(), student_logp[1].exp().detach())


def update(actor, optimizer, rows, batch, cfg, rank, size, device, references=None):
    batch_local = batch[rank::size]
    if len(batch_local) != 2 or cfg["micro_batch"] != 1:
        raise ValueError("Expected two unpadded samples on each of four ranks.")
    optimizer.zero_grad(set_to_none=True)
    stats = []
    type_weight = validate_config(cfg)
    # FSDP + accumulation average over B=8. Scale by B/G to obtain the
    # intended mean over the six Good samples, not a diluted mean over eight.
    good_scale = (cfg["good_per_batch"] + cfg["bad_per_batch"]) / cfg["good_per_batch"]
    for i, k in enumerate(batch_local):
        row = rows[k]
        ctx = actor.no_sync() if hasattr(actor, "no_sync") and i + 1 < len(batch_local) else nullcontext()
        with ctx:
            ids = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
            args = (ids, row["target_start"], row["route_label"], MARKER_ID, cfg["route_lambda"])
            kl, p_answer, ref_answer = 0.0, 0.0, 0.0
            if type_weight:
                ref = ([-0.6931471805599453] * 2 if row["route_label"] else references[k])
                ref = torch.tensor(ref, device=device, dtype=torch.float32)
                loss, ce, bce, p, kl_value, answer_value = actor(*args, type_weight * good_scale, ref)
                kl, p_answer, ref_answer = kl_value.item(), answer_value.item(), ref[1].exp().item()
            else:
                loss, ce, bce, p = actor(*args)
            finite = torch.isfinite(loss.detach()).int()
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite Route-SFT loss; the last saved epoch is unchanged.")
            # FSDP averages across ranks, each rank contributes two rows.
            (loss / len(batch_local)).backward()
        stats.append((row["route_label"], loss.detach().item(), ce.item(), bce.item(), p.item(),
                      kl, p_answer, ref_answer))
    clip = getattr(actor, "clip_grad_norm_", None)
    norm = clip(cfg["clip_grad"]) if clip else torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["clip_grad"])
    if not torch.isfinite(norm).item():
        raise FloatingPointError("Non-finite Route-SFT gradient; the last saved epoch is unchanged.")
    optimizer.step()
    return stats, float(norm)


def run_epoch_export(out, step, actor, parent, rank, contract):
    from transformers import AutoProcessor

    folder = out / f"hf-step-{step:05d}"
    if rank == 0:
        already = model_ready(folder) and (folder / "completed.json").is_file()
    else:
        already = None
    if gather(already)[0]:
        if rank == 0:
            done = read_json(folder / "completed.json")
            if done != {"step": step, "algorithm": "route-sft", "contract": contract}:
                raise ValueError("Existing Route-SFT export belongs to an incompatible run.")
            receipt = folder / "export_sha256.json"
            if not receipt.is_file():
                write_json(receipt, sft_manifest(folder))
        barrier()
        return
    replica = load_model(parent, torch.bfloat16, attention="sdpa") if rank == 0 else None
    processor = AutoProcessor.from_pretrained(parent) if rank == 0 else None
    if rank == 0:
        register_marker(processor.tokenizer)
    export_policy(folder, actor, replica, processor, step, rank,
                  metadata={"algorithm": "route-sft", "contract": contract})
    if rank == 0:
        write_json(folder / "export_sha256.json", sft_manifest(folder))
    barrier()


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
        raise ValueError("Route-SFT export does not match the supplied 2B parent and v3 pack; use a new WORK.")
    receipt = read_json(path / "export_sha256.json") if (path / "export_sha256.json").is_file() else {}
    manifest = sft_manifest(path)
    if not receipt or manifest != receipt:
        raise ValueError("Final Route-SFT export SHA-256 does not match its receipt.")
    return {"completed": done, **manifest}


def _loop(work, parent, data, rank, size, device):
    from transformers import get_cosine_schedule_with_warmup

    cfg = recipe()
    type_weight = validate_config(cfg)
    m = pack_manifest(data, cfg)
    parent_id = gather(verify_parent(parent, cfg) if rank == 0 else None)[0]
    rows, batches = load_pack(data, cfg, m)
    # The pack was captured by 8B. Replay its exact IDs only after checking
    # the actual 2B training parent's tokenizer, not just the public base.
    if rank == 0:
        tokenizer = parent_tokenizer(parent)
        verify_tokenizer(rows, tokenizer)
        if type_weight:
            verify_type_tokenizer(tokenizer)
    barrier()
    contract = route_identity(data, m, cfg, parent_id)
    out = work / "route_sft"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "contract.json"
        if path.is_file() and read_json(path) != contract:
            raise ValueError("Existing Route-SFT work directory has a different data/model/config identity.")
        write_json(path, contract)
    barrier()
    total = len(batches) * cfg["epochs"]
    cp = latest(out)
    saved = load_checkpoint(cp) if cp else None
    if saved and saved["contract"] != contract:
        raise ValueError("Saved Route-SFT checkpoint has a different data/model/config identity.")
    start = saved["step"] if saved else 0
    if start not in range(0, total + 1, len(batches)):
        raise ValueError("Expected a complete epoch checkpoint.")
    references = (prepare_reference_cache(out, parent, rows, contract, rank, size, device)
                  if type_weight and start < total else None)
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    actor = make_sharded_actor(parent, device, saved["actor"] if saved else None, scorer_type=RouteScorer)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad),
                                  lr=cfg["lr"], betas=cfg["betas"], weight_decay=cfg["weight_decay"],
                                  foreach=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total), total)
    if saved:
        restore_optimizer(actor, optimizer, scheduler, saved, rank)
        restore_rng(saved["rng"][rank], device)
    del saved
    actor.train()

    # Recover an export if interruption happened after the training state commit.
    if start:
        run_epoch_export(out, start, actor, parent, rank, contract)
    for index in range(start, total):
        batch = batches[index % len(batches)]
        stats, norm = update(actor, optimizer, rows, batch, cfg, rank, size, device, references)
        scheduler.step()
        step = index + 1
        stats = gather(stats)
        if rank == 0:
            flat = [x for local in stats for x in local]
            metrics = {"step": step, "epoch": (index // len(batches)) + 1,
                       "lr": scheduler.get_last_lr()[0], "grad_norm": norm,
                       "loss": fmean(x[1] for x in flat), "suffix_ce": fmean(x[2] for x in flat),
                       "route_bce": fmean(x[3] for x in flat),
                       "good_p_rethink": fmean(x[4] for x in flat if x[0] == 0),
                       "bad_p_rethink": fmean(x[4] for x in flat if x[0] == 1)}
            if type_weight:
                good_stats = [x for x in flat if x[0] == 0]
                metrics.update({"good_type_kl": fmean(x[5] for x in good_stats),
                                "weighted_type_kl": type_weight * fmean(x[5] for x in good_stats),
                                "good_student_p_answer": fmean(x[6] for x in good_stats),
                                "good_reference_p_answer": fmean(x[7] for x in good_stats)})
            write_json(out / "metrics" / f"step-{step:05d}.json", metrics)
            if step == 1 or step % 5 == 0 or step % len(batches) == 0:
                print(metrics, flush=True)
        if step % len(batches) == 0:
            save_checkpoint(out, actor, optimizer, scheduler, step, contract, rank, device, keep=2, verify=True)
            run_epoch_export(out, step, actor, parent, rank, contract)
    if rank == 0:
        print(f"Route-SFT complete at {out / f'hf-step-{total:05d}'}", flush=True)
    barrier()


def worker(work, parent, data):
    rank, size, device = initialize()
    try:
        _loop(work, parent, data, rank, size, device)
    finally:
        dist.destroy_process_group()
