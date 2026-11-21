"""Optional A100/NCCL preflight: one discarded production update, no file/API writes."""
import argparse
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def longest_batch(rows, batches):
    # Keep a real ordered 6:2 batch intact; never combine outliers into a fictitious batch.
    return max(range(len(batches)), key=lambda b: (
        max(len(rows[k]["input_ids"]) for k in batches[b]),
        sum(len(rows[k]["input_ids"]) for k in batches[b])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="Exact local 2B GenEvolve SFT-550 HF export")
    parser.add_argument("--data", type=Path, required=True, help="Verified public or original 281-batch pack")
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from genevolve_repro.distributed import initialize, load_model, make_ddp_route_actor
    from genevolve_repro.route_data import MARKER_ID, load_pack, recipe, verify_parent
    from genevolve_repro.route_sft import RouteScorer, _rank0_checked, hardware_suggest, update
    from genevolve_repro.route_type_kl import score_threeway

    cfg = recipe()
    rank, size, device = initialize(world_size=4)
    try:
        hardware_suggest(torch, device, size)
        _rank0_checked(rank, size, lambda: verify_parent(args.parent, cfg))
        rows, batches = load_pack(args.data, cfg)
        idx = longest_batch(rows, batches)
        batch = batches[idx]
        # Score only the six Good references needed here, using exactly their pre-action context.
        parent = load_model(args.parent, torch.bfloat16).to(device).eval().requires_grad_(False)
        local = {}
        with torch.no_grad():
            for k in batch[rank::size]:
                row = rows[k]
                if not row["route_label"]:
                    n = row["target_start"]
                    prefix = torch.tensor([row["input_ids"][:n]], device=device)
                    local[k] = score_threeway(parent, prefix, n).cpu().tolist()
        del parent
        gc.collect()
        torch.cuda.empty_cache()
        pieces = [None] * size
        dist.all_gather_object(pieces, local)
        refs = {k: v for piece in pieces for k, v in piece.items()}
        if set(refs) != {k for k in batch if not rows[k]["route_label"]} or len(refs) != 6:
            raise RuntimeError("Incomplete Good references.")
        torch.manual_seed(cfg["seed"])
        actor = make_ddp_route_actor(args.parent, device, None, RouteScorer)
        actor.train()
        optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=cfg["lr"],
                                      betas=cfg["betas"], weight_decay=cfg["weight_decay"], foreach=False)
        marker = actor.module.model.lm_head.weight[MARKER_ID].detach()
        before = marker.clone()
        # Peak recipe LR, not the zero-LR first warmup step: check an actual update, then discard it.
        metrics = update(actor, optimizer, rows, batch, cfg, device, refs)
        delta = (marker - before).abs().max().item()
        peers = [torch.empty_like(marker) for _ in range(size)]
        dist.all_gather(peers, marker.contiguous())
        if not delta > 0 or not metrics["grad_norm"] > 0 or not all(torch.equal(marker, p) for p in peers):
            raise RuntimeError("No real marker-row update or replicas disagree.")
        peak = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30], device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            print({"status": "passed", "ranks": size, "fixed_batch_zero_based": idx,
                   "longest_context_tokens": max(len(rows[k]["input_ids"]) for k in batch),
                   "global_good": len(refs), "global_bad": len(batch) - len(refs),
                   "grad_norm": metrics["grad_norm"], "marker_max_abs_update": delta,
                   "max_peak_allocated_GiB": round(peak.item(), 2),
                   "discarded_updates": 1, "saved_checkpoints": 0}, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
