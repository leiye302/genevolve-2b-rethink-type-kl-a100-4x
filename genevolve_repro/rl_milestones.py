"""Fixed training budget, evaluation cadence, and restart-safe handoffs."""
from hashlib import sha256
from contextlib import contextmanager
import json
import os
import subprocess
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from .common import ROOT, read_json
from .sft_export import file_sha256, sft_manifest

TOTAL = 40
STOPS = (10, 20, 30, 40)
ALGORITHM = "genevolve-grpo+sdl+actor-routed-hints"
STORAGE = "continuous-v2-weights-only-10-20-30-final-resumable-40"


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def policy_path(work, step):
    return work / "rethink" / ("final" if step == TOTAL else f"hf-step-{step:05d}")


def export_meta(contract, state_digest=None):
    meta = {"algorithm": ALGORITHM, "rl_total_steps": TOTAL, "train_contract_sha256": digest(contract),
            "artifact": "weights_only" if state_digest is None else "final_weights"}
    if state_digest is not None:
        meta["resumable_state_sha256"] = state_digest
    return meta


def require_restartable(work):
    out = work / "rethink"
    if not out.exists() or not any(out.iterdir()):
        from .evaluate import eval_folder
        if any(eval_folder(work, policy_path(work, step), "bench", n).exists()
               for step, n in [*((s, 8) for s in STOPS), (TOTAL, None)]):
            raise ValueError("Stale RL evaluations without their training lineage: restart in a fresh WORK.")
        return
    states = {p.name for p in out.glob("checkpoint-*")}
    if (states != {"checkpoint-00040"} or not (out / "contract.json").is_file()
            or read_json(out / "contract.json").get("storage") != STORAGE
            or not (out / "checkpoint-00040" / "state_sha256.json").is_file()
            or not (out / "checkpoint-00040" / "training.pt").is_file()):
        raise ValueError("No complete v2 step-40 resumable state. Steps 10/20/30 are weights only. "
                         "Preserve this run; restart RL from its Route-SFT parent in a fresh WORK, not from milestone weights.")


def verified_policy(work, sft_identity, step):
    out = work / "rethink"
    policy = policy_path(work, step)
    if not policy.exists():
        return False
    contract = read_json(out / "contract.json")
    if (contract.get("sft") != sft_identity or contract.get("total_steps") != TOTAL
            or contract.get("algorithm") != ALGORITHM or contract.get("world_size") != 4
            or contract.get("milestones") != list(STOPS) or contract.get("storage") != STORAGE):
        raise ValueError("RL milestone belongs to a different SFT/run.")
    cp = out / f"checkpoint-{step:05d}"
    state = None
    if step == TOTAL:
        state = read_json(cp / "state_sha256.json")["training.pt"]
        if state != file_sha256(cp / "training.pt"):
            raise ValueError("RL40 resumable state failed verification.")
    elif cp.exists():
        raise ValueError("Unexpected intermediate optimizer state in weights-only run.")
    elif any(p.is_file() and (p.suffix in {".pt", ".pth", ".bin"} or p.name == "trainer_state.json")
             for p in policy.rglob("*")):
        raise ValueError("Unexpected training state inside weights-only HF export.")
    if (read_json(policy / "completed.json") != {"step": step, **export_meta(contract, state)}
            or read_json(policy / "export_sha256.json") != sft_manifest(policy)):
        raise ValueError("RL milestone HF state is incomplete or mismatched; refusing to evaluate it.")
    return True


def run_bench_eval(work, policy, max_cases, evaluate):
    from .evaluate import bench_complete
    if max_cases not in (8, None):
        raise ValueError("Expected the fixed eight cases or full 594-case benchmark.")
    if not bench_complete(work, policy, max_cases):
        evaluate(policy, max_cases)
        if not bench_complete(work, policy, max_cases):
            raise ValueError("Benchmark worker exited without a valid complete case ledger.")


def run_schedule(work, sft_identity, train_once, evaluate):
    """Keep one live trainer for 1..40. No intermediate training-state files."""
    from .evaluate import bench_complete
    require_restartable(work)
    completed = verified_policy(work, sft_identity, TOTAL)
    if not completed:
        train_once()
    for step in STOPS:
        if not (completed and step == TOTAL) and not verified_policy(work, sft_identity, step):
            raise ValueError(f"Missing verified HF export for RL step {step}.")
        if step < TOTAL and not bench_complete(work, policy_path(work, step), 8):
            raise ValueError(f"Missing/inconsistent eight-case ledger for RL step {step}; cannot certify this run.")
    run_bench_eval(work, policy_path(work, TOTAL), 8, evaluate)
    run_bench_eval(work, policy_path(work, TOTAL), None, evaluate)


def control_group():
    # Non-leaders may wait for slow online image services. Do not enqueue a
    # pending CUDA/NCCL collective with its short watchdog during evaluation.
    return dist.new_group(backend="gloo", timeout=timedelta(days=2))


@contextmanager
def protect_training_rng(device):
    # RAM only. Include NumPy as well as Python and this rank's CPU/CUDA RNG;
    # model serialization and any diagnostic callbacks cannot advance training.
    import numpy as np
    from .distributed import get_rng, restore_rng
    tensors, numpy = get_rng(device), np.random.get_state()
    try:
        yield
    finally:
        restore_rng(tensors, device)
        np.random.set_state(numpy)


def evaluation_child(work, policy, count):
    """A new Python process isolates Python/NumPy/Torch/CUDA RNG and model state."""
    if count != 8:
        raise ValueError("Only fixed-eight probes run while the RL trainer lives.")
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if not visible:
        raise ValueError("Missing an assigned GPU for the evaluation Actor.")
    for name in list(env):
        if name.startswith("TORCHELASTIC_") or name in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}:
            env.pop(name)
    env.update(CUDA_VISIBLE_DEVICES=visible, IMAGE_DOWNLOAD_DIR=str(work / "cache/eval-rank0"))
    subprocess.run([sys.executable, "-m", "genevolve_repro.evaluate", "--work", str(work),
                    "--policy", str(policy)], cwd=ROOT, env=env, check=True)


def live_fixed8(work, policy, rank, group):
    torch.cuda.empty_cache()
    result = [""]
    if rank == 0:
        try:
            free, _ = torch.cuda.mem_get_info()
            if free < 32 * 2**30:
                print(f"RL probe needs 32 GiB free on trainer GPU 0; currently free {free / 2**30:.1f} GiB.", flush=True)
                raise RuntimeError("Insufficient remaining GPU memory for isolated evaluator.")
            run_bench_eval(work, policy, 8, lambda p, n: evaluation_child(work, p, n))
        except Exception as exc:
            # Do not broadcast potentially private exception messages from services.
            result[0] = type(exc).__name__
            print(f"Fixed-eight evaluation failed at {policy.name}: {result[0]}. Inspect the evaluator output.", flush=True)
    dist.broadcast_object_list(result, src=0, group=group)
    if result[0]:
        raise RuntimeError(f"RL stopped before the next update: fixed-eight evaluation failed ({result[0]}).")
