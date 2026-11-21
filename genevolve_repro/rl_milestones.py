"""Fixed training budget, evaluation cadence, and restart-safe handoffs."""
from hashlib import sha256
import json

from .common import read_json
from .sft_export import file_sha256, sft_manifest

TOTAL = 40
STOPS = (10, 20, 30, 40)
ALGORITHM = "genevolve-grpo+sdl+actor-routed-hints"


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def stop_step(total, stop_after, rethink):
    if stop_after is None:
        return total
    if not rethink or total != TOTAL or type(stop_after) is not int or stop_after not in STOPS:
        raise ValueError("--stop-after requires rethink --steps 40 and a milestone 10/20/30/40.")
    return stop_after


def policy_path(work, step):
    return work / "rethink" / ("final" if step == TOTAL else f"hf-step-{step:05d}")


def export_meta(contract, state_digest):
    return {"algorithm": ALGORITHM, "rl_total_steps": TOTAL, "train_contract_sha256": digest(contract),
            "resumable_state_sha256": state_digest}


def verified_policy(work, sft_identity, step):
    out = work / "rethink"
    policy = policy_path(work, step)
    if not policy.exists():
        return False
    contract = read_json(out / "contract.json")
    if (contract.get("sft") != sft_identity or contract.get("total_steps") != TOTAL
            or contract.get("algorithm") != ALGORITHM or contract.get("world_size") != 4
            or contract.get("milestones") != list(STOPS)):
        raise ValueError("RL milestone belongs to a different SFT/run.")
    cp = out / f"checkpoint-{step:05d}"
    state = read_json(cp / "state_sha256.json")["training.pt"]
    # The coordinator calls this once per existing milestone, outside GPU training.
    if (not (cp / "training.pt").is_file() or (cp / "training.pt").stat().st_size == 0
            or state != file_sha256(cp / "training.pt")
            or read_json(policy / "completed.json") != {"step": step, **export_meta(contract, state)}
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


def run_schedule(work, sft_identity, train_to, evaluate):
    """Callbacks run under the caller's one work lock; workers never overlap."""
    for step in STOPS:
        if not verified_policy(work, sft_identity, step):
            train_to(step)
            if not verified_policy(work, sft_identity, step):
                raise ValueError(f"Missing verified HF export for RL step {step}.")
        run_bench_eval(work, policy_path(work, step), 8, evaluate)
    run_bench_eval(work, policy_path(work, TOTAL), None, evaluate)
