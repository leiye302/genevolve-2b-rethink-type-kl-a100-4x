import argparse
import json
from pathlib import Path
import subprocess
import sys

from genevolve_repro.common import ROOT, WORLD_SIZE, load_upstream, read_json, required_env
from genevolve_repro.sft_export import export_files, file_sha256, sft_manifest


def distributed(subcommand, work, *arguments):
    # RL keeps its native four ranks. Evaluation retains one Actor process so
    # the fixed-eight rollout order and per-case seed stream do not change.
    if subcommand == "_route-sft":
        from genevolve_repro.route_data import recipe
        workers = recipe()["world_size"]
        if workers != 4:
            raise ValueError("The published Route-SFT launcher requires four A100 80GB GPUs.")
    else:
        workers = 1 if subcommand == "_eval" else WORLD_SIZE
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={workers}", str(ROOT / "run.py"),
        "--work", str(work), subcommand, *arguments,
    ], cwd=ROOT, check=True)


def default_eval_cases():
    return read_json(ROOT / "configs" / "evaluation.json")["max_cases"]


def final_route_export(work, cfg=None):
    from genevolve_repro.route_data import recipe

    cfg = recipe() if cfg is None else cfg
    return work / "route_sft" / f"hf-step-{cfg['batches_per_epoch'] * cfg['epochs']:05d}"


def evaluation_arguments(args):
    result = ["--split", args.split]
    if args.policy is not None:
        result.extend(["--policy", str(args.policy.resolve())])
    if args.max_cases is not None:
        result.extend(["--max-cases", str(args.max_cases)])
    else:
        result.append("--full-benchmark")
    return result


def eval_flags(parser):
    parser.add_argument("--split", choices=["bench", "rl-val", "wise"], default="bench")
    parser.add_argument("--policy", type=Path, help="HF export (default: final 562-step Route-SFT export)")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--max-cases", type=int, default=default_eval_cases(),
                       help="Fixed prefix of the pinned split (default: its first 8 cases)")
    scope.add_argument("--full-benchmark", dest="max_cases", action="store_const", const=None,
                       default=argparse.SUPPRESS, help="Explicitly evaluate the entire selected split")


def check_rethink_services():
    from genevolve_repro.hints import settings_from_env

    required_env("SERPER_API_KEY")
    required_env("JUDGE_API_KEY")
    settings_from_env()


def validate_steps(steps):
    if steps is not None and steps < 1:
        raise ValueError("--steps must be positive.")


def route_data_folder(work, given):
    if given is not None:
        return given.resolve()
    from genevolve_repro.route_release import prepare_route
    return prepare_route(work)


def ensure_bench(work):
    from genevolve_repro.data import prepare, verify_data
    # Never replace a corrupt / mismatched split with a different local benchmark.
    try:
        verify_data(work)
    except (ValueError, FileNotFoundError):
        prepare(work)


def dispatch_bench(work, policy, max_cases):
    execute(argparse.Namespace(command="eval", policy=policy, split="bench", max_cases=max_cases), work)


def official_steps(value):
    if value not in (None, 40):
        raise ValueError("The published RETHINK experiment runs exactly 40 RL updates; --steps must be 40.")
    return 40


def run_pipeline(args, work):
    """One job lock; resume completed stages and evaluations from their sealed artifacts."""
    from genevolve_repro.native_sft import run_native
    from genevolve_repro.route_data import pack_manifest, recipe, route_identity, verify_parent
    from genevolve_repro.route_sft import verify_route_export
    from genevolve_repro.rl_milestones import run_bench_eval
    from genevolve_repro.runtime import preflight

    official_steps(args.steps)
    check_rethink_services()
    load_upstream()
    ensure_bench(work)
    cfg = recipe()
    # Optional explicit reuse; by default train from the pinned bare public base.
    parent = args.parent.resolve() if args.parent else run_native(work)
    parent_id = verify_parent(parent, cfg)
    evaluate = lambda policy, count: dispatch_bench(work, policy, count)
    run_bench_eval(work, parent, default_eval_cases(), evaluate)
    data = route_data_folder(work, args.data)
    pack = pack_manifest(data, cfg)
    contract = route_identity(data, pack, cfg, parent_id)
    total = cfg["batches_per_epoch"] * cfg["epochs"]
    route_export = final_route_export(work, cfg)
    route_ready = all((route_export / name).is_file() for name in ("completed.json", "export_sha256.json"))
    if not route_ready:
        preflight("route-sft", work)
        print(f"Route-SFT: {total} updates from the verified native SFT-550", flush=True)
        distributed("_route-sft", work, "--parent", str(parent), "--data", str(data))
    route_id = verify_route_export(route_export, expected_contract=contract)
    run_bench_eval(work, route_export, default_eval_cases(), evaluate)
    run_rethink_40(work, route_export, route_id)


def run_rethink_40(work, route_export, route_id):
    from genevolve_repro.data import embedding_ready, prepare_embedding
    from genevolve_repro.rl_milestones import run_schedule
    from genevolve_repro.runtime import check_milestone_space, preflight

    if not embedding_ready(work):
        prepare_embedding(work)
    def train_once():
        check_milestone_space(work, route_export)
        preflight("rethink", work)
        distributed("_rethink", work, "--sft-checkpoint", str(route_export), "--steps", "40",
                    "--milestone-eval")
    run_schedule(work, route_id, train_once, lambda policy, count: dispatch_bench(work, policy, count))


def execute(args, work):
    if args.command == "all":
        run_pipeline(args, work)
        return
    if args.command in {"prepare-native", "native-sft"}:
        from genevolve_repro.native_sft import prepare_native, run_native
        if args.command == "prepare-native":
            prepare_native(work)
            return
        parent = run_native(work)
        if args.eval:
            from genevolve_repro.rl_milestones import run_bench_eval
            ensure_bench(work)
            run_bench_eval(work, parent, default_eval_cases(), lambda policy, n: dispatch_bench(work, policy, n))
        return
    if args.command in {"grpo", "rethink"}:
        export_files(args.sft_checkpoint)
        validate_steps(args.steps)
    if args.command in {"grpo", "eval"}:
        keys = ("WISE_API_KEY",) if getattr(args, "split", None) == "wise" else ("JUDGE_API_KEY",)
        for key in ("SERPER_API_KEY", *keys):
            required_env(key)
    if args.command == "rethink":
        from genevolve_repro.route_sft import verify_route_export

        official_steps(args.steps)
        check_rethink_services()
        ensure_bench(work)
        policy = args.sft_checkpoint.resolve()
        run_rethink_40(work, policy, verify_route_export(policy))
        return
    if args.command == "prepare":
        from genevolve_repro.data import prepare
        prepare(work)
    if args.command == "prepare-route":
        from genevolve_repro.route_data import load_pack, recipe

        folder = route_data_folder(work, None)
        rows, batches = load_pack(folder, recipe())
        print(json.dumps({"route_data": str(folder), "rows": len(rows), "batches": len(batches),
                          "max_tokens": max(len(row["input_ids"]) for row in rows), "verified": True}, indent=2))
    if args.command == "prepare-embedding":
        from genevolve_repro.data import prepare_embedding
        prepare_embedding(work)
    if args.command == "route-sft":
        from genevolve_repro.route_data import pack_manifest, recipe, verify_parent
        from genevolve_repro.route_sft import verify_route_export
        from genevolve_repro.runtime import preflight

        cfg = recipe()
        parent = args.parent.resolve()
        verify_parent(parent, cfg)
        data = route_data_folder(work, args.data)
        pack_manifest(data, cfg)
        evaluate_final = getattr(args, "eval", False)
        if evaluate_final:
            # Resolve all required bench data *before* the long training job.
            from genevolve_repro.data import prepare, verify_data
            required_env("SERPER_API_KEY")
            required_env("JUDGE_API_KEY")
            try:
                verify_data(work)
            except (ValueError, FileNotFoundError):
                prepare(work)
        preflight("route-sft", work)
        distributed("_route-sft", work, "--parent", str(parent), "--data", str(data))
        if evaluate_final:
            export = final_route_export(work, cfg)
            verify_route_export(export)
            execute(argparse.Namespace(command="eval", policy=export, split="bench",
                                       max_cases=default_eval_cases()), work)
    if args.command == "grpo":
        from genevolve_repro.runtime import preflight

        preflight(args.command, work)
        flags = ["--sft-checkpoint", str(args.sft_checkpoint.resolve())]
        if args.steps is not None:
            flags.extend(["--steps", str(args.steps)])
        distributed("_" + args.command, work, *flags)
    if args.command == "eval":
        if args.policy is None:
            args.policy = final_route_export(work)
        export_files(args.policy.resolve())
        distributed("_eval", work, *evaluation_arguments(args))


def sft_flag(parser):
    parser.add_argument("--sft-checkpoint", required=True, type=Path,
                        help="Qwen3-VL-2B HF export; Route-SFT final for rethink, baseline SFT-550 for GRPO")


def route_flags(parser, *, internal=False, from_base=False):
    parser.add_argument("--data", required=internal, type=Path,
                        help="verified 281-batch Route-SFT pack (default: use the pinned public release)")
    parser.add_argument("--parent", required=not from_base, type=Path, help="verified GenEvolve SFT-550; omit on all to train it from base")


def main():
    parser = argparse.ArgumentParser(description="4×A100 GenEvolve 2B: native SFT-550 → Route-SFT-562 → RETHINK RL-40")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="download pinned RL and benchmark splits")
    commands.add_parser("prepare-route", help="download and check the public Route-SFT pack (CPU only)")
    commands.add_parser("prepare-embedding", help="download GenEvolve's pinned experience-retrieval model")
    commands.add_parser("prepare-native", help="prepare the original SFT data and bare 2B base")
    commands.add_parser("native-sft", help="train or reuse the completed native SFT-550").add_argument(
        "--eval", action="store_true", help="also evaluate its final checkpoint on fixed 8")
    sft_flag(commands.add_parser("check-sft", help="check and fingerprint an existing SFT export"))
    pipeline = commands.add_parser("all", help="all three stages, six fixed-8 evaluations and a final full benchmark")
    route_flags(pipeline, from_base=True)
    pipeline.add_argument("--steps", type=int, choices=[40], default=40, help="published RL horizon (40)")
    pipeline.add_argument("--eval", action="store_true", help="accepted for compatibility; all scheduled evaluations are mandatory")
    for name in ("check-route", "route-sft", "_route-sft"):
        parser_stage = commands.add_parser(name)
        route_flags(parser_stage, internal=name == "_route-sft")
        if name == "route-sft":
            parser_stage.add_argument("--eval", action="store_true",
                                      help="after update 562 ONLY: evaluate the final checkpoint on the fixed eight cases")
    for name in ("grpo", "_grpo", "rethink", "_rethink"):
        stage = commands.add_parser(name)
        sft_flag(stage)
        if name in {"rethink", "_rethink"}:
            stage.add_argument("--steps", type=int, choices=[40], default=40, help="published RETHINK RL horizon")
        else:
            stage.add_argument("--steps", type=int, help="baseline only: RL updates (default: one pass over the train split)")
        if name == "_rethink":
            stage.add_argument("--milestone-eval", action="store_true", required=True)
    for name in ("eval", "_eval"):
        eval_flags(commands.add_parser(name))
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "check-sft":
        try:
            print(json.dumps(sft_manifest(args.sft_checkpoint.resolve()), indent=2))
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "check-route":
        from genevolve_repro.route_data import (
            load_pack, pack_manifest, parent_tokenizer, recipe, verify_parent, verify_tokenizer,
        )

        cfg = recipe()
        try:
            parent = verify_parent(args.parent.resolve(), cfg)
            data = route_data_folder(work, args.data)
            m = pack_manifest(data, cfg)
            rows, batches = load_pack(data, cfg, m)
            verify_tokenizer(rows, parent_tokenizer(args.parent.resolve()))
            print(json.dumps({"parent": parent, "trajectory_source": "8B", "route_data": str(data),
                              "route_data_variant": m.get("release_variant", m["schema"]),
                              "route_manifest_sha256": file_sha256(data / "manifest.json"),
                              "rows": len(rows), "batches": len(batches), "updates": len(batches) * cfg["epochs"],
                              "max_tokens": max(len(row["input_ids"]) for row in rows),
                              "all_online_prefixes_matched": True}, indent=2))
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    elif args.command == "_route-sft":
        from genevolve_repro.route_sft import worker
        worker(work, args.parent.resolve(), args.data.resolve())
    elif args.command in {"_grpo", "_rethink"}:
        from genevolve_repro.evolve import worker
        worker(work, args.sft_checkpoint.resolve(), args.steps, rethink=args.command == "_rethink",
               milestone_eval=getattr(args, "milestone_eval", False))
    elif args.command == "_eval":
        from genevolve_repro.evaluate import worker
        policy = args.policy.resolve() if args.policy else final_route_export(work)
        worker(work, policy, args.split, args.max_cases)
    else:
        from filelock import FileLock, Timeout
        work.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(work / ".run.lock", timeout=0):
                execute(args, work)
        except Timeout as error:
            raise SystemExit(f"Another run is using {work}.") from error
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
