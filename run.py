import argparse
import json
from pathlib import Path
import subprocess
import sys

from genevolve_repro.common import ROOT, WORLD_SIZE, load_upstream, read_json, required_env
from genevolve_repro.sft_export import export_files, file_sha256, sft_manifest


def distributed(subcommand, work, *arguments):
    # Stage one and its matched evaluation use the first visible GPU. Stage
    # two continues to see all four assigned GPUs and uses its original ranks.
    workers = 1 if subcommand in {"_route-sft", "_eval"} else WORLD_SIZE
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


def run_pipeline(args, work):
    """Run the two method stages under the caller's single work-directory lock."""
    from genevolve_repro.data import prepare, verify_data
    from genevolve_repro.route_data import pack_manifest, recipe, route_identity, verify_parent
    from genevolve_repro.runtime import preflight

    validate_steps(args.steps)
    check_rethink_services()
    load_upstream()
    cfg = recipe()
    parent = args.parent.resolve()
    parent_id = verify_parent(parent, cfg)
    data = route_data_folder(work, args.data)
    pack = pack_manifest(data, cfg)
    contract = route_identity(data, pack, cfg, parent_id)
    total = cfg["batches_per_epoch"] * cfg["epochs"]
    route_export = final_route_export(work, cfg)
    route_ready = all((route_export / name).is_file() for name in ("completed.json", "export_sha256.json"))
    if not route_ready:
        preflight("route-sft", work)

    # Avoid contacting the dataset host on resume if the pinned data and
    # paired reference images are already available in this work directory.
    try:
        verify_data(work)
    except (ValueError, FileNotFoundError):
        prepare(work)

    if not route_ready:
        print(f"Stage 1: Route-SFT ({total} updates)", flush=True)
        distributed("_route-sft", work, "--parent", str(parent), "--data", str(data))
    else:
        print("Stage 1: found existing Route-SFT export; checking handoff", flush=True)
    print("Stage 2: GenEvolve GRPO + native SDL with Actor-routed hints", flush=True)
    # The same handoff as the standalone subcommand also validates the
    # checkpoint on repeat invocations before any stage-2 workers start.
    execute(argparse.Namespace(command="rethink", sft_checkpoint=route_export, steps=args.steps,
                               route_contract=contract), work)
    if args.eval:
        print("Final stage-2 evaluation: fixed 8-case GenEvolve-Bench subset", flush=True)
        execute(argparse.Namespace(command="eval", split="bench", max_cases=default_eval_cases(),
                                   policy=work / "rethink" / "final"), work)


def execute(args, work):
    if args.command == "all":
        run_pipeline(args, work)
        return
    if args.command in {"grpo", "rethink"}:
        export_files(args.sft_checkpoint)
        validate_steps(args.steps)
    if args.command in {"grpo", "eval"}:
        keys = ("WISE_API_KEY",) if getattr(args, "split", None) == "wise" else ("JUDGE_API_KEY",)
        for key in ("SERPER_API_KEY", *keys):
            required_env(key)
    if args.command == "rethink":
        from genevolve_repro.data import embedding_ready, prepare_embedding
        from genevolve_repro.route_sft import verify_route_export

        check_rethink_services()
        verify_route_export(args.sft_checkpoint.resolve(), expected_contract=getattr(args, "route_contract", None))
        if not embedding_ready(work):
            prepare_embedding(work)
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
        from genevolve_repro.route_data import pack_manifest, recipe
        from genevolve_repro.route_sft import verify_route_export
        from genevolve_repro.runtime import preflight

        data = route_data_folder(work, args.data)
        cfg = recipe()
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
        distributed("_route-sft", work, "--parent", str(args.parent.resolve()), "--data", str(data))
        if evaluate_final:
            export = final_route_export(work, cfg)
            verify_route_export(export)
            execute(argparse.Namespace(command="eval", policy=export, split="bench",
                                       max_cases=default_eval_cases()), work)
    if args.command in {"grpo", "rethink"}:
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


def route_flags(parser, *, internal=False):
    parser.add_argument("--data", required=internal, type=Path,
                        help="verified 281-batch Route-SFT pack (default: use the pinned public release)")
    parser.add_argument("--parent", required=True, type=Path, help="matching GenEvolve 2B SFT checkpoint-550 HF export")


def main():
    parser = argparse.ArgumentParser(description="GenEvolve 2B: one-GPU two-epoch Route-SFT, four-GPU native SDL")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="download pinned RL and benchmark splits")
    commands.add_parser("prepare-route", help="download and check the public Route-SFT pack (CPU only)")
    commands.add_parser("prepare-embedding", help="download GenEvolve's pinned experience-retrieval model")
    sft_flag(commands.add_parser("check-sft", help="check and fingerprint an existing SFT export"))
    pipeline = commands.add_parser("all", help="prepare data and run both method stages")
    route_flags(pipeline)
    pipeline.add_argument("--steps", type=int, help="planned stage-2 updates (default: 322, same as the baselines)")
    pipeline.add_argument("--eval", action="store_true", help="also score the final stage-2 policy on the fixed eight cases")
    for name in ("check-route", "route-sft", "_route-sft"):
        parser_stage = commands.add_parser(name)
        route_flags(parser_stage, internal=name == "_route-sft")
        if name == "route-sft":
            parser_stage.add_argument("--eval", action="store_true",
                                      help="after update 562 ONLY: evaluate the final checkpoint on the fixed eight cases")
    for name in ("grpo", "_grpo", "rethink", "_rethink"):
        stage = commands.add_parser(name)
        sft_flag(stage)
        stage.add_argument("--steps", type=int, help="RL updates (default: one pass over the train split)")
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
        worker(work, args.sft_checkpoint.resolve(), args.steps, rethink=args.command == "_rethink")
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
