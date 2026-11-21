import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .actor_text import ACTOR_TEXT_PROTOCOL

from .common import BENCH, RL, ROOT, jsonl, load_upstream, model_ready, read_json, required_env, write_json
from .data import examples, verify_split
from .distributed import barrier, initialize, load_model
from .evaluation_render import EVALUATION_RENDER_POLICY, render_for_evaluation
from .feedback import LiveFeedback, image_scorer, judge_settings
from .policy import make_agent, valid_program
from .route_data import MARKER_ID
from .teacher import is_routed


def evaluate_case(agent, feedback, example, metadata, image_path):
    trajectory = agent.collect(example.prompt)
    result = trajectory.result
    row = {
        **{key: metadata.get(key) or "unknown" for key in ("eval_type", "category", "difficulty")},
        "id": example.id, "prompt": example.prompt, "gt_image": str(example.gt),
        "result": result.to_dict(), "success": False, "eval_success": False,
        "program_valid": valid_program(result),
    }
    turns = getattr(trajectory, "turns", ())
    marked = [t for t in turns if MARKER_ID in t.tokens and t.response and "<RETHINK>" in t.response]
    row["route_telemetry"] = {
        "turns": len(turns), "marker_turns": len(marked),
        "native_boundary_turns": sum(is_routed(t) for t in marked),
    }
    row.update(render_for_evaluation(feedback.renderer, result, example.prompt, image_path))
    if row["success"]:
        row.update(
            eval_success=True,
            scores=feedback.score_image(example, image_path),
        )
    return row


def summarize(folder, rows):
    scorer = image_scorer()
    path = folder / "results_eval.json"
    write_json(path, rows)
    summary = scorer.build_benchmark_summary(path, len(rows))
    write_json(folder / "summary.json", summary)
    scorer._write_csv(str(folder / "summary.csv"), summary)
    return summary


def fixed_eight_metrics(rows, *, image_exists=lambda p: Path(p).is_file()):
    """Validate the eight paired outcomes before considering any failure a zero."""
    ids = [str(row["id"]) for row in rows]
    if ids != [str(i) for i in range(8)]:
        raise ValueError("Expected all eight fixed IDs in order.")
    scores = []
    marked = turns = marked_cases = native = 0
    for row in rows:
        route = row["route_telemetry"]
        t, m, n = (route[k] for k in ("turns", "marker_turns", "native_boundary_turns"))
        if not all(type(v) is int for v in (t, m, n)) or not 0 <= n <= m <= t:
            raise ValueError("Invalid marker telemetry.")
        turns += t
        marked += m
        native += n
        marked_cases += int(m > 0)
        if row["success"]:
            score = (row.get("scores") or {}).get("overall")
            if (not row["eval_success"] or not image_exists(row["image_path"])
                    or type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Successful image is missing its artifact or real Judge score.")
            scores.append(score)
        elif (row["eval_success"] or not row.get("render_fallback")
              or "requires at least one valid reference image" not in row.get("image_status", "")):
            raise ValueError(f"Case {row['id']}: unexpected rendering/service failure; inspect it before reporting any score.")
    if turns == 0:
        raise ValueError("No actor rounds; refusing to report a zero marker rate.")
    return {"case_ids": ids, "cases": 8, "images_scored": len(scores),
            "score_all8_missing_zero": sum(scores) / 8, "score_scored_only": sum(scores) / len(scores) if scores else None,
            "marker_turns": marked, "all_turns": turns, "marker_turn_rate": marked / turns,
            "cases_with_marker": marked_cases, "marker_case_rate": marked_cases / 8,
            "marker_at_native_boundary_turns": native}


def _evaluate(work, policy, split, max_cases, rank, size, device):
    from transformers import AutoTokenizer

    load_upstream()
    required_env("SERPER_API_KEY")
    if not model_ready(policy):
        raise RuntimeError(f"Expected an HF policy export at {policy}.")
    if split == "wise":
        from .wise import WiseFeedback, summarize_wise, wise_examples

        pool, metadata = wise_examples()
        count = 1000
    else:
        data, name = (BENCH, "test") if split == "bench" else (RL, "eval")
        dataset = work / "data" / data
        split_sha = verify_split(dataset, name)
        pool = examples(dataset, name)
        metadata = list(jsonl(dataset / f"{name}.jsonl"))
        count = 594 if split == "bench" else 600
        if len(pool) != count:
            raise ValueError(f"Expected all {count} cases in the pinned {data} split.")
    if max_cases is not None:
        if not 1 <= max_cases <= count:
            raise ValueError(f"--max-cases must be between 1 and {count}.")
        count = max_cases
    selected_ids = [str(pool[i].id) for i in range(count)]
    light = read_json(ROOT / "configs" / "evaluation.json")
    if split == "bench" and count == light["max_cases"] and selected_ids != light["selected_case_ids"]:
        raise ValueError("The fixed eight case IDs/order differ from the pinned evaluation contract.")
    marker = policy / "completed.json"
    suffix = "" if max_cases is None else f"-{max_cases}-cases"
    name = policy.name if policy.name != "final" else f"{policy.parent.name}-final"
    folder = work / "eval" / f"{split}-{name}{suffix}"
    cfg = read_json(ROOT / "configs" / "evolve.json")
    if split != "rl-val":
        cfg.update(temperature=light["temperature"], top_p=light["top_p"])
    if split == "bench" and cfg["seed"] != light["seed"]:
        raise ValueError("Wrong GenEvolve-Bench generation seed.")
    contract = {
        "policy": str(policy), "split": split, "count": count,
        "checkpoint": read_json(marker) if marker.is_file() else None,
        "render_policy": EVALUATION_RENDER_POLICY,
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL,
        "route_telemetry_version": 1,
        "evaluation_scope": "full_benchmark" if max_cases is None else "fixed_subset",
        "selected_case_ids": selected_ids,
        "generation": {"temperature": cfg["temperature"], "top_p": cfg["top_p"],
                       "seed_base": cfg["seed"], "seed_rule": "base + zero-based split index"},
    }
    if split != "wise":
        contract["judge"] = judge_settings(cfg)
        contract["split_sha256"] = split_sha
    lock = folder / "config.json"
    if lock.is_file() and read_json(lock) != contract:
        raise RuntimeError(
            f"Evaluation in {folder} uses a different policy, judge, rendering protocol, or Actor text protocol. "
            "Archive it before rerunning."
        )
    if rank == 0:
        write_json(lock, contract)
    barrier()
    jobs = [(i, folder / "cases" / f"{i:05d}.json") for i in range(rank, count, size)]
    jobs = [(i, path) for i, path in jobs if not path.is_file()]
    if jobs:
        os.environ.setdefault("IMAGE_DOWNLOAD_DIR", str(work / "cache" / f"eval-rank{rank}"))
        torch.manual_seed(cfg["seed"] + rank)
        model = load_model(policy, torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(policy, use_fast=True)
        agent = make_agent(model, tokenizer, device, cfg)
        feedback = WiseFeedback() if split == "wise" else LiveFeedback(cfg, size)
        for i, path in jobs:
            # Make a resumed evaluation independent of which earlier cases exist.
            torch.manual_seed(cfg["seed"] + i)
            image_name = f"{pool[i].id}.png" if split == "wise" else f"{i:05d}.png"
            image = folder / "images" / image_name
            row = (
                feedback.evaluate(agent, pool[i], metadata[i], image) if split == "wise"
                else evaluate_case(agent, feedback, pool[i], metadata[i], image)
            )
            write_json(path, row)
            print(f"{split} case {i + 1}/{count} scored={row['success']}", flush=True)
    barrier()
    if rank == 0:
        rows = [read_json(folder / "cases" / f"{i:05d}.json") for i in range(count)]
        checked = fixed_eight_metrics(rows) if split == "bench" and count == light["max_cases"] else None
        scores = summarize_wise(folder, rows) if split == "wise" else summarize(folder, rows)["all"]
        if checked is not None:
            if (scores["denominator"] != 8 or scores["count_eval_success"] != checked["images_scored"] or
                    abs(scores["overall_missing_zero"] - checked["score_all8_missing_zero"]) > .000051):
                raise ValueError("Official summary and fixed-eight case ledger differ.")
            write_json(folder / "fixed8_metrics.json", checked)
        if split != "wise":
            fields = ("turns", "marker_turns", "native_boundary_turns")
            route = {k: sum(row["route_telemetry"][k] for row in rows) for k in fields}
            route.update(cases=count, decision_trigger_rate=route["marker_turns"] / max(1, route["turns"]),
                         native_boundary_rate_given_marker=(
                             route["native_boundary_turns"] / route["marker_turns"] if route["marker_turns"] else None
                         ),
                         cases_with_marker=sum(row["route_telemetry"]["marker_turns"] > 0 for row in rows))
            route["case_trigger_rate"] = route["cases_with_marker"] / count
            write_json(folder / "route_telemetry.json", route)
        print(scores, flush=True)
        print(f"Benchmark artifacts at {folder}", flush=True)


def worker(work, policy, split, max_cases):
    rank, size, device = initialize(world_size=1)
    try:
        _evaluate(work, policy, split, max_cases, rank, size, device)
    finally:
        dist.destroy_process_group()
