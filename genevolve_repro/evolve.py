from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
import random
from statistics import fmean

import torch
import torch.distributed as dist

from .actor_text import ACTOR_TEXT_PROTOCOL

from .common import RL, ROOT, load_upstream, model_ready, read_json, required_env, source_lock, write_json
from .data import embedding_ready, examples, verify_split
from .distributed import (
    copy_actor, export_policy, gather, initialize, latest, load_checkpoint, load_model,
    make_sharded_actor, max_count, restore_optimizer, restore_rng, save_checkpoint,
)
from .feedback import LiveFeedback, contrast_pair, judge_settings
from .memory import Embedder, ExperienceBank
from .objectives import advantages, grpo_loss, kl_reward, loss_terms
from .policy import ActorScorer, make_agent, score_turn
from .route_data import verify_parent
from .route_sft import verify_route_export
from .sft_export import file_sha256
from .rl_milestones import STOPS, STORAGE, TOTAL, control_group, export_meta, live_fixed8, policy_path, protect_training_rng, require_restartable
from . import teacher


@dataclass
class Group:
    example: object
    trajectories: list
    outcomes: list


def gather_group(agent, feedback, example, folder, count):
    trajectories = [agent.collect(example.prompt) for _ in range(count)]
    with ThreadPoolExecutor(max_workers=min(count, 4)) as pool:
        pending = [
            pool.submit(feedback.reward, example, trajectory, folder / f"{i:02d}.png")
            for i, trajectory in enumerate(trajectories)
        ]
        outcomes = [task.result() for task in pending]
    return Group(example, trajectories, outcomes)


@torch.no_grad()
def score_group(group, scorer, cfg, device):
    for trajectory in group.trajectories:
        for turn in trajectory.turns:
            turn.old = score_turn(scorer, turn.prefix, turn.tokens, cfg["temperature"], device).cpu()


@torch.no_grad()
def score_reference(groups, scorer, cfg, device):
    for group in groups:
        for trajectory in group.trajectories:
            for turn in trajectory.turns:
                turn.reference = score_turn(scorer, turn.prefix, turn.tokens, cfg["temperature"], device).cpu()


def update_actor(actor, optimizer, groups, cfg, device, dummy_id, *, use_sdl=False):
    # Every rank owns the same number of whole prompt groups. FSDP averages
    # gradients across ranks, so the local sequence denominator is sufficient.
    normalizer = sum(len(group.trajectories) for group in groups)
    jobs = []
    for group in groups:
        rewards = [
            kl_reward(outcome.reward, trajectory.turns, cfg["kl_coef"])
            for trajectory, outcome in zip(group.trajectories, group.outcomes, strict=True)
        ]
        for trajectory, advantage in zip(group.trajectories, advantages(rewards), strict=True):
            jobs.extend((turn, advantage) for turn in trajectory.turns)
    count = max_count(len(jobs), device)
    if count == 0:
        raise RuntimeError("All trajectories were empty. Check the model and generation budgets.")
    optimizer.zero_grad(set_to_none=True)
    grpo_sum = sdl_sum = 0.0
    for i in range(count):
        # Offloaded FSDP needs no_sync for accumulation. A short dummy turn
        # keeps collective order identical when trajectories end at different times.
        context = actor.no_sync() if hasattr(actor, "no_sync") and i + 1 < count else nullcontext()
        with context:
            if i < len(jobs):
                turn, advantage = jobs[i]
                new = score_turn(actor, turn.prefix, turn.tokens, cfg["temperature"], device)
                if use_sdl:
                    grpo, sdl = loss_terms(new, turn, advantage, cfg)
                    loss = (grpo + cfg["sdl_coef"] * sdl) / normalizer
                    sdl_sum += sdl.detach().item()
                else:
                    grpo = grpo_loss(new, turn, advantage, cfg)
                    loss = grpo / normalizer
                grpo_sum += grpo.detach().item()
            else:
                loss = score_turn(actor, [dummy_id], [dummy_id], cfg["temperature"], device).sum() * 0
            loss.backward()
    clip = getattr(actor, "clip_grad_norm_", None)
    norm = clip(1.0) if clip else torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    if not torch.isfinite(norm).item():
        raise RuntimeError("Non-finite actor gradient. The previous checkpoint is intact.")
    optimizer.step()
    if use_sdl:
        return grpo_sum / normalizer, float(norm), sdl_sum / normalizer
    return grpo_sum / normalizer, float(norm)


def write_rollouts(folder, group, use_sdl=False):
    write_json(folder / "rollouts.json", {
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL,
        "id": group.example.id,
        "prompt": group.example.prompt,
        "trajectories": [
            {"result": trajectory.result.to_dict(), "outcome": outcome.to_dict(),
             "sampled_tokens": sum(len(turn.tokens) for turn in trajectory.turns),
             **({"sdl_tokens": sum(int(turn.distill.sum()) for turn in trajectory.turns
                                   if turn.distill is not None)} if use_sdl else {})}
            for trajectory, outcome in zip(group.trajectories, group.outcomes, strict=True)
        ],
    })


def _loop(work, sft, rank, size, device, limit, *, rethink=False, milestone_eval=False):
    from transformers import AutoProcessor, AutoTokenizer

    load_upstream()
    required_env("SERPER_API_KEY")
    if rethink:
        from .hints import settings_from_env

        hint_settings = settings_from_env()
        hint_cfg = teacher.recipe()
        if not embedding_ready(work):
            raise RuntimeError("Missing the pinned embedding model; run prepare-embedding before rethink.")
    cfg = read_json(ROOT / "configs" / "evolve.json")
    if rethink:
        cfg.update(read_json(ROOT / "configs" / "genevolve_sdl.json"))
    if cfg["prompt_batch"] % size:
        raise ValueError("prompt_batch must split into whole groups across GPU ranks.")
    dataset = work / "data" / RL
    pool = examples(dataset, "train")
    if len(pool) != 2575:
        raise RuntimeError("The pinned RL train split must contain 2,575 prompts.")
    random.Random(cfg["seed"]).shuffle(pool)
    total = limit if limit is not None else math.ceil(len(pool) / cfg["prompt_batch"])
    if milestone_eval and (not rethink or total != TOTAL or size != 4):
        raise ValueError("The continuous 10/20/30/40 protocol requires four-rank rethink --steps 40.")
    if total < 1:
        raise ValueError("At least one RL update is required.")
    out = work / ("rethink" if rethink else "grpo")
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final"
    # Only one rank reads the several-GB SFT weight files for provenance.
    inputs = gather({
        "sft": verify_route_export(sft) if rethink else verify_parent(sft),
        "train_jsonl_sha256": verify_split(dataset, "train"),
    } if rank == 0 else None)[0]
    contract = {"algorithm": "genevolve-grpo+sdl+actor-routed-hints" if rethink else "grpo",
                "sources": source_lock(), **inputs,
                "config": cfg, "judge": judge_settings(cfg), "world_size": size, "total_steps": total}
    if rethink:
        contract.update(actor_text_protocol=ACTOR_TEXT_PROTOCOL, local_hints=hint_cfg,
                        hint_api=hint_settings.identity(),
                        teacher_text_sha256={
                            name: file_sha256(ROOT / "configs" / name)
                            for name in ("local_hint_system.txt",)
                        })
    if milestone_eval:
        contract.update(milestones=list(STOPS), storage=STORAGE)
        # Rank zero checks the pre-existing directory before anyone writes into
        # it. Only a finalized step-40 state may resume in this work directory.
        if rank == 0:
            require_restartable(work)
        gather(rank)
    if (out / "contract.json").is_file() and read_json(out / "contract.json") != contract:
        raise RuntimeError("This work directory has a different RL, SFT, or training-data identity.")
    if rank == 0:
        write_json(out / "contract.json", contract)
    if not milestone_eval and model_ready(final) and (final / "completed.json").is_file() and read_json(final / "completed.json")["step"] == total:
        if rank == 0:
            print(f"RL complete at {final}", flush=True)
        return
    checkpoint_path = latest(out)
    saved = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if saved and saved["contract"] != contract:
        raise RuntimeError("Saved checkpoint configuration does not match this run.")
    if rethink:
        if saved and "memory" not in saved:
            raise RuntimeError("Incomplete GenEvolve checkpoint: missing the experience bank.")
        memory = ExperienceBank(cfg["experience_capacity"], saved["memory"] if saved else None)
    start = saved["step"] if saved else 0
    if milestone_eval:
        if (start not in (0, TOTAL) or (saved and
                (len(saved["rng"]) != size or saved["scheduler"]["T_max"] != total
                 or saved["scheduler"]["last_epoch"] != start
                 or checkpoint_path.name != f"checkpoint-{start:05d}"))):
            raise ValueError("Unexpected resume state/step for the 40-update RL schedule.")
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    actor = make_sharded_actor(sft, device, saved["actor"] if saved else None)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad),
                                  lr=cfg["learning_rate"], weight_decay=0.0, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    rng = saved["rng"][rank] if saved else None
    if saved:
        restore_optimizer(actor, optimizer, scheduler, saved, rank)
    del saved
    processor = AutoProcessor.from_pretrained(sft)
    tokenizer = AutoTokenizer.from_pretrained(sft, use_fast=True)
    if rethink:
        teacher.verify_marker(tokenizer)
    replica = load_model(sft, torch.bfloat16).eval()
    reference = load_model(sft, torch.bfloat16).eval()
    os.environ.setdefault("IMAGE_DOWNLOAD_DIR", str(work / "cache" / f"grpo-rank{rank}"))
    feedback = LiveFeedback(cfg, size)
    embedder = Embedder(work / "embedding") if rethink else None
    agent = make_agent(replica, tokenizer, device, cfg, track_actions=rethink)
    actor.train()
    local_batch = cfg["prompt_batch"] // size
    # Only restore after constructing every stateful helper. Initializing a
    # fresh tokenizer/model/embedder on resume must not consume rollout RNG.
    if rng:
        restore_rng(rng, device)

    eval_group = control_group() if milestone_eval and start < total else None
    for step in range(start, total):
        replica.to(device)
        copy_actor(actor, replica)
        scorer = ActorScorer(replica)
        groups, guidance, added, paths = [], [], [], []
        for offset in range(local_batch):
            index = (step * cfg["prompt_batch"] + rank * local_batch + offset) % len(pool)
            example = pool[index]
            path = out / "rollouts" / f"step-{step + 1:05d}" / f"rank-{rank}-{offset}"
            group = gather_group(agent, feedback, example, path, cfg["rollouts_per_prompt"])
            score_group(group, scorer, cfg, device)
            groups.append(group)
            paths.append(path)
            if rethink:
                # The retrieved entry is from earlier updates only. Keep the
                # native contrastive experience-generation path unchanged.
                vector = embedder(example.prompt)
                guidance.append(memory.retrieve(vector))
                pair = contrast_pair(group.trajectories, group.outcomes, cfg["min_reward_gap"])
                if pair is not None:
                    best, worst, gap = pair
                    bundle = feedback.summarize(example, best, worst)
                    added.append({"vector": vector, "gap": gap, "source_id": example.id, "bundle": bundle})

        if rethink:
            # Same current-Actor replica used for rollout; no update has been
            # performed yet. The frozen reference below is ONLY the baseline KL.
            hint_stats = teacher.score_groups(groups, scorer, tokenizer, cfg, device, hint_settings, guidance)
            hint_totals = gather(dict(hint_stats))
        for path, group in zip(paths, groups, strict=True):
            write_rollouts(path, group, use_sdl=rethink)
        replica.cpu()
        torch.cuda.empty_cache()
        reference.to(device)
        score_reference(groups, ActorScorer(reference), cfg, device)
        reference.cpu()
        torch.cuda.empty_cache()
        result = update_actor(actor, optimizer, groups, cfg, device, tokenizer.eos_token_id, use_sdl=rethink)
        grpo, norm = result[:2]
        scheduler.step()
        if rethink:
            memory.update([entry for entries in gather(added) for entry in entries])
        outcomes = [outcome for group in groups for outcome in group.outcomes]
        metrics = gather({
            "reward": fmean(outcome.reward for outcome in outcomes),
            "valid_fraction": fmean(outcome.valid for outcome in outcomes),
            "grpo": grpo, "grad_norm": norm,
            **({"sdl": result[2]} if rethink else {}),
        })
        if rank == 0:
            row = {k: fmean(item[k] for item in metrics) for k in metrics[0]}
            row.update(step=step + 1, lr=scheduler.get_last_lr()[0])
            if rethink:
                row.update(sdl_tokens=sum(s.get("sdl_tokens", 0) for s in hint_totals),
                           experience_entries=len(memory.entries))
                for name in ("turns", "generated_marker", "native_boundary", "eligible", "hint_trajectories", "api_attempts",
                             "local_used", "native_fallback", "teacher_turns"):
                    row[f"rethink/{name}"] = sum(s.get(name, 0) for s in hint_totals)
                row["rethink/trigger_fraction"] = row["rethink/generated_marker"] / max(1, row["rethink/turns"])
            write_json(out / "metrics" / f"step-{step + 1:05d}.json", row)
            print(row, flush=True)
        n = step + 1
        if milestone_eval and n in STOPS[:-1]:
            # Keep the complete trainer alive in THIS process. Only the BF16
            # model/processor is serialized (no temporary optimizer/RNG/bank).
            hf = policy_path(work, n)
            optimizer.zero_grad(set_to_none=True)
            with protect_training_rng(device):
                export_policy(hf, actor, replica, processor, n, rank, metadata=export_meta(contract), seal=True)
                live_fixed8(work, hf, rank, eval_group)
        due = n == total if milestone_eval else (n % cfg["save_every"] == 0 or n == total)
        if due:
            save_checkpoint(out, actor, optimizer, scheduler, step + 1, contract, rank, device,
                            keep=1 if milestone_eval else cfg["keep_checkpoints"],
                            verify=milestone_eval, memory=memory if rethink else None)

    if not milestone_eval:
        export_policy(final, actor, replica, processor, total, rank)
    else:
        hf = policy_path(work, total)
        state = read_json(out / f"checkpoint-{total:05d}" / "state_sha256.json")["training.pt"]
        if not hf.exists():
            export_policy(hf, actor, replica, processor, total, rank, metadata=export_meta(contract, state), seal=True)


def worker(work, sft, limit, *, rethink=False, milestone_eval=False):
    rank, size, device = initialize()
    try:
        _loop(work, sft, rank, size, device, limit, rethink=rethink, milestone_eval=milestone_eval)
    finally:
        dist.destroy_process_group()
