"""Add Actor-routed local hints to the native GenEvolve experience teacher."""

from collections import Counter
import json
import re

import torch

from . import hints
from .common import ROOT, read_json
from .objectives import choose_distillation_tokens
from .policy import score_turn, teacher_prefix, tokenize_history
from .route_data import MARKER, MARKER_ID

NATIVE_ROUTE = re.compile(
    r"<think>\s*(?P<reason>.+?)\s*</think>\s*<RETHINK>\s*"
    r"<(?P<action>tool_call|answer)>\s*(?P<payload>\{.*\})\s*</(?P=action)>", re.DOTALL,
)
IMAGE_ID = re.compile(r"\bIMG_\d+\b")


def recipe():
    return read_json(ROOT / "configs" / "rethink.json")


def verify_marker(tokenizer):
    if (tokenizer.encode(MARKER, add_special_tokens=False) != [MARKER_ID]
            or MARKER_ID not in tokenizer.all_special_ids):
        raise ValueError("Expected the Stage-1 registered single <RETHINK> ID.")


def is_routed(turn):
    text = turn.response or ""
    match = NATIVE_ROUTE.fullmatch(text.strip())
    if not (turn.tokens.count(MARKER_ID) == 1 and text.count(MARKER) == 1 and match is not None
            and bool(match["reason"].strip())
            and all(tag not in match["reason"] for tag in ("<think>", "</think>", "<tool_call>", "<answer>"))):
        return False
    try:
        json.loads(match["payload"])
    except ValueError:
        return False
    return True


def plan_hints(groups, cfg, settings, client=None):
    stats = Counter()
    jobs = []
    for group in groups:
        for trajectory in group.trajectories:
            selected = []
            for n, turn in enumerate(trajectory.turns, 1):
                stats["turns"] += 1
                if MARKER_ID in turn.tokens:
                    stats["generated_marker"] += 1
                    if is_routed(turn):
                        stats["native_boundary"] += 1
                    # As in ALFWorld, only the Actor's token decides routing.
                    selected.append(n)
            if selected:
                stats["eligible"] += len(selected)
                stats["hint_trajectories"] += 1
                jobs.append((trajectory, selected))
    client = client or hints.request_trajectories
    incoming = {}
    if jobs:
        for (trajectory, selected), (answers, calls) in zip(jobs, client(settings, cfg, jobs), strict=True):
            stats["api_attempts"] += calls
            for n in selected:
                if n in answers:
                    incoming[id(trajectory.turns[n - 1])] = answers[n]
    return incoming, stats


def preferred_prefix(tokenizer, turn, guidance, hint, context, hint_max_tokens):
    """Preserve the exact native Teacher unless a local hint can be added."""
    if hint is not None:
        ids = tokenizer.encode(hint, add_special_tokens=False)
        observed = set().union(*(set(IMAGE_ID.findall(m["content"])) for m in turn.messages
                                 if m["role"] == "user" and m["content"].startswith("<tool_response>")))
        if (0 < len(ids) <= hint_max_tokens and not set(ids) & set(tokenizer.all_special_ids)
                and set(IMAGE_ID.findall(hint)) <= observed):
            # Keep retrieved experience byte-for-byte; only add one local block
            # to a copy of this row's native Teacher messages. An empty bank is
            # unchanged for unmarked turns; a marked turn may have a hint alone.
            messages = [dict(message) for message in turn.messages]
            system = messages[0]["content"]
            before, sep, after = system.partition("# Tools")
            if not sep:
                raise ValueError("The pinned agent's system prompt has no # Tools section.")
            messages[0]["content"] = (
                f"{before}\n<local_decision_hint>\n{hint}\n</local_decision_hint>\n\n{sep}{after}"
            )
            prefix = teacher_prefix(tokenizer, messages, guidance) if guidance else tokenize_history(tokenizer, messages)
            if len(prefix) + len(turn.tokens) <= context:
                return prefix, True
    if not guidance:
        return None, False
    prefix = teacher_prefix(tokenizer, turn.messages, guidance)
    if len(prefix) + len(turn.tokens) > context:
        return None, False
    return prefix, False


@torch.no_grad()
def score_groups(groups, scorer, tokenizer, cfg, device, settings, guidance_by_group, client=None):
    r = recipe()
    selected, stats = plan_hints(groups, r, settings, client=client)
    for group, guidance in zip(groups, guidance_by_group, strict=True):
        for trajectory in group.trajectories:
            for turn in trajectory.turns:
                turn.teacher = None
                if not turn.action:
                    # Native SDL scores only turns with a native action span.
                    continue
                hint = selected.get(id(turn))
                if not (guidance or hint):
                    continue
                prefix, local = preferred_prefix(tokenizer, turn, guidance, hint,
                                                 cfg["context_tokens"], r["hint_max_tokens"])
                if prefix is None:
                    continue
                if local:
                    stats["local_used"] += 1
                turn.teacher = score_turn(scorer, prefix, turn.tokens, cfg["temperature"], device).detach().cpu()
                stats["teacher_turns"] += 1
            stats["sdl_tokens"] += choose_distillation_tokens(trajectory.turns, cfg["sdl_top_fraction"])
    stats["native_fallback"] = stats["eligible"] - stats["local_used"]
    return stats
