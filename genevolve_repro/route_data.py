"""Load the sealed Route-SFT token pack; never reconstruct or truncate targets."""

from collections import Counter
from pathlib import Path
import re

from .actor_text import ACTOR_TEXT_PROTOCOL, decode_actor_response
from .common import ROOT, jsonl, read_json
from .native_handoff import NATIVE_KEY, RECEIPT, verify_native_identity, verify_native_parent
from .route_release import release_lock
from .sft_export import MODEL_FAMILY, REQUIRED, file_sha256, sft_manifest

MARKER = "<RETHINK>"
MARKER_ID = 151669
EOS_ID = 151645


def recipe():
    return read_json(ROOT / "configs" / "route_sft.json")


def compatible_export_config(actual, expected):
    """A finished HF policy is portable; a resume checkpoint is not.

    Only the number of data-parallel workers may differ between the historical
    single-card run and its four-card counterpart. Do not use this comparison
    for partial checkpoints, mutable optimizer states or the training work dir.
    """
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    for cfg in (actual, expected):
        if type(cfg.get("world_size")) is not int or cfg["world_size"] not in (1, 4):
            return False
    return {**actual, "world_size": expected["world_size"]} == expected


def require(condition, message):
    if not condition:
        raise ValueError(message)


def release_artifacts(digest, cfg):
    """Return the public files for this digest, or None for the original pack."""
    if digest == cfg["manifest_sha256"]:
        return None
    files = release_lock()["files"]
    require(digest == files["manifest.json"]["sha256"], "Route-SFT manifest SHA-256 mismatch.")
    return {name: files[name]["sha256"] for name in ("train.jsonl", "batches.jsonl")}


def verify_artifact_identity(digest, artifacts, cfg):
    public = release_artifacts(digest, cfg)
    if public is None:
        locked = read_json(ROOT / "configs/route_original_v4.json")
        require(digest == locked["manifest_sha256"], "Wrong original v4 manifest.")
        public = {name + ".jsonl": locked[name + "_sha256"] for name in ("train", "batches")}
    require(artifacts == public, "Route-SFT files do not match the pinned SHA-256.")


def pack_manifest(folder, cfg):
    path = Path(folder) / "manifest.json"
    require(path.is_file(), f"Expected manifest.json for the sealed {cfg['schema']} pack.")
    digest = file_sha256(path)
    public = release_artifacts(digest, cfg)
    m = read_json(path)
    for key, expected in {
        "schema": cfg["schema"], "batches": cfg["batches_per_epoch"],
        "good": cfg["batches_per_epoch"] * cfg["good_per_batch"],
        "bad": cfg["batches_per_epoch"] * cfg["bad_per_batch"],
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL, "marker_id": MARKER_ID,
        "eos_id": EOS_ID, "ordered_batches": True,
        "history_truncation": False, "official_split_overlap_exact_normalized": 0,
    }.items():
        require(m.get(key) == expected, f"Unexpected Route-SFT manifest field: {key}.")
    if public is not None:
        require(m.get("release_variant") == "reviewed-v4-public" and m.get("retokenized_input_history") is True
                and m.get("original_manifest_sha256") == cfg["manifest_sha256"]
                and m.get("supervision_identical_to_reviewed_v4") is True
                and m.get("artifact_sha256") == public, "Public Route-SFT provenance differs.")
    require(m["checkpoint550_file_sha256"] == source_checkpoint_lock(),
            "Route-SFT collection provenance does not match the sealed 8B source.")
    verify_artifact_identity(digest, {n: m["artifact_sha256"][n] for n in ("train.jsonl","batches.jsonl")}, cfg)
    return m


def source_checkpoint_lock():
    """The weights that generated the natural examples, not the 2B training start."""
    return read_json(ROOT / "configs" / "route_data_source_8b_sha256.json")


def verify_parent_identity(identity, cfg=None):
    cfg = cfg or recipe()
    require(isinstance(identity, dict)
            and identity.get("model_family") == MODEL_FAMILY
            and identity.get("sft_global_step") == cfg["parent_step"],
            f"Expected a {MODEL_FAMILY} GenEvolve SFT checkpoint-{cfg['parent_step']}.")
    hashes = identity.get("sha256")
    require(isinstance(hashes, dict) and set(REQUIRED) <= hashes.keys()
            and any(name.endswith(".safetensors") for name in hashes)
            and all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
                    for value in hashes.values()),
            "Incomplete 2B parent fingerprint.")
    if NATIVE_KEY in identity:
        return verify_native_identity(identity)
    require(identity == read_json(ROOT / "configs/sft550_sha256.json"),
            "Weights/tokenizer differ from the exact SFT-550 parent. A fresh native run needs its validated training receipt.")
    return identity


def verify_parent(parent, cfg=None):
    parent = Path(parent)
    identity = sft_manifest(parent)
    if (parent / RECEIPT).exists():
        identity = verify_native_parent(parent, identity)
    return verify_parent_identity(identity, cfg)


def load_pack(folder, cfg, manifest=None):
    folder = Path(folder)
    m = manifest if manifest is not None else pack_manifest(folder, cfg)
    for name in ("train.jsonl", "batches.jsonl"):
        path = folder / name
        require(path.is_file() and file_sha256(path) == m["artifact_sha256"][name],
                f"Route-SFT artifact SHA-256 mismatch: {name}.")
    rows, batches = [], []
    ids_seen, labels_seen = set(), Counter()
    for index, row in enumerate(jsonl(folder / "train.jsonl")):
        ids, labels = row["input_ids"], row["labels"]
        start, history, label = row["target_start"], row["history_token_count"], row["route_label"]
        uid = row["id"]
        require(row["split"] == "train_only" and isinstance(uid, str) and uid not in ids_seen
                and label in (0, 1), f"Invalid decision identity or label at row {index}.")
        require(0 < history < start < len(ids) <= cfg["max_length"]
                and row["route_position"] == start - 1 and row["marker_id"] == MARKER_ID
                and all(type(token) is int and 0 <= token < 151936 for token in ids)
                and labels == [-100] * start + ids[start:]
                and ids[-1] == EOS_ID and ids[history:].count(EOS_ID) == 1
                and ids.count(MARKER_ID) == label and (ids[start] == MARKER_ID) == bool(label),
                f"Invalid causal boundary, mask or token at row {index}.")
        require(row["assistant_prefix"].endswith("</think>\n") and MARKER not in row["assistant_prefix"]
                and row["target"].startswith(MARKER) == bool(label),
                f"Invalid assistant/target split at row {index}.")
        rows.append(row)
        ids_seen.add(uid)
        labels_seen[label] += 1
    require(len(rows) == m["rows"] and labels_seen == {0: m["good"], 1: m["bad"]}, "Wrong row counts.")
    batch_seen = set()
    for j, batch in enumerate(jsonl(folder / "batches.jsonl")):
        indices = batch["row_indices_zero_based"]
        require(batch["batch_index_zero_based"] == j and len(indices) == cfg["good_per_batch"] + cfg["bad_per_batch"]
                and len(set(indices)) == len(indices)
                and all(type(k) is int and 0 <= k < len(rows) and k not in batch_seen for k in indices),
                f"Invalid fixed batch {j}.")
        items = [rows[k] for k in indices]
        require([x["id"] for x in items] == batch["ids"], f"Decision IDs disagree in batch {j}.")
        good = {x["id"]: x for x in items if x["route_label"] == 0}
        require(len(good) == cfg["good_per_batch"] and len(items) - len(good) == cfg["bad_per_batch"]
                and batch["good"] == cfg["good_per_batch"] and batch["bad"] == cfg["bad_per_batch"]
                and batch["synthetic_bad"] == sum(x["sample_origin"] == "synthetic" for x in items),
                f"Batch {j} is not 6 Good / 2 Bad.")
        for x in items:
            if x["sample_origin"] == "synthetic":
                mate = good.get(x["source_good_id"])
                require(x["route_label"] == 1 and mate is not None
                        and x["source_visible_sha256"] == mate["source_visible_sha256"]
                        and x["source_task_id"] == mate["source_task_id"],
                        f"Broken same-state pair in batch {j}.")
        batch_seen.update(indices)
        batches.append(indices)
    require(len(batches) == m["batches"] and len(batch_seen) == len(rows), "Incomplete fixed-batch coverage.")
    return rows, batches


def register_marker(tokenizer):
    require(tokenizer.eos_token_id == EOS_ID, "Unexpected original EOS.")
    added = tokenizer.add_special_tokens({"additional_special_tokens": [MARKER]},
                                         replace_additional_special_tokens=False)
    require(added == 1 and tokenizer.encode(MARKER, add_special_tokens=False) == [MARKER_ID]
            and MARKER_ID in tokenizer.all_special_ids,
            "Expected one newly registered marker in the checkpoint's reserved vocabulary slots.")
    return tokenizer


def verify_tokenizer(rows, tokenizer):
    """Compare the online chat template and decode boundary, not just the stored labels."""
    from .policy import tokenize_history

    class DecodeView:
        # The shared decoder does not modify the tokenizer. Cache its vocabulary
        # locally for the exhaustive preflight instead of rebuilding it per row.
        all_special_ids = tokenizer.all_special_ids

        def get_vocab(self):
            return vocabulary

        def decode(self, *args, **kwargs):
            return tokenizer.decode(*args, **kwargs)

    vocabulary = tokenizer.get_vocab()
    view = DecodeView()
    tail = tokenizer.encode("</think>\n", add_special_tokens=False)
    for j, row in enumerate(rows):
        ids = row["input_ids"]
        end, start = row["history_token_count"], row["target_start"]
        # The lock has already verified the captured IDs themselves.
        require(tokenize_history(tokenizer, row["messages"]) == ids[:end],
                f"Online template differs from frozen history at row {j}.")
        require(ids[start-len(tail):start] == tail
                and decode_actor_response(view, ids[end:]) == row["assistant_prefix"] + row["target"],
                f"Runtime decoder or masked separator differs at row {j}.")


def parent_tokenizer(parent):
    from transformers import AutoTokenizer

    return register_marker(AutoTokenizer.from_pretrained(parent, use_fast=True, local_files_only=True))


def route_identity(folder, m, cfg, parent_identity):
    return {
        "config": cfg, "manifest_sha256": file_sha256(Path(folder) / "manifest.json"),
        "artifact_sha256": {name: m["artifact_sha256"][name] for name in ("train.jsonl", "batches.jsonl")},
        "parent": verify_parent_identity(parent_identity, cfg),
        "route_data_source_sha256": m["checkpoint550_file_sha256"],
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL,
        "objective": "original+good_3way_forward_KL",
        "schedule_steps": 562, "save_resumable_at": [281, 562], "export_hf_at": [562],
    }
