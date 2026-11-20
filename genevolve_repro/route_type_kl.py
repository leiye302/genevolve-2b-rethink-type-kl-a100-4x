"""Good-only conditional Tool/Answer KL; no change to tokens or action protocol."""

import gc
import math

import torch
from torch.nn import functional as F

from .common import read_json, write_json
from .sft_export import file_sha256

# These are native tokenizer IDs, NOT new special tokens. Answer is a
# two-token distinguishing prefix. Its next '>' can merge with whitespace/JSON.
TYPE_IDS = (151657, 27, 9217)  # <tool_call>, <, answer
VERSION = "parent550-good-tool-answer-prefix-forward-kl-v1"


def validate_config(cfg):
    weight = cfg.get("good_type_kl_lambda", 0.0)
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("good_type_kl_lambda must be finite and nonnegative.")
    if weight and cfg.get("good_type_kl_version") != VERSION:
        raise ValueError("Unknown Good type-KL objective version.")
    return weight


def verify_type_tokenizer(tokenizer):
    if (tokenizer.encode("<tool_call>", add_special_tokens=False) != [TYPE_IDS[0]]
            or tokenizer.encode("<answer", add_special_tokens=False) != list(TYPE_IDS[1:])):
        raise ValueError("Native Tool/Answer distinguishing token prefixes changed.")


def type_log_probs(first_logits, after_open_logits, type_ids=TYPE_IDS):
    """Conditional normalized p(tool|s), p('<answer'|s), in log space.

    The first-step full-vocabulary denominator cancels. This intentionally
    is NOT the route BCE's full-vocabulary probability, nor p('<'|s) alone.
    """
    tool, opening, answer = type_ids
    first_logits, after_open_logits = first_logits.float(), after_open_logits.float()
    scores = torch.stack((first_logits[tool], first_logits[opening]
                          + F.log_softmax(after_open_logits, dim=-1)[answer]))
    return F.log_softmax(scores, dim=0)


def reference_to_student_kl(student_logp, reference_logp):
    # Direction is forward KL(reference || student). No reference gradients.
    return F.kl_div(student_logp.float(), reference_logp.detach().float(),
                    reduction="sum", log_target=True)


def candidate_prefix(input_ids, start, type_ids=TYPE_IDS):
    # H/R exactly as supplied to CE, ending BEFORE its first target. Append
    # only the candidate '<'. No ground-truth action or future observation.
    return torch.cat((input_ids[:, :start], input_ids.new_tensor([[type_ids[1]]])), dim=1)


@torch.no_grad()
def score_reference(model, input_ids, start, type_ids=TYPE_IDS):
    ids = candidate_prefix(input_ids, start, type_ids)
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    first, after_open = model.lm_head(hidden[0, -2:]).unbind(0)
    return type_log_probs(first, after_open, type_ids)


def validate_entries(entries, rows, expected, *, complete):
    if not isinstance(entries, dict) or not set(entries) <= {str(k) for k in expected}:
        raise ValueError("Unexpected reference cache row coverage.")
    if complete and set(entries) != {str(k) for k in expected}:
        raise ValueError("Incomplete reference cache row coverage.")
    for key, item in entries.items():
        logp = torch.tensor(item["logp"], dtype=torch.float64)
        if (item["id"] != rows[int(key)]["id"] or logp.shape != (2,)
                or not torch.isfinite(logp).all() or abs(float(logp.logsumexp(0))) > 2e-5):
            raise ValueError("Invalid reference cache probabilities or sample identity.")


def prepare_reference_cache(out, parent, rows, contract, rank, size, device, *, loader=None):
    """Score frozen parent once, before loading the training actor.

    Each rank processes a disjoint Good shard. Atomic progress files allow
    restart; sealed shards are checked by SHA-256. Only 2 floats/Good are kept.
    No GPU reference remains resident during SFT.
    """
    from .distributed import gather, load_model

    cache_identity = {"version": VERSION, "contract": contract, "shards": size,
                      "reference_dtype": "bfloat16", "reference_attention": "flash_attention_2",
                      "type_ids": list(TYPE_IDS)}
    folder = out / "good_type_reference"
    folder.mkdir(parents=True, exist_ok=True)
    path, receipt = folder / f"rank-{rank}.json", folder / f"rank-{rank}.sha256.json"
    expected = [k for k, row in enumerate(rows) if row["route_label"] == 0 and k % size == rank]
    saved = read_json(path) if path.is_file() else {"identity": cache_identity, "entries": {}}
    if saved["identity"] != cache_identity:
        raise ValueError("Good type reference belongs to another parent/data/config; use a new WORK.")
    entries = saved["entries"]
    sealed = receipt.is_file()
    if sealed and read_json(receipt) != {path.name: file_sha256(path)}:
        raise ValueError("Good type reference SHA-256 mismatch.")
    validate_entries(entries, rows, expected, complete=sealed)
    missing = [k for k in expected if str(k) not in entries]
    if missing:
        print(f"rank {rank}: caching frozen parent Tool/Answer probabilities for {len(missing)} Good rows", flush=True)
        model = (loader or load_model)(parent, torch.bfloat16).to(device).eval().requires_grad_(False)
        for count, k in enumerate(missing, 1):
            row = rows[k]
            ids = torch.tensor([row["input_ids"][:row["target_start"]]], device=device, dtype=torch.long)
            logp = score_reference(model, ids, row["target_start"])
            entries[str(k)] = {"id": row["id"], "logp": logp.cpu().tolist()}
            if count % 25 == 0 or count == len(missing):
                validate_entries(entries, rows, expected, complete=False)
                write_json(path, saved)
                print(f"rank {rank}: reference {len(entries)}/{len(expected)}", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not path.is_file():
        write_json(path, saved)
    validate_entries(entries, rows, expected, complete=True)
    write_json(receipt, {path.name: file_sha256(path)})
    shards = gather(entries)
    combined = {}
    for shard in shards:
        if combined.keys() & shard.keys():
            raise ValueError("Duplicate Good reference across ranks.")
        combined.update(shard)
    validate_entries(combined, rows, [k for k, row in enumerate(rows) if row["route_label"] == 0], complete=True)
    return {int(k): item["logp"] for k, item in combined.items()}
