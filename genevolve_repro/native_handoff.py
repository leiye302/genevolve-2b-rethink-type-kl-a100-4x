"""Bind a newly run native SFT-550 to its own HF weights and complete resume files.

This is a local integrity/provenance receipt, not a signature by the original
authors. No training path, optimizer content or example is embedded in it.
There is deliberately no CLI for attesting an arbitrary user-supplied export.
"""
import hashlib
import json
from pathlib import Path
import re

from .common import read_json, write_json
from .sft_export import file_sha256, sft_manifest


RECEIPT = "native_parent_receipt.json"
NATIVE_KEY = "native_parent_provenance"
SCHEMA = "genevolve-fresh-native-2b-sft550-4rank-v1"
STEP = 550
TAG = "global_step550"
SHA = re.compile(r"[a-f0-9]{64}\Z")
OPT = re.compile(r"(?:bf16_)?zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt\Z")
MODEL = re.compile(r"(?:(?:bf16_)?zero_pp_rank_(\d+)_)?mp_rank_00_model_states\.pt\Z")
TOP = {"latest", "trainer_state.json", "training_args.bin", *(f"rng_state_{i}.pth" for i in range(4))}


def canonical_sha(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_legacy_contract(contract):
    # Canonical JSON of native_contract() at 72514a4f3c39ac972fcc348c35bd0e6f41cc78e1.
    # A read-only allowlist entry, *not* an alternate recipe for new training or sealing.
    return canonical_sha(contract)=="c5b8becf2b85d41d8c1331c5ec830f845031c00b27f2a012f2a5883a3785ffcd"


def _pinned_contract():
    # Dependency-free native prep is imported lazily. Legacy parents need none of it.
    from .native_sft import native_contract
    return native_contract()


def _need(ok, msg):
    if not ok:
        raise ValueError(msg)


def _check_resume_inventory(hashes):
    _need(isinstance(hashes, dict) and TOP <= hashes.keys(), "Missing native trainer or four-rank RNG state.")
    model, opt = [], []
    for path, digest in hashes.items():
        _need(isinstance(path, str) and isinstance(digest, str) and SHA.fullmatch(digest),
              "Invalid native resume fingerprint.")
        if path in TOP or path == "scheduler.pt":
            continue
        prefix, sep, name = path.partition("/")
        _need(prefix == TAG and sep and "/" not in name, "Unexpected native resume file.")
        m, o = MODEL.fullmatch(name), OPT.fullmatch(name)
        if m:
            model.append(-1 if m[1] is None else int(m[1]))
        elif o:
            opt.append(int(o[1]))
        else:
            raise ValueError("Unexpected native DeepSpeed shard.")
    _need(sorted(model) in ([-1], [0,1,2,3]) and sorted(opt)==[0,1,2,3],
          "Incomplete or duplicate native four-rank DeepSpeed model/optimizer states.")


def _check_regular_file(folder, name):
    path = folder / name
    _need(path.is_file() and not path.is_symlink() and folder.resolve() in path.resolve().parents
          and path.stat().st_size > 0, f"Incomplete or external native artifact: {name}.")
    return path


def _resume_manifest(folder):
    stepdir = folder / TAG
    _need(stepdir.is_dir() and not stepdir.is_symlink(), "Missing native DeepSpeed step-550 directory.")
    _need(_check_regular_file(folder, "latest").read_text(encoding="utf-8").strip() == TAG,
          "Native DeepSpeed latest is not step 550.")
    names = TOP | {f"{TAG}/{p.name}" for p in stepdir.iterdir() if p.suffix == ".pt"}
    if (folder/"scheduler.pt").exists():
        names.add("scheduler.pt")
    # Validate names/ranks before touching potentially large shards.
    _check_resume_inventory({name: "0"*64 for name in names})
    result = {name:file_sha256(_check_regular_file(folder, name)) for name in sorted(names)}
    return result


def _identity(hf, contract, resume):
    attestation = {"schema": SCHEMA, "native_training_contract": contract,
                   "hf_manifest_sha256": canonical_sha(hf), "resume_sha256":resume}
    attestation["binding_sha256"] = canonical_sha(attestation)
    return {**hf, NATIVE_KEY:attestation}


def verify_native_identity(identity):
    """Validate the embedded parent identity retained in a completed Route-SFT contract."""
    _need(isinstance(identity, dict) and set(identity)=={"model_family","sft_global_step","sha256",NATIVE_KEY},
          "Invalid generated native parent identity.")
    hf = {k:v for k,v in identity.items() if k!=NATIVE_KEY}
    att = identity[NATIVE_KEY]
    _need(isinstance(att, dict) and set(att)=={"schema","native_training_contract","hf_manifest_sha256",
                                               "resume_sha256","binding_sha256"},
          "Invalid generated native parent receipt.")
    _check_resume_inventory(att["resume_sha256"])
    _need(hf["sft_global_step"]==STEP and att["schema"]==SCHEMA and
          (att["native_training_contract"]==_pinned_contract() or is_legacy_contract(att["native_training_contract"])) and
          att["hf_manifest_sha256"]==canonical_sha(hf) and
          att["binding_sha256"]==canonical_sha({k:v for k,v in att.items() if k!="binding_sha256"}),
          "Generated native parent provenance or bound HF fingerprint differs.")
    return identity


def seal_native_parent(folder, contract):
    """Only the native launcher calls this after successful verified training completion."""
    folder = Path(folder)
    _need(contract==_pinned_contract(), "Cannot seal outputs from an unpinned native training contract.")
    hf = sft_manifest(folder)
    _need(hf["sft_global_step"]==STEP, "Cannot seal an incomplete native SFT-550.")
    for name in hf["sha256"]:
        _check_regular_file(folder,name)
    identity = _identity(hf,contract,_resume_manifest(folder))
    receipt = folder / RECEIPT
    if receipt.exists():
        _need(read_json(_check_regular_file(folder,RECEIPT))==identity, "Refusing to replace a different native parent receipt.")
    else:
        write_json(receipt,identity)
    return identity


def verify_native_parent(folder, hf):
    """Rehash the files that will actually be loaded; a copied or stale receipt is insufficient."""
    folder = Path(folder)
    stored = verify_native_identity(read_json(_check_regular_file(folder,RECEIPT)))
    for name in hf["sha256"]:
        _check_regular_file(folder,name)
    hashes = _resume_manifest(folder)
    expected = _identity(hf,stored[NATIVE_KEY]["native_training_contract"], hashes)
    _need(stored==expected, "Native HF or resume files differ from their training receipt.")
    return expected
