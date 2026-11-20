# GenEvolve Route-SFT reviewed v3 — public training pack

Download [this repository's public dataset Release](https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x/releases/tag/route-sft-reviewed-v3-public), or run from a checkout of this repository:

```bash
python -m pip install filelock==3.20.0
python run.py --work "$WORK" prepare-route
```

Only the three files needed by Route-SFT are in the archive: `manifest.json`, `train.jsonl` and `batches.jsonl`. The archive and each file are pinned by [size and SHA-256](configs/route_release.json). The fetch command validates hashes, row boundaries and batch membership without model weights, GPUs or API credentials. Use `check-route --parent "$PARENT"` to also check the actual training parent's tokenizer.

The sample pool contains **2,376** decisions (1,782 Good, 594 Bad) in **297 ordered 6:2 batches**, from the reviewed v3 selections. A Bad sample includes a natural error or a reviewed same-history contrast. Only continuation tokens and the route boundary are supervised. There is no additional validation split in this pack. The reference origin is [GenEvolve](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c) and its [official Apache-2.0 data](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench/tree/9f291879d5dfe1195f47883e23069d3f6e829afe). These are derived training decisions, not an author-provided GenEvolve dataset. The offline collection checkpoint was 8B; the matched model trained here is the 2B SFT-550.

For publication, machine-specific path values, URL access parameters and contact addresses were removed only from prior tool responses. The changed history tokens and supervision offsets were recomputed with the pinned 2B tokenizer. The sample IDs, batch order, labels, current reasoning, and **all supervised token IDs** are unchanged from reviewed v3. The public pack is not byte-identical to the original private archive. It contains no rendered images, model weights, raw rollout archives, or private annotation logs. Official RL and evaluation splits are unchanged and are fetched separately with `prepare`.

Start a fresh `WORK` when switching from the original v3 export to the public pack. For a previous run that used the original archive, keep the original `--data` argument. Checkpoints and their stage-2 handoff record the actual manifest and file digests.
