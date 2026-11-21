# GenEvolve Route-SFT v4: the 281-batch library

This is a derived GenEvolve training-decision dataset; it is **not** the official benchmark. Its 2,248 distinct decisions are arranged in **281 fixed, ordered batches**, each with 6 Good and 2 Bad decisions. One epoch consumes each row once. Two epochs repeat that same order: 562 optimizer updates and 4,496 row presentations. No held-out benchmark example is intentionally used for SFT and there is no separate SFT validation split in this pack.

| Label | Unique examples | Origin |
|---|---:|---|
| Good | 1,686 | Natural decisions, with the original sampled action |
| Bad | 562 | 319 original natural errors + 54 natural resamples + 189 reviewed synthetic same-state contrasts |

**The 8B model hash in the data manifest identifies the trajectory collector, not the training parent.** Route-SFT starts from **2B native SFT-550**: either the historically pinned weights or a verified new native run from the pinned 2B base. Files from the old 297-batch v3 release, the natural-only 186-batch experiment, or the alternative recovery library are not interchangeable.

## Obtain the data

Use the [pinned v4 dataset release](https://github.com/leiye302/genevolve-2b-rethink-type-kl-a100-4x/releases/tag/route-sft-reviewed-v4-public). For an automatic anonymous download and verification:

```bash
python -m pip install filelock==3.20.0
python run.py --work "$WORK" prepare-route
```

For an offline machine, download **genevolve-route-sft-reviewed-v4-public.zip**, extract its three root-level files into one directory, then pass that directory via `--data` to `check-route` or `route-sft`. Do not repack, reorder or reformat any member:

```bash
python run.py --work "$WORK" check-route --data "$PACK" --parent "$PARENT"
```

`manifest.json` records the training contract and provenance; `train.jsonl` contains the model-visible messages, native tokens, mask and supervision boundary; `batches.jsonl` records explicit row indices and decision IDs. Only those three files are in the ZIP. [Archive and member checksums](configs/route_release.json) are enforced. Raw trajectories, rendered/reference images, API credentials, optimizer state, audit logs and excluded/reserve examples are **not** bundled.

## Public versus original inputs

The public download is the privacy-cleaned counterpart of the exact pack used in the reported run. A runtime filesystem path, address or URL access parameter in earlier tool feedback can leak even if only **token IDs** are published. Such strings are removed or replaced in tool messages; affected historical token prefixes, the position of the routing boundary and the ignored portion of the labels are recomputed using the pinned tokenizer. The current reasoning, the entire continuation after the history, supervised action tokens, Good/Bad labels, paired relationships, row identities and all 281 batches are unchanged. Public inputs must **not** be described as byte-identical to the reported run.

| Whole-pack privacy/equivalence check | Verified result |
|---|---:|
| Rows with changed history / rows with entirely unchanged input IDs | 1,679 / 569 |
| Tool-feedback replacements (runtime paths / URL access parameters / contact addresses, counting repeated occurrences) | 5,672 / 6 / 3 |
| Total history tokens, original → public | 10,946,238 → 10,543,499 |
| Supervised tokens per epoch, original = public | 233,323 (zero changed) |
| Current reasoning prefixes changed / supervised actions changed | 0 / 0 |
| Fixed batch files | Byte-identical |

If you already have authorized access to the **original unmodified v4** pack, supply its three training files with the same `--data "$PACK"` option and use a fresh `WORK`. [Its independent original-file hashes](configs/route_original_v4.json) select the exact historical data path; no arbitrary directory is accepted. The public and original manifest hashes are different. Do not attempt to resume a checkpoint trained with one using the other.

Upstream: [GenEvolve source](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c) and [GenEvolve benchmark data](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench/tree/9f291879d5dfe1195f47883e23069d3f6e829afe). Official train/RL and benchmark downloads are separate from this Route-SFT decision pack.
