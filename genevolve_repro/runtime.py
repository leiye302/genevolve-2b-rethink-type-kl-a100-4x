"""Check stage-specific resources before allocating any training workers."""

import shutil

from .common import WORLD_SIZE


def preflight(stage, work):
    import torch

    if stage == "route-sft":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise ValueError("Route-SFT requires one visible CUDA GPU.")
        with torch.cuda.device(0):
            free, total = torch.cuda.mem_get_info()
            if not torch.cuda.is_bf16_supported() or total < 29 * 2**30 or free < min(total * .85, 58 * 2**30):
                raise ValueError("Route-SFT requires an available BF16 GPU with >=29 GiB; see the hardware recipe.")
        folder = work / "route_sft"
        need = (9 if (folder / "checkpoint-00562" / "state_sha256.json").is_file() else
                35 if (folder / "checkpoint-00281" / "state_sha256.json").is_file() else 58)
        if shutil.disk_usage(work).free < need * 2**30:
            raise ValueError(f"Allow at least {need} GiB free in WORK for outstanding Route-SFT saves.")
        return
    if torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError("Expose exactly four 80GB CUDA GPUs for this recipe.")
    if any(torch.cuda.get_device_properties(i).total_memory < 75 * 2**30 for i in range(WORLD_SIZE)):
        raise ValueError("The baseline-matched recipe requires four 80GB-class CUDA GPUs.")
    for i in range(WORLD_SIZE):
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("All four assigned GPUs must support BF16.")
            free, _ = torch.cuda.mem_get_info(i)
            if free < 68 * 2**30:
                raise ValueError("The four GPUs must be free; use a separate renderer or free the assigned cards.")
    # Native FSDP stores one FP32 actor and FP32 Adam states in each full
    # checkpoint. Allow for two existing checkpoints plus a temporary save.
    folder = work / stage
    if not (folder / "contract.json").is_file() and shutil.disk_usage(work).free < 100 * 2**30:
        raise ValueError("Allow at least 100 GiB free in WORK for 2B checkpoint space; dataset and renderer are separate.")
