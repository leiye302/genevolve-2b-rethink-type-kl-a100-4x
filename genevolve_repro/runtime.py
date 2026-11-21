"""Check stage-specific resources before allocating any training workers."""

import shutil

from .common import WORLD_SIZE


def preflight(stage, work):
    import torch

    if stage == "route-sft":
        from .route_data import recipe
        size = recipe()["world_size"]
        if size != 4 or not torch.cuda.is_available() or torch.cuda.device_count() != size:
            raise ValueError("Route-SFT requires exactly four visible A100 80GB GPUs; use CUDA_VISIBLE_DEVICES.")
        for i in range(size):
            with torch.cuda.device(i):
                name = torch.cuda.get_device_name(i)
                free, total = torch.cuda.mem_get_info(i)
                if "A100" not in name.upper() or total < 75 * 2**30 or not torch.cuda.is_bf16_supported():
                    raise ValueError(f"Route-SFT requires four BF16 A100 80GB GPUs; visible device {i}: "
                                     f"{name}, {total / 2**30:.1f} GiB.")
                if free < 68 * 2**30:
                    raise ValueError(f"Route-SFT needs >=68 GiB free on each A100; visible device {i} "
                                     f"has {free / 2**30:.1f} GiB. Stop the renderer or move it to another GPU.")
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
