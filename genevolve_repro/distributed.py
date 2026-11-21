from contextlib import nullcontext
from datetime import timedelta
from functools import partial
import os
from pathlib import Path
import random
import shutil
from tempfile import mkdtemp

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload, FullOptimStateDictConfig, FullStateDictConfig,
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import WORLD_SIZE, read_json, write_json
from .policy import ActorScorer
from .sft_export import file_sha256


def initialize(world_size=None):
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    expected = WORLD_SIZE if world_size is None else world_size
    if size != expected or torch.cuda.device_count() < expected:
        raise RuntimeError(f"Expose {expected} CUDA GPU(s) and start the worker through run.py.")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2), device_id=device)
    return rank, size, device


def load_model(path, dtype, attention="flash_attention_2"):
    from transformers import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=dtype, attn_implementation=attention, low_cpu_mem_usage=True,
    )


def make_sharded_actor(path, device, saved=None, scorer_type=ActorScorer, *, cpu_offload=True):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    # Load the attention backend after loading FP32 master weights. FSDP
    # supplies BF16 parameters for the training forward and backward.
    policy = load_model(path, torch.float32, attention="sdpa")
    policy.set_attn_implementation("flash_attention_2")
    policy.model.visual.requires_grad_(False)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer = scorer_type(policy)
    if saved is not None:
        scorer.load_state_dict(saved)
    return FSDP(
        scorer,
        device_id=device,
        auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                 transformer_layer_cls={Qwen3VLTextDecoderLayer}),
        # NO_SHARD + tied weights + mixed-precision accumulation can attach a
        # BF16 shared gradient to an FP32 original view in torch 2.8. Keep flat
        # parameter ownership on one rank; FP32 masters/BF16 compute are unchanged.
        use_orig_params=not (dist.is_initialized() and dist.get_world_size() == 1),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        ignored_modules=[policy.model.visual],
        limit_all_gathers=True,
    )


def make_ddp_route_actor(path, device, saved, scorer_type):
    """One replica per A100. Keep trainable parameters and AdamW moments in FP32."""
    policy = load_model(path, torch.float32, attention="sdpa")
    policy.model.visual.requires_grad_(False)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer = scorer_type(policy)
    if saved is not None:
        scorer.load_state_dict(saved, strict=True)
    scorer.to(device)
    # Set FA2 after moving to this rank's GPU. Autocast in update_ddp produces
    # BF16 attention activations without reducing master/AdamW state precision.
    policy.set_attn_implementation("flash_attention_2")
    if policy.config.text_config._attn_implementation != "flash_attention_2":
        raise RuntimeError("The Qwen text decoder did not activate FlashAttention 2.")
    if any(p.dtype != torch.float32 for p in scorer.parameters() if p.requires_grad):
        raise RuntimeError("DDP Route-SFT requires FP32 trainable master weights.")
    return DDP(scorer, device_ids=[device.index], output_device=device.index,
               broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)


@torch.no_grad()
def copy_actor(sharded, target, rank0_only=False):
    # Generation uses a local BF16 replica so decoding does not all-gather
    # parameters for each token. The replica is refreshed before each batch.
    if not isinstance(sharded, FSDP):
        if not rank0_only or not dist.is_initialized() or dist.get_rank() == 0:
            source = sharded.module if isinstance(sharded, DDP) else sharded
            target.load_state_dict(source.model.state_dict())
        return
    # CPU parameters need no offload; Torch also rejects summon_full_params
    # offload for NO_SHARD (including a single CUDA rank). Copies into a CPU
    # replica still happen below, without changing the trained weights.
    offload = (sharded.compute_device.type != "cpu"
               and sharded.sharding_strategy != ShardingStrategy.NO_SHARD)
    with FSDP.summon_full_params(
        sharded, writeback=False, offload_to_cpu=offload, rank0_only=rank0_only,
    ):
        if not rank0_only or dist.get_rank() == 0:
            source = {
                name.replace("._fsdp_wrapped_module.", "."): parameter
                for name, parameter in sharded.module.model.named_parameters()
            }
            for name, parameter in target.named_parameters():
                parameter.copy_(source[name])


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def max_count(count, device):
    if not dist.is_initialized():
        return count
    value = torch.tensor(count, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return int(value.item())


def barrier():
    if dist.is_initialized():
        dist.barrier()


def full_state(actor):
    if not isinstance(actor, FSDP):
        return nullcontext()
    offload = actor.compute_device.type != "cpu"
    return FSDP.state_dict_type(
        actor,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=offload, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=offload, rank0_only=True),
    )


def checkpoints(folder):
    paths = [
        path for path in folder.glob("checkpoint-*")
        if path.is_dir() and not path.is_symlink() and path.name.removeprefix("checkpoint-").isdigit()
        and (path / "training.pt").is_file()
    ]
    return sorted(paths, key=lambda p: int(p.name.removeprefix("checkpoint-")))


def latest(folder):
    paths = checkpoints(folder)
    return paths[-1] if paths else None


def load_checkpoint(path):
    payload = path / "training.pt"
    receipt = path / "state_sha256.json"
    if receipt.is_file() and read_json(receipt)["training.pt"] != file_sha256(payload):
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path.name}.")
    return torch.load(payload, map_location="cpu", mmap=True, weights_only=True)


def restore_optimizer(actor, optimizer, scheduler, saved, rank):
    optimizer_state = saved["optimizer"]
    if isinstance(actor, FSDP):
        with full_state(actor):
            optimizer_state = FSDP.optim_state_dict_to_load(actor, optimizer, optimizer_state if rank == 0 else {})
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(saved["scheduler"])


def get_rng(device):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(value, device):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"], device)


def save_checkpoint(folder, actor, optimizer, scheduler, step, contract, rank, device, keep=2, verify=False, *, memory=None):
    states = gather(get_rng(device))
    if isinstance(actor, DDP):
        # All replicas are already synchronized. Use unwrapped names on disk;
        # the complete AdamW state is needed once, not four times.
        if rank == 0:
            model = actor.module.state_dict()
            optimizer_state = optimizer.state_dict()
    else:
        with full_state(actor):
            model = actor.state_dict()
            optimizer_state = FSDP.optim_state_dict(actor, optimizer) if isinstance(actor, FSDP) else optimizer.state_dict()
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"checkpoint-{step:05d}"
        temporary = Path(mkdtemp(prefix=".checkpoint-", dir=folder))
        payload = temporary / "training.pt"
        state = {
            "step": step, "contract": contract, "actor": model, "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(), "rng": states,
        }
        if memory is not None:
            state["memory"] = memory.entries
        torch.save(state, payload)
        if verify:
            restored = torch.load(payload, map_location="cpu", mmap=True, weights_only=True)
            if (restored["step"] != step or restored["contract"] != contract
                    or restored["actor"].keys() != model.keys()
                    or restored["optimizer"].keys() != optimizer_state.keys()
                    or restored.get("memory") != state.get("memory")
                    or len(restored["rng"]) != len(states)):
                raise ValueError("The newly saved checkpoint is incomplete.")
            del restored
            write_json(temporary / "state_sha256.json", {"training.pt": file_sha256(payload)})
        temporary.rename(target)
        for previous in checkpoints(folder)[:-keep]:
            resolved = previous.resolve()
            if resolved.parent != folder.resolve():
                raise ValueError(f"Unexpected checkpoint directory: {resolved}")
            shutil.rmtree(resolved)
    barrier()


def export_policy(folder, actor, replica, processor, step, rank, metadata=None):
    copy_actor(actor, replica, rank0_only=True)
    if rank == 0:
        temporary = Path(mkdtemp(prefix=".export-", dir=folder.parent))
        replica.save_pretrained(temporary, safe_serialization=True)
        processor.save_pretrained(temporary)
        write_json(temporary / "completed.json", {"step": step, **(metadata or {})})
        temporary.rename(folder)
    barrier()
