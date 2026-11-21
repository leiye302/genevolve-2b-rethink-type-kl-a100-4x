"""Original 2B multimodal SFT-550; no route marker, KL, hint, retrieval or image judge."""
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .common import ROOT, read_json, source_lock, write_json
from .native_sft_data import git_digest, inputs, prepare_native
from .sft_export import export_files


def trainer_root():
    return ROOT/'.deps/llamafactory'


def native_contract():
    """Path-independent and CPU/dependency-free; also used when importing a completed parent."""
    lock=source_lock()
    native_recipe=ROOT/'configs/native_sft.json'
    text=native_recipe.read_bytes().replace(b'\r\n',b'\n')
    assets=inputs()
    if git_digest(b'blob',text).hex()!=assets['upstream_sft_json_git_blob']:
        raise ValueError('Native SFT recipe differs from its pinned 2B upstream revision.')
    return {
        'format':'genevolve-original-multimodal-2b-sft550-v1',
        'sources':{k:lock[k] for k in ('base','data','cold_start','sft_trainer')},
        'inputs':assets, 'lhf_recipe':read_json(native_recipe),
        'four_rank_topology':{'world_size':4,'micro_batch_per_gpu':2,'gradient_accumulation':4,
                              'global_batch':32,'epochs':2,'optimizer_updates':550},
        'runtime':{'python':'3.11','torch':'2.8.0','cuda':'12.8','transformers':'4.57.1',
                   'tokenizers':'0.22.1','accelerate':'1.11.0','datasets':'4.0.0','deepspeed':'0.17.6',
                   'qwen-vl-utils':'0.0.14','flash-attn':'2.8.3','huggingface-hub':'0.36.0'},
        'launcher_sources_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()
                                 .replace(b'\r\n',b'\n')).hexdigest() for n in
                                 ('native_sft.py','native_sft_data.py','native_sft_probe.py')},
    }


def verify_runtime(contract):
    import torch
    trainer=trainer_root()
    def git(*args):
        return subprocess.check_output(['git','-C',str(trainer),*args],text=True).strip()
    if (not trainer.is_dir() or git('rev-parse','HEAD')!=contract['sources']['sft_trainer']['revision']
            or git('status','--porcelain','--untracked-files=no')):
        raise RuntimeError('Install the clean pinned LLaMA-Factory via bash scripts/setup.sh.')
    spec=importlib.util.find_spec('llamafactory')
    if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(trainer.resolve()/'src'):
        raise RuntimeError('Activate the native training venv with its pinned editable LLaMA-Factory.')
    ds=(trainer/'examples/deepspeed/ds_z3_config.json').read_bytes().replace(b'\r\n',b'\n')
    if git_digest(b'blob',ds).hex()!=contract['inputs']['zero3_config_git_blob']:
        raise ValueError('ZeRO-3 trainer config is not the pinned, non-offloaded config.')
    versions=contract['runtime']
    if sys.version_info[:2]!=(3,11) or torch.version.cuda!=versions['cuda']:
        raise RuntimeError('Native cold-start runtime requires Python 3.11 + CUDA 12.8 PyTorch.')
    for package,version in versions.items():
        if package in ('python','cuda'):continue
        if importlib.metadata.version(package).split('+')[0]!=version:
            raise RuntimeError(f'Native runtime version mismatch for {package}: expected {version}.')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4:
        raise RuntimeError('Expose exactly four A100-80GB devices for native SFT.')
    for i in range(4):
        with torch.cuda.device(i):
            free,total=torch.cuda.mem_get_info(i)
            if ('A100' not in torch.cuda.get_device_name(i).upper() or total<75*2**30 or free<68*2**30
                    or not torch.cuda.is_bf16_supported()):
                raise RuntimeError(f'GPU {i} is not an available BF16 A100-80GB (>=68 GiB free).')


def native_checkpoint(folder,step):
    """Check the HF model+processor and the four ZeRO partitions, RNG and training state."""
    if folder.name!=f'checkpoint-{step}' or folder.is_symlink() or not folder.is_dir():
        raise ValueError(f'Expected native checkpoint-{step}.')
    export_files(folder)
    for name in ('trainer_state.json','training_args.bin','latest',*(f'rng_state_{r}.pth' for r in range(4))):
        p=folder/name
        if p.is_symlink() or not p.is_file() or not p.stat().st_size:
            raise ValueError(f'Incomplete native checkpoint-{step}: {name}')
    if (read_json(folder/'trainer_state.json').get('global_step')!=step
            or (folder/'latest').read_text().strip()!=f'global_step{step}'):
        raise ValueError('Native Trainer and ZeRO checkpoint progress disagree.')
    ds=folder/f'global_step{step}'
    if not ds.is_dir() or ds.is_symlink():
        raise ValueError('Missing native ZeRO state directory.')
    def ranks(which):
        values=[]
        for p in ds.glob('*'+which+'_states.pt'):
            m=re.fullmatch(r'(?:(?:bf16_)?zero_pp_rank_(\d+)_)?mp_rank_00_'+which+'_states.pt',p.name)
            if not m or not p.stat().st_size or not p.is_file() or p.is_symlink():
                raise ValueError('Invalid native DeepSpeed shard.')
            values.append(-1 if m[1] is None else int(m[1]))
        return sorted(values)
    if ranks('optim')!=[0,1,2,3] or ranks('model') not in ([-1],[0,1,2,3]):
        raise ValueError('Missing or duplicate ZeRO model/optimizer partitions.')
    return folder


def launch_config(work, resume=None):
    work=Path(work).resolve()
    cfg=read_json(ROOT/'configs/native_sft.json')
    folder=work/'data'/inputs()['dataset_folder']
    cfg.update(model_name_or_path=str(work/'base'), dataset_dir=str(folder),media_dir=str(folder),
               output_dir=str(work/'native_sft'),deepspeed=str(trainer_root()/'examples/deepspeed/ds_z3_config.json'))
    if resume is not None:cfg['resume_from_checkpoint']=str(resume)
    return cfg


def run_native(work):
    """Driver, called once on the host, not by four torchrun child processes."""
    from .native_handoff import RECEIPT, seal_native_parent
    from .native_sft_probe import verify_labels
    work=Path(work).resolve()
    out=work/'native_sft'
    meta=work/'native_sft_run'
    contract=native_contract()
    bind=meta/'contract.json'
    success=meta/'trainer-succeeded.json'
    final=out/'checkpoint-550'
    if out.is_symlink() or meta.is_symlink() or bind.is_symlink() or success.is_symlink():
        raise ValueError('Refusing linked native output/provenance directory.')
    if bind.is_file():
        if read_json(bind)!=contract:raise ValueError('Native run contract changed. Use a fresh WORK.')
    elif out.exists() and (not out.is_dir() or any(out.iterdir())) or (meta.exists() and any(meta.iterdir())):
        raise ValueError('Refusing pre-existing unbound native outputs. Use a fresh WORK.')
    def finish():
        if not bind.is_file() or not success.is_file() or read_json(success)!={'contract':contract,'completed':550}:
            raise ValueError('No matching successful native invocation; cannot issue a parent receipt.')
        native_checkpoint(out/'checkpoint-275',275)
        native_checkpoint(final,550)
        if not (out/'trainer_state.json').is_file() or read_json(out/'trainer_state.json').get('global_step')!=550:
            raise ValueError('Native training did not finalize all 550 updates.')
        seal_native_parent(final,contract)
        return final
    if success.is_file():return finish()
    if (final/RECEIPT).exists() or final.exists():
        raise ValueError('Unconfirmed step-550 output: recover the original run records, or resume from its verified '
                         '275 checkpoint after archiving the incomplete/unconfirmed 550 output; not sealing it.')
    unknown=set(out.glob('checkpoint-*'))-{out/'checkpoint-275'}
    if unknown:raise ValueError('Unexpected native checkpoints. Use a fresh WORK.')
    resume=native_checkpoint(out/'checkpoint-275',275) if (out/'checkpoint-275').exists() else None
    verify_runtime(contract)
    work.mkdir(parents=True,exist_ok=True)
    need=(65 if resume else 105)*2**30
    if shutil.disk_usage(work).free<need:
        raise ValueError(f'Native original data, two full saves and HF export require at least {need//2**30} GiB free.')
    base,folder=prepare_native(work)
    cfg=launch_config(work,resume)
    labels=verify_labels(base,folder,cfg)
    write_json(meta/'native-label-probe.json',labels)
    write_json(bind,contract)
    args=meta/'launch.yaml'
    write_json(args,cfg)
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',
                    '--module','llamafactory.launcher',str(args)],cwd=ROOT,check=True)
    native_checkpoint(out/'checkpoint-275',275)
    native_checkpoint(final,550)
    export_files(out)
    if read_json(out/'trainer_state.json').get('global_step')!=550:
        raise ValueError('Trainer returned without the full expected native export.')
    write_json(success,{'contract':contract,'completed':550})
    return finish()
