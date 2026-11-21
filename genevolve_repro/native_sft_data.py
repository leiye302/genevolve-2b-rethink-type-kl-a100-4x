"""Offline verification or pin-only downloading of the original multimodal cold-start inputs."""
import hashlib
from pathlib import Path

from .common import ROOT, jsonl, read_json, source_lock, write_json
from .sft_export import file_sha256


def inputs():
    return read_json(ROOT / 'configs/native_sft_inputs.json')


def sft_registry():
    return {
        'genevolve_' + split: {
            'file_name': split + '.jsonl', 'formatting': 'sharegpt',
            'columns': {'messages': 'messages', 'images': 'images'},
            'tags': {'role_tag':'role', 'content_tag':'content', 'user_tag':'user',
                     'assistant_tag':'assistant', 'system_tag':'system'},
        } for split in ('train','eval')
    }


def git_digest(kind, raw):
    return hashlib.sha1(kind + b' ' + str(len(raw)).encode() + b'\0' + raw).digest()


def image_tree(directory):
    """Rebuild the HF Git tree from the actual image bytes' SHA256 LFS pointers."""
    entries=[]
    for path in directory.iterdir():
        name=path.name.encode('utf-8')
        if path.is_symlink():
            raise ValueError('Unexpected symlink in the pinned native image tree.')
        if path.is_dir():
            order,mode,oid=name+b'/',b'40000',image_tree(path)
        else:
            if path.suffix!='.jpg' or not path.is_file():
                raise ValueError('Unexpected non-JPEG in the pinned native image tree.')
            pointer=(f'version https://git-lfs.github.com/spec/v1\noid sha256:{file_sha256(path)}\n'
                     f'size {path.stat().st_size}\n').encode('ascii')
            order,mode,oid=name,b'100644',git_digest(b'blob',pointer)
        entries.append((order,mode+b' '+name+b'\0'+oid))
    return git_digest(b'tree',b''.join(item for _,item in sorted(entries)))


def scan_native_rows(folder, lock):
    referenced=set()
    for split in ('train','eval'):
        path=folder/(split+'.jsonl')
        if file_sha256(path)!=lock[split+'_jsonl_sha256']:
            raise ValueError(f'Native {split} JSONL differs from the official snapshot.')
        count=0
        for row in jsonl(path):
            msgs=row['messages']; roles=[m['role'] for m in msgs]
            if (len(msgs)<3 or roles!=['system']+['user','assistant']*((len(roles)-1)//2)
                    or sum(m['content'].count('<image>') for m in msgs)!=len(row['images'])):
                raise ValueError(f'Invalid native {split} image/assistant alignment at row {count}.')
            for name in row['images']:
                p=Path(name)
                if p.is_absolute() or '..' in p.parts or p.parts[:1]!=('images',) or p.suffix!='.jpg':
                    raise ValueError('Unsafe/unexpected native image path.')
                referenced.add(name)
            count+=1
        if count!=lock[split+'_rows']:
            raise ValueError(f'Incorrect native {split} row count.')
    if len(referenced)!=lock['referenced_jpegs']:
        raise ValueError('Incorrect number of pinned native references.')
    return referenced


def prepare_native(work, *, download=True):
    """Prepare just the cold start: NO embedding, RL data, API keys or renderer."""
    work=Path(work).resolve()
    lock=inputs(); pins=source_lock()
    base=work/'base'; folder=work/'data'/lock['dataset_folder']
    def fetch(name,dest,patterns,kind):
        if not download:
            raise FileNotFoundError(f'Missing native {name}; run prepare-native on a connected host.')
        from huggingface_hub import snapshot_download
        pin=pins[name]
        snapshot_download(pin['repo'], revision=pin['revision'],repo_type=kind,
                          local_dir=dest,allow_patterns=patterns)
    if not all((base/n).is_file() for n in lock['base_sha256']):
        fetch('base',base,list(lock['base_sha256']),'model')
    # Reject alternate vocab / unpinned added tokens, configs, templates and shards,
    # rather than just checking that the original files are also present.
    active={p.name for p in base.iterdir() if p.suffix in ('.json','.jinja','.safetensors','.model','.txt','.bin')}
    if active!=set(lock['base_sha256']) or any(file_sha256(base/n)!=sha for n,sha in lock['base_sha256'].items()):
        raise ValueError('Native training requires the exact bare public 2B weights and tokenizer.')
    folder_prefix=lock['dataset_folder']+'/'
    splits=[folder_prefix+s+'.jsonl' for s in ('train','eval')]
    if not all((folder/(s+'.jsonl')).is_file() for s in ('train','eval')):
        fetch('data',work/'data',splits,'dataset')
    names=scan_native_rows(folder,lock)
    if any(not (folder/p).is_file() for p in names):
        fetch('data',work/'data',[folder_prefix+'images/*'],'dataset')
    if image_tree(folder/'images').hex()!=lock['images_git_tree']:
        raise ValueError('Native image bytes or directory inventory differ from the pinned HF tree.')
    write_json(folder/'dataset_info.json',sft_registry())
    return base,folder
