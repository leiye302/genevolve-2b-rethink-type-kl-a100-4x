"""Anonymous public-pack/tokenizer + real tiny-Qwen loss/gradient smoke.

No real model weights, API calls, CUDA or claims about production performance.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genevolve_repro.common import source_lock, write_json
from genevolve_repro.route_data import MARKER_ID, load_pack, recipe, register_marker, verify_tokenizer
from genevolve_repro.route_release import prepare_route
from genevolve_repro.route_type_kl import score_threeway, verify_type_ids
from genevolve_repro.sft_export import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Optional already verified public pack')
    args = parser.parse_args()
    work = args.work.resolve()
    cfg = recipe()
    folder = args.data or prepare_route(work)
    rows, batches = load_pack(folder, cfg)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    import torch
    from genevolve_repro.route_sft import RouteScorer

    source = source_lock()['base']
    tokenizer_dir = work / 'public-tokenizer'
    snapshot_download(source['repo'], revision=source['revision'], token=False, local_dir=tokenizer_dir,
                      allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'chat_template.json',
                                      'chat_template.jinja', 'merges.txt', 'vocab.json', 'config.json'])
    tokenizer = register_marker(AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True))
    verify_tokenizer(rows, tokenizer)
    verify_type_ids(tokenizer)
    counts = Counter((r['route_label'], r['target_action_kind']) for r in rows)
    print({'verified_rows': len(rows), 'batches': len(batches), 'types': str(counts)}, flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model_cfg = Qwen3VLConfig(
        tie_word_embeddings=True, image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
        text_config={'vocab_size': 151936, 'hidden_size': 32, 'intermediate_size': 64,
                     'num_hidden_layers': 1, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'head_dim': 8, 'max_position_embeddings': 16384, 'tie_word_embeddings': True,
                     'rope_scaling': {'rope_type': 'default', 'mrope_section': [2, 1, 1]},
                     'bos_token_id': 151643, 'eos_token_id': 151645, 'pad_token_id': 151643},
        vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                       'out_hidden_size': 32, 'deepstack_visual_indexes': [], 'num_position_embeddings': 16},
    )
    model_cfg._attn_implementation = 'sdpa'
    model = Qwen3VLForConditionalGeneration(model_cfg).eval().requires_grad_(False)

    def kind(row):
        return row['route_label'], '<answer>' in row['target'][:32]

    eligible = [b for b in batches if len({kind(rows[k]) for k in b}) == 4]
    batch = min(eligible, key=lambda b: sum(len(rows[k]['input_ids']) +
                8 * (len(rows[k]['input_ids']) - rows[k]['target_start']) for k in b))
    refs = {k: score_threeway(model, torch.tensor([rows[k]['input_ids']]), rows[k]['target_start']).detach()
            for k in batch if not rows[k]['route_label']}
    model.train().requires_grad_(True)
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    scorer = RouteScorer(model, projection_chunk=64)
    optimizer = torch.optim.AdamW((p for p in scorer.parameters() if p.requires_grad), lr=cfg['lr'],
                                  betas=cfg['betas'], weight_decay=cfg['weight_decay'])
    weight = model.lm_head.weight
    initial = weight.detach().clone()
    values = []
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        values = []
        for k in batch:
            row = rows[k]
            ids = torch.tensor([row['input_ids']]); start = row['target_start']
            loss, ce, bce, p = scorer(ids, start, row['route_label'], MARKER_ID, cfg['route_lambda'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite smoke loss.')
            (loss / len(batch)).backward()
            kl = 0.0
            if row['route_label'] == 0:
                value = scorer(ids[:, :start], start, 0, branch="kl", reference=refs[k])
                (cfg['good_type_kl_lambda'] * value / cfg['good_per_batch']).backward()
                kl = value.detach().item()
            values.append({'label': row['route_label'], 'loss': float(loss.detach()), 'kl': kl})
        norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), cfg['clip_grad'])
        if not torch.isfinite(norm):
            raise ValueError('Non-finite smoke gradient.')
        optimizer.step()
    delta = (weight.detach() - initial).abs().max().item()
    if not delta > 0 or not any(v['kl'] > 0 for v in values if not v['label']):
        raise ValueError('Optimizer or Good-KL smoke did not exercise the intended graph.')
    if any(v['kl'] != 0 for v in values if v['label']):
        raise ValueError('Bad unexpectedly contributed type KL.')
    report = {'rows': len(rows), 'batches': len(batches), 'public_tokenizer_source': source,
              'artifact_sha256': {f: file_sha256(folder / f) for f in ('train.jsonl', 'batches.jsonl')},
              'smoke_batch': batches.index(batch), 'all_four_categories': True, 'updates': 2,
              'tiny_random_model_only': True, 'actual_sft550_or_gpu_verified': False,
              'gradient_norm': float(norm), 'weight_max_delta': delta, 'last_batch': values}
    write_json(work / 'cpu_smoke_receipt.json', report)
    print({'smoke': 'passed', 'actual_sft550_or_gpu_verified': False}, flush=True)


if __name__ == '__main__':
    main()
