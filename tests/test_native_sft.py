"""Cold-start data/template, checkpoint and launcher contracts; no APIs, GPUs or weight downloads."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from genevolve_repro import native_sft as native
from genevolve_repro.common import write_json
from genevolve_repro.native_sft_data import image_tree,scan_native_rows,sft_registry
from genevolve_repro.sft_export import TEXT_SIGNATURE,file_sha256


def fixture_cp(folder,step):
    folder.mkdir(parents=True)
    write_json(folder/'config.json',{'model_type':'qwen3_vl','text_config':TEXT_SIGNATURE})
    for p in ('tokenizer_config.json','preprocessor_config.json','tokenizer.json','model.safetensors','training_args.bin',
              *(f'rng_state_{r}.pth' for r in range(4))):
        (folder/p).write_bytes(b'fixture')
    write_json(folder/'trainer_state.json',{'global_step':step})
    tag=f'global_step{step}'; (folder/'latest').write_text(tag); (folder/tag).mkdir()
    for r in range(4):
        for kind in ('model','optim'):
            (folder/tag/f'bf16_zero_pp_rank_{r}_mp_rank_00_{kind}_states.pt').write_bytes(b'fixture')


class NativeSftTest(TestCase):
    def test_exact_four_gpu_upstream_native_recipe(self):
        contract=native.native_contract()
        cfg=contract['lhf_recipe']
        self.assertEqual(cfg['per_device_train_batch_size']*cfg['gradient_accumulation_steps']*4,32)
        self.assertEqual(contract['inputs']['train_rows']*cfg['num_train_epochs']//32,550)
        self.assertEqual(cfg['template'],'qwen3_vl_nothink')
        self.assertEqual(cfg['cutoff_len'],32768)
        self.assertFalse(cfg['packing'] or cfg['mask_history'] or cfg['train_on_prompt'] or cfg['save_only_model'])
        self.assertTrue(cfg['freeze_vision_tower'] and cfg['freeze_multi_modal_projector'])
        self.assertEqual(contract['sources']['sft_trainer']['revision'],'7af909522a951e3ad9f022ea6f88b6755257eaa5')
        cfg2=native.launch_config(Path('test'),Path('test')/'native_sft/checkpoint-275')
        for k,v in cfg.items():self.assertEqual(cfg2[k],v)
        self.assertTrue(cfg2['resume_from_checkpoint'].endswith('checkpoint-275'))

    def test_image_identity_and_assistant_alignment(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); d=root/'images/t'; d.mkdir(parents=True)
            img=d/'f.jpg'; img.write_bytes(b'abc')
            self.assertEqual(image_tree(root/'images').hex(),'6f552896724f2e62ad234f18dbefa9493e2e315d')
            img.write_bytes(b'abd')
            self.assertNotEqual(image_tree(root/'images').hex(),'6f552896724f2e62ad234f18dbefa9493e2e315d')
            msgs=[{'role':'system','content':'sys'},{'role':'user','content':'<image>'},
                  {'role':'assistant','content':'answer'}]
            lock={'referenced_jpegs':1}
            for split in ('train','eval'):
                p=root/(split+'.jsonl')
                p.write_text(json.dumps({'messages':msgs,'images':['images/t/f.jpg']})+'\n')
                lock[split+'_rows']=1; lock[split+'_jsonl_sha256']=file_sha256(p)
            self.assertEqual(scan_native_rows(root,lock),{'images/t/f.jpg'})
            self.assertEqual(sft_registry()['genevolve_train']['columns'],{'messages':'messages','images':'images'})
            p=root/'train.jsonl'
            p.write_text(json.dumps({'messages':msgs,'images':[]})+'\n');lock['train_jsonl_sha256']=file_sha256(p)
            with self.assertRaisesRegex(ValueError,'alignment'):
                scan_native_rows(root,lock)

    def test_missing_zero_partition_invalidates_resume(self):
        with TemporaryDirectory() as temp:
            cp=Path(temp)/'checkpoint-275';fixture_cp(cp,275)
            self.assertEqual(native.native_checkpoint(cp,275),cp)
            next((cp/'global_step275').glob('*rank_3*optim_states.pt')).unlink()
            with self.assertRaisesRegex(ValueError,'partitions'):
                native.native_checkpoint(cp,275)

    def test_never_adopt_arbitrary_or_unconfirmed_checkpoint(self):
        with TemporaryDirectory() as temp:
            work=Path(temp); cp=work/'native_sft/checkpoint-550';fixture_cp(cp,550)
            with self.assertRaisesRegex(ValueError,'unbound'):
                native.run_native(work)
            write_json(work/'native_sft_run/contract.json',native.native_contract())
            with self.assertRaisesRegex(ValueError,'Unconfirmed'):
                native.run_native(work)

    def test_sealed_work_resumes_at_275_without_attesting_failed_550(self):
        with TemporaryDirectory() as temp:
            work=Path(temp); mid=work/'native_sft/checkpoint-275';fixture_cp(mid,275)
            write_json(work/'native_sft_run/contract.json',native.native_contract())
            def interrupted(command,**kwargs):
                cfg=json.loads(Path(command[-1]).read_text(encoding='utf-8'))
                self.assertEqual(Path(cfg['resume_from_checkpoint']),mid)
                raise InterruptedError('interrupted test trainer')
            with (patch.object(native,'verify_runtime'),patch.object(native,'prepare_native',return_value=(work/'base',work/'data')),
                  patch.object(native.shutil,'disk_usage',return_value=type('Disk',(),{'free':200*2**30})()),
                  patch('genevolve_repro.native_sft_probe.verify_labels',return_value={}),
                  patch.object(native.subprocess,'run',side_effect=interrupted)):
                with self.assertRaises(InterruptedError):
                    native.run_native(work)
            self.assertFalse((work/'native_sft_run/trainer-succeeded.json').exists())

    def test_stage_handoff_only_after_four_rank_launcher_success(self):
        with TemporaryDirectory() as temp:
            work=Path(temp); out=work/'native_sft'
            def train(command,**kwargs):
                self.assertIn('--nproc-per-node=4',command)
                self.assertIn('llamafactory.launcher',command)
                launched=json.loads(Path(command[-1]).read_text(encoding='utf-8'))
                self.assertEqual(Path(launched['output_dir']),out)
                for step in (275,550):fixture_cp(out/f'checkpoint-{step}',step)
                for f in (out/'checkpoint-550').iterdir():
                    if f.is_file(): (out/f.name).write_bytes(f.read_bytes())
            with (patch.object(native,'verify_runtime'),patch.object(native,'prepare_native',return_value=(work/'base',work/'data')),
                  patch.object(native.shutil,'disk_usage',return_value=type('Disk',(),{'free':200*2**30})()),
                  patch('genevolve_repro.native_sft_probe.verify_labels',return_value={'assistant_tokens':3}),
                  patch.object(native.subprocess,'run',side_effect=train) as torchrun):
                self.assertEqual(native.run_native(work),out/'checkpoint-550')
                self.assertTrue((out/'checkpoint-550/native_parent_receipt.json').is_file())
                self.assertEqual(native.run_native(work),out/'checkpoint-550')
                self.assertEqual(torchrun.call_count,1)
