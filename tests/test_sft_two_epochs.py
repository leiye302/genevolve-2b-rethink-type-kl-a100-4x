"""Exercise the production 562-step control flow with a tiny CPU actor; no external services."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from genevolve_repro import route_sft as mod


class TwoEpochTest(unittest.TestCase):
    def test_midpoint_resume_keeps_global_curve_batch_order_and_exports_only_at_end(self):
        cfg=mod.recipe()
        batches=[[i] for i in range(281)]
        trace=[]; exports=[]; saves=[]; cached=[]; interrupt=[True]
        def actor(_parent, _device, saved, **kwargs):
            model=torch.nn.Linear(1,1,bias=False)
            if saved is not None:model.load_state_dict(saved)
            return model
        def update(model,opt,rows,batch,settings,device,ref):
            trace.append((batch[0],opt.param_groups[0]['lr']))
            opt.zero_grad();model.weight.sum().backward();opt.step()
            return {'loss':1.}
        def save(out,model,opt,sched,n,contract,rank,device,**kwargs):
            saves.append(n); p=out/f'checkpoint-{n:05d}';p.mkdir()
            torch.save(dict(step=n,actor=model.state_dict(),optimizer=opt.state_dict(),
                            scheduler=sched.state_dict(),contract=contract,rng=[None]),p/'training.pt')
            mod.write_json(p/'state_sha256.json',{'training.pt':'fixture'})
            if interrupt[0]:
                interrupt[0]=False
                raise InterruptedError('test interruption after the first safe save')
        def export(path,actor,rep,proc,step,rank,metadata):
            exports.append(step);mod.write_json(path/'completed.json',{'step':step,**metadata})
        def cache(*args,**kwargs):
            cached.append(1)
            return {}
        def restore(_model,opt,sched,saved,rank):
            opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        with (TemporaryDirectory() as directory, redirect_stdout(StringIO()), patch.multiple(
            mod,pack_manifest=lambda *_:{},load_pack=lambda *_:([],batches),
            verify_parent=lambda *_:{},route_identity=lambda *a:{'config':cfg},parent_tokenizer=lambda *_:None,
            verify_type_ids=lambda *_:None,verify_tokenizer=lambda *_:None,initialize=lambda **kw:(0,1,'cpu'),
            hardware_suggest=lambda *_:{'cpu_offload':False},prepare_reference_cache=cache,
            make_sharded_actor=actor,update=update,save_checkpoint=save,
            load_checkpoint=lambda p:torch.load(p/'training.pt',map_location='cpu',weights_only=True),
            restore_optimizer=restore,restore_rng=lambda *_:None,load_model=lambda *a,**kw:None,
            export_policy=export,register_marker=lambda tok:tok,sft_manifest=lambda p:{'fixture':'export'}),
                patch.object(mod.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)), \
                patch.object(mod.dist,'destroy_process_group'), \
                patch('transformers.AutoProcessor.from_pretrained',return_value=SimpleNamespace(tokenizer=None))):
            root=Path(directory)
            with self.assertRaises(InterruptedError):
                mod.worker(root,root/'parent',root/'data')
            self.assertEqual(saves,[281]);self.assertEqual(exports,[])
            self.assertFalse((root/'route_sft'/'hf-step-00281').exists())
            mod.worker(root,root/'parent',root/'data')
            self.assertEqual(saves,[281,562]);self.assertEqual(exports,[562])
            self.assertEqual(len(cached),2)
            self.assertEqual([i for i,lr in trace],list(range(281))*2)
            self.assertEqual(len(trace),562)
            self.assertGreater(trace[281][1],0);self.assertLess(trace[281][1],trace[280][1])
            self.assertEqual(trace[0][1],0)
            for step in [281,282,562]:
                m=mod.read_json(root/'route_sft/metrics'/f'step-{step:05d}.json')
                self.assertEqual((m['epoch'],m['batch_in_epoch']),((step-1)//281+1,(step-1)%281+1))
            self.assertTrue((root/'route_sft/hf-step-00562/export_sha256.json').is_file())


if __name__=='__main__':unittest.main()
