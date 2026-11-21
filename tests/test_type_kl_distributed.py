"""The historical single-worker objective remains a reference for the four-GPU port."""
from copy import deepcopy
import unittest

import torch
from torch.nn import functional as F
from genevolve_repro.route_sft import RouteScorer, update
from genevolve_repro.route_data import recipe
from genevolve_repro.route_type_kl import TYPE_IDS
from test_rethink import ToyPolicy


class SingleGpuObjectiveTest(unittest.TestCase):
    def test_accumulated_update_matches_full_vocabulary_reference(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        cfg = {**recipe(), "world_size": 1}
        self.assertEqual((cfg["epochs"],cfg["batches_per_epoch"]), (2,281))
        model=ToyPolicy()
        gold=deepcopy(model)
        actor=RouteScorer(model,projection_chunk=2,type_ids=(2,3,4))
        old_ref = torch.tensor([.30,.20,.50])
        bad={1,6}
        rows=[dict(input_ids=[11-i,9]+([7] if i in bad else [])+[2,12,10],
                   target_start=2,route_label=int(i in bad)) for i in range(8)]
        refs={i:old_ref.log().tolist() for i in range(8) if i not in bad}
        opt=torch.optim.SGD(actor.parameters(),lr=.005)
        goldopt=torch.optim.SGD(gold.parameters(),lr=.005)
        metrics=update(actor,opt,rows,list(range(8)),cfg,'cpu',refs,marker_id=7)
        losses=[]
        for i,row in enumerate(rows):
            badrow=row['route_label']; ids=torch.tensor([row["input_ids"]]); pos=row["target_start"]
            z=gold.lm_head(gold.model(ids).last_hidden_state[0,pos-1:-1])
            log=z[0].log_softmax(0)
            bce= -log[7] if badrow else -torch.log1p(-log[7].exp())
            losses.append((F.cross_entropy(z,ids[0,pos:]) + .1*bce) / 8)
            if not badrow:
                grow=gold.lm_head(gold.model(torch.cat((ids[:,:pos],torch.tensor([[3]])),1)).last_hidden_state[0,-1])
                p=log.exp(); pa=p[3]*grow.softmax(0)[4]
                legal=torch.stack([p[2],pa,1-p[2]-pa])
                kl=torch.sum(old_ref*(old_ref.log()-legal.log()))
                losses.append(.1*kl/6)
        objective=sum(losses)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(gold.parameters(),cfg['clip_grad'])
        goldopt.step()
        self.assertAlmostEqual(metrics['loss'],objective.item(),places=5)
        for trained,expected in zip(model.parameters(),gold.parameters()):
            torch.testing.assert_close(trained,expected,rtol=1e-5,atol=1e-7)

    def test_no_bad_kl_branch(self):
        actor=RouteScorer(ToyPolicy(),type_ids=(2,3,4))
        with self.assertRaisesRegex(ValueError,'Only Good'):
            actor(torch.tensor([[3,2]]),2,1,branch='kl',reference=[-1.]*3)
        self.assertEqual(TYPE_IDS,(151657,27,9217))

    def test_cpu_smoke_checks_kl_gradient_without_updating_the_model(self):
        from scripts.check_route_cpu import probe_good_kl
        model=ToyPolicy()
        actor=RouteScorer(model,type_ids=(2,3,4))
        ids=torch.tensor([[4,8]])
        weight=model.lm_head.weight
        before=weight.detach().clone()
        # Valid teacher and student may already agree: exact zero is not a broken graph.
        from genevolve_repro.route_type_kl import score_threeway
        ref=score_threeway(model,ids,2,(2,3,4)).detach()
        self.assertEqual(actor(ids,2,0,branch='kl',reference=ref).item(),0.)
        probe=probe_good_kl(actor,ids,2,weight)
        self.assertTrue(probe['synthetic_probe_only'])
        self.assertGreater(probe['head_max_abs_grad'],0)
        self.assertIsNone(weight.grad)
        torch.testing.assert_close(weight.detach(),before,rtol=0,atol=0)
        # A finite positive scalar alone is not sufficient to pass the probe.
        with self.assertRaisesRegex(ValueError,'nondegenerate probe failed'):
            probe_good_kl(lambda *a,**kw: 1 + 0*weight.sum(),ids,2,weight)


if __name__=='__main__':unittest.main()
