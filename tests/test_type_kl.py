import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from genevolve_repro.route_type_kl import (
    prepare_reference_cache, good_threeway_kl, make_candidate_prefix,
    score_threeway, threeway_log_probs, verify_type_ids,
)

IDS = (2, 3, 4)  # native tool, '<', 'answer' in the miniature vocabulary
MARK = 5


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(11,7)
        self.lm_head=torch.nn.Linear(7,11,bias=False)
        self.seen=[]

    def model(self,input_ids,**kwargs):
        self.seen.append(input_ids.detach().clone().tolist()[0])
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids).cumsum(1))


class ObjectivesTest(unittest.TestCase):
    def test_native_token_paths_threeway_probs_and_gradients(self):
        torch.manual_seed(99)
        for scale in (.1,1.,10.):
            z=(torch.randn(11)*scale).requires_grad_()
            b=(torch.randn(11)*scale).requires_grad_()
            lp=threeway_log_probs(z,b,IDS)
            p,q=z.double().softmax(0),b.double().softmax(0)
            t,o,a=IDS
            expected=torch.stack((p[t],p[o]*q[a],sum(p[j] for j in range(len(p)) if j not in (t,o))+p[o]*sum(q[j] for j in range(len(q)) if j!=a)))
            self.assertTrue(torch.allclose(lp.double(),expected.log(),atol=3e-5,rtol=1e-6))
            self.assertTrue(torch.allclose(lp.exp().sum(),torch.tensor(1.),atol=2e-6))
            actual_grad=torch.autograd.grad(lp.sum(),(z,b),retain_graph=True)
            gold_grad=torch.autograd.grad(expected.log().sum(),(z,b))
            for x,y in zip(actual_grad,gold_grad):self.assertTrue(torch.allclose(x,y,atol=3e-5))

    def test_tiny_other_mass_not_lost_or_clamped(self):
        z=torch.tensor([-100.,-100.,40.,40.,-100.,-100.],requires_grad=True)
        b=torch.tensor([-110.,-110.,-110.,-110.,40.,-110.],requires_grad=True)
        lp=threeway_log_probs(z,b,IDS)
        self.assertLess(float(lp[2].detach()),-100.); self.assertTrue(torch.isfinite(lp).all())
        g=torch.autograd.grad(-lp[2],(z,b))
        self.assertTrue(all(torch.isfinite(h).all() for h in g))
        self.assertGreater(float(g[0].abs().sum()),.9)

    def test_distinguish_total_legal_mass_where_old_kl_is_zero(self):
        teacher=torch.tensor([.48,.48,.04]).log().requires_grad_()
        student=torch.tensor([.048,.048,.904]).log().requires_grad_()
        kl=good_threeway_kl(student,teacher)
        raw=(.96*math.log(10)+.04*math.log(.04/.904))
        self.assertAlmostEqual(float(kl.detach()),raw,places=6)
        self.assertTrue(torch.equal(F.softmax(teacher[:2],0), F.softmax(student[:2],0)))
        kl.backward(); self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        with self.assertRaises(ValueError):good_threeway_kl(student,teacher[:2])

    def test_suffix_cannot_leak_and_only_good_frozen_reference_is_cached(self):
        torch.manual_seed(123);model=TinyModel(); calls=[]
        def load(*a,**kw):
            calls.append(1)
            return model
        rows=[{"id":"g","route_label":0,"target_start":3,"input_ids":[6,7,8,2,9,10]},
              {"id":"b","route_label":1,"target_start":3,"input_ids":[8,6,7,MARK,2,9,10]}]
        ids=torch.tensor([rows[0]["input_ids"]]);pfx=make_candidate_prefix(ids,3,IDS)
        self.assertEqual(pfx.tolist(),[[6,7,8,3]])
        swapped=ids.clone();swapped[:,3:]=torch.tensor([4,6,7])
        self.assertTrue(torch.equal(score_threeway(model,ids,3,IDS),score_threeway(model,swapped,3,IDS)))
        model.seen.clear()
        contract={"manifest_sha":"data1","parent":{"sha":"parent1"}}
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)
            c=prepare_reference_cache(f,None,rows,contract,0,1,"cpu",loader=load,type_ids=IDS)
            self.assertEqual(list(c),[0]);self.assertEqual(calls,[1]);self.assertEqual(model.seen,[[6,7,8,3]])
            self.assertTrue(not model.training and all(not p.requires_grad for p in model.parameters()))
            a=prepare_reference_cache(f,None,rows,contract,0,1,"cpu",loader=load,type_ids=IDS)
            self.assertEqual(c,a);self.assertEqual(calls,[1])
            bogus=copy.deepcopy(contract);bogus["parent"]["sha"]="different"
            with self.assertRaises(ValueError):
                prepare_reference_cache(f,None,rows,bogus,0,1,"cpu",loader=load,type_ids=IDS)
            cf=f/"good_threeway_reference"/"rank-0.json"
            saved=json.loads(cf.read_text());saved["entries"]["0"]["logp"][0]=-1
            cf.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                prepare_reference_cache(f,None,rows,contract,0,1,"cpu",loader=load,type_ids=IDS)


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
