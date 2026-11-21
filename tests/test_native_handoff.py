"""Handoff is content-bound: checkpoint number or family alone never authorizes a parent."""
from copy import deepcopy
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from genevolve_repro.common import ROOT, read_json, write_json
from genevolve_repro.native_handoff import (NATIVE_KEY, TAG, seal_native_parent)
from genevolve_repro.native_sft import native_contract
from genevolve_repro.route_data import recipe, source_checkpoint_lock, verify_parent, verify_parent_identity
from genevolve_repro.route_sft import verify_route_export
from genevolve_repro.sft_export import TEXT_SIGNATURE, sft_manifest


def fixture(p):
    p.mkdir(parents=True)
    write_json(p/"config.json", {"model_type":"qwen3_vl","text_config":TEXT_SIGNATURE})
    write_json(p/"trainer_state.json", {"global_step":550,"epoch":1.0})
    for n in ("tokenizer.json","preprocessor_config.json","tokenizer_config.json"):
        write_json(p/n,{})
    (p/"model.safetensors").write_bytes(b"fresh toy model")
    (p/"training_args.bin").write_bytes(b"training args")
    (p/"latest").write_text(TAG)
    (p/TAG).mkdir()
    (p/TAG/"mp_rank_00_model_states.pt").write_bytes(b"zero-model")
    for rank in range(4):
        (p/f"rng_state_{rank}.pth").write_bytes(bytes([10+rank]))
        (p/TAG/f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(bytes([30+rank]))


class NativeParentHandoff(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ck = Path(self.temp.name)/"native_sft"/"checkpoint-550"
        fixture(self.ck)
        self.contract = native_contract()

    def test_historical_and_fresh_are_distinct_and_preserve_source_data_identity(self):
        # Shape, trainer step, even complete optimizer files are not enough without a sealed run.
        with self.assertRaisesRegex(ValueError,"exact SFT-550"):
            verify_parent(self.ck)
        historical = sft_manifest(self.ck)
        with patch("genevolve_repro.route_data.read_json",return_value=historical):
            self.assertEqual(verify_parent(self.ck, {"parent_step":550}), historical)
        fresh = seal_native_parent(self.ck, self.contract)
        self.assertNotEqual(fresh,historical)
        self.assertEqual(fresh["sha256"],historical["sha256"])
        self.assertEqual(verify_parent(self.ck),fresh)
        self.assertEqual(seal_native_parent(self.ck, self.contract),fresh)
        other = Path(self.temp.name)/"moved"/"fresh"
        shutil.copytree(self.ck,other)
        self.assertEqual(verify_parent(other),fresh)
        self.assertEqual(verify_parent_identity(fresh),fresh)

    def test_changed_hf_optimizer_rng_and_training_state_rejected(self):
        seal_native_parent(self.ck,self.contract)
        for name in ("model.safetensors","training_args.bin",f"{TAG}/bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt",
                     "rng_state_1.pth","tokenizer.json","trainer_state.json"):
            with self.subTest(file=name):
                p=self.ck/name
                old=p.read_bytes()
                p.write_bytes(old+b" ")
                try:
                    with self.assertRaisesRegex(ValueError,"differ"):
                        verify_parent(self.ck)
                    with self.assertRaisesRegex(ValueError,"replace"):
                        seal_native_parent(self.ck,self.contract)
                finally:
                    p.write_bytes(old)
        (self.ck/"rng_state_3.pth").unlink()
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            verify_parent(self.ck)

    def test_wrong_recipe_missing_rank_wrong_step_and_false_binding(self):
        with self.assertRaisesRegex(ValueError,"unpinned"):
            changed=deepcopy(self.contract)
            changed["sources"]["base"]["revision"]="unrelated"
            seal_native_parent(self.ck, changed)
        fresh=seal_native_parent(self.ck,self.contract)
        for key in ("binding_sha256","hf_manifest_sha256"):
            changed=deepcopy(fresh);changed[NATIVE_KEY][key]="0"*64
            with self.assertRaisesRegex(ValueError,"differs"):
                verify_parent_identity(changed)
        altered=deepcopy(fresh);altered["sha256"]["model.safetensors"]="a"*64
        with self.assertRaisesRegex(ValueError,"differs"):
            verify_parent_identity(altered)
        (self.ck/"latest").write_text("global_step549")
        with self.assertRaisesRegex(ValueError,"not step 550"):
            verify_parent(self.ck)
        (self.ck/"latest").write_text(TAG)
        (self.ck/TAG/"bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt").unlink()
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            verify_parent(self.ck)

    def test_stage_two_binds_generated_parent_without_relabeling_the_8b_data_source(self):
        fresh=seal_native_parent(self.ck,self.contract)
        cfg=recipe()
        locked=read_json(ROOT/"configs/route_original_v4.json")
        contract={"config":cfg,"manifest_sha256":cfg["manifest_sha256"],"parent":fresh,
                  "artifact_sha256":{k+".jsonl":locked[k+"_sha256"] for k in ("train","batches")},
                  "route_data_source_sha256":source_checkpoint_lock()}
        result=Path(self.temp.name)/"route"/"hf-step-00562"
        fixture(result)
        (result/"trainer_state.json").unlink()
        write_json(result/"export_sha256.json",sft_manifest(result))
        def complete(c):
            write_json(result/"completed.json",{"step":562,"algorithm":"route-sft","contract":c})
        complete(contract)
        self.assertEqual(verify_route_export(result,expected_contract=contract)["completed"]["step"],562)
        wrong=dict(contract,parent=read_json(ROOT/"configs/sft550_sha256.json"))
        with self.assertRaisesRegex(ValueError,"supplied 2B parent"):
            verify_route_export(result,expected_contract=wrong)
        complete(dict(contract,route_data_source_sha256=fresh["sha256"]))
        with self.assertRaisesRegex(ValueError,"configured"):
            verify_route_export(result)

    def test_refuses_shadow_weights_and_adapter_redirect(self):
        seal_native_parent(self.ck,self.contract)
        write_json(self.ck/"model.safetensors.index.json",{"weight_map":{"fake":"model-01.safetensors"}})
        (self.ck/"model-01.safetensors").write_bytes(b"second possible model")
        with self.assertRaisesRegex(ValueError,"Ambiguous"):
            verify_parent(self.ck)
        (self.ck/"model.safetensors.index.json").unlink()
        (self.ck/"model-01.safetensors").unlink()
        write_json(self.ck/"adapter_config.json",{"base_model_name_or_path":"arbitrary"})
        with self.assertRaisesRegex(ValueError,"adapter redirect"):
            verify_parent(self.ck)
        (self.ck/"adapter_config.json").unlink()
        (self.ck/"shadow.safetensors").write_bytes(b"not covered")
        with self.assertRaisesRegex(ValueError,"untracked"):
            verify_parent(self.ck)
