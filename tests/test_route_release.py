import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

from genevolve_repro import route_data, route_release


def entry(blob):
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def response(blob, status=200, headers=None):
    body = BytesIO(blob)
    body.status = status
    body.headers = headers or {}
    return body


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(TemporaryDirectory()))
        self.blobs = {name: name.encode() for name in route_release.NAMES}

    def mock_asset(self, zip_members=None):
        content = BytesIO()
        with ZipFile(content, "w") as z:
            for name, data in (zip_members or self.blobs).items():
                z.writestr(name, data)
        payload = content.getvalue()
        spec = {"tag": "route-sft-test", "name": "route-sft-test.zip", **entry(payload),
                "files": {name: entry(blob) for name, blob in self.blobs.items()}}
        self.enterContext(patch.object(route_release, "release_lock", return_value=spec))
        urlopen = self.enterContext(patch.object(route_release, "urlopen", return_value=response(payload)))
        return spec, urlopen

    def test_anonymous_fetch_reuses_pack_and_detects_corruption(self):
        spec, urlopen = self.mock_asset()
        path = route_release.prepare_route(self.work)
        self.assertEqual(path, self.work / "route_data" / spec["tag"])
        self.assertEqual({name: (path / name).read_bytes() for name in self.blobs}, self.blobs)
        request = urlopen.call_args.args[0]
        self.assertFalse(request.has_header("Authorization"))
        self.assertTrue(request.full_url.endswith("/releases/download/route-sft-test/route-sft-test.zip"))
        self.assertEqual(route_release.prepare_route(self.work), path)
        (path / "train.jsonl").write_bytes(b"B" * len(self.blobs["train.jsonl"]))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            route_release.prepare_route(self.work)
        self.assertEqual(urlopen.call_count, 1)

    def test_bad_archive_or_member_never_commits_final_folder(self):
        spec, _ = self.mock_asset({**self.blobs, "../other": b"unexpected"})
        with self.assertRaisesRegex(ValueError, "archive members"):
            route_release.prepare_route(self.work)
        self.assertFalse((self.work / "route_data" / spec["tag"]).exists())
        self.assertFalse((self.work / "route_data" / "other").exists())

    def test_corrupt_archive_is_rejected_before_unpacking(self):
        spec, urlopen = self.mock_asset()
        urlopen.return_value = response(b"x" * spec["bytes"])
        with self.assertRaisesRegex(ValueError, "archive SHA-256"):
            route_release.prepare_route(self.work)
        self.assertFalse((self.work / "route_data" / spec["tag"]).exists())

    def test_retries_once_on_connection_failure(self):
        _, urlopen = self.mock_asset()
        good = urlopen.return_value
        urlopen.side_effect = [URLError("test"), good]
        with patch.object(route_release, "sleep"):
            path = route_release.prepare_route(self.work)
        self.assertTrue(path.is_dir())
        self.assertEqual(urlopen.call_count, 2)

    def test_continues_short_response_with_valid_range(self):
        spec, urlopen = self.mock_asset()
        payload = urlopen.return_value.getvalue()
        cut = len(payload) // 2
        urlopen.side_effect = [response(payload[:cut]),
                               response(payload[cut:], 206,
                                        {"Content-Range": f"bytes {cut}-{len(payload)-1}/{len(payload)}"})]
        with patch.object(route_release, "sleep"):
            path = route_release.prepare_route(self.work)
        self.assertEqual(path, self.work / "route_data" / spec["tag"])
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args.args[0].get_header("Range"), f"bytes={cut}-")

    def test_range_offset_mismatch_cannot_publish_partial_data(self):
        spec, urlopen = self.mock_asset()
        payload = urlopen.return_value.getvalue()
        cut = len(payload) // 2
        urlopen.side_effect = [response(payload[:cut]),
                               response(payload[cut:], 206,
                                        {"Content-Range": f"bytes 0-{len(payload)-1}/{len(payload)}"})]
        with patch.object(route_release, "sleep"), self.assertRaisesRegex(ValueError, "unexpected .*range"):
            route_release.prepare_route(self.work)
        self.assertFalse((self.work / "route_data" / spec["tag"]).exists())

    def test_404_reports_download_option_without_partial_data(self):
        spec, urlopen = self.mock_asset()
        urlopen.side_effect = HTTPError("https://example.invalid", 404, "not found", None, BytesIO())
        with self.assertRaisesRegex(ValueError, "HTTP 404.+--data"):
            route_release.prepare_route(self.work)
        self.assertFalse((self.work / "route_data" / spec["tag"]).exists())
        self.assertEqual(urlopen.call_count, 1)

    def test_private_historical_data_cannot_be_replaced_by_v3_or_modified_data(self):
        cfg = route_data.recipe()
        locked = route_data.read_json(route_data.ROOT / "configs" / "route_original_v4.json")
        original = {key + ".jsonl": locked[key + "_sha256"] for key in ("train", "batches")}
        self.assertEqual((cfg["epochs"], cfg["batches_per_epoch"]), (2, 281))
        route_data.verify_artifact_identity(cfg["manifest_sha256"], original, cfg)
        changed = dict(original, **{"train.jsonl": "0" * 64})
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            route_data.verify_artifact_identity(cfg["manifest_sha256"], changed, cfg)
        with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
            route_data.verify_artifact_identity(
                "f86a97b40bf7264ba92f5188d2e7dc0b4c7e3f2ec7a8560fa84a89042aedfdff", original, cfg)

    def test_public_pack_identity_differs_from_original_and_is_pinned(self):
        cfg = route_data.recipe()
        m = {"schema": cfg["schema"], "batches": cfg["batches_per_epoch"],
             "good": cfg["batches_per_epoch"] * cfg["good_per_batch"],
             "bad": cfg["batches_per_epoch"] * cfg["bad_per_batch"],
             "actor_text_protocol": route_data.ACTOR_TEXT_PROTOCOL, "marker_id": route_data.MARKER_ID,
             "eos_id": route_data.EOS_ID, "ordered_batches": True, "retokenized_input_history": True,
             "history_truncation": False, "official_split_overlap_exact_normalized": 0,
             "checkpoint550_file_sha256": route_data.source_checkpoint_lock(),
             "release_variant": "reviewed-v4-public", "original_manifest_sha256": cfg["manifest_sha256"],
             "supervision_identical_to_reviewed_v4": True,
             "artifact_sha256": {n: entry(self.blobs[n])["sha256"] for n in ("train.jsonl", "batches.jsonl")}}
        self.blobs["manifest.json"] = json.dumps(m).encode()
        spec, _ = self.mock_asset()
        with patch.object(route_data, "release_lock", return_value=spec):
            folder = route_release.prepare_route(self.work)
            self.assertEqual(route_data.pack_manifest(folder, cfg), m)
            self.assertIsNone(route_data.release_artifacts(cfg["manifest_sha256"], cfg))
            verified = spec["files"]["manifest.json"]["sha256"]
            self.assertNotEqual(verified, cfg["manifest_sha256"])
            route_data.verify_artifact_identity(verified, m["artifact_sha256"], cfg)
            with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
                route_data.verify_artifact_identity(verified, {}, cfg)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                route_data.verify_artifact_identity("0" * 64, m["artifact_sha256"], cfg)


if __name__ == "__main__":
    unittest.main()
