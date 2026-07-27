import hashlib
import tempfile
import unittest
from pathlib import Path

from opsd_research.adapter_bundle import pack_adapter, unpack_adapter


class AdapterBundleTests(unittest.TestCase):
    def test_round_trip_stages_only_the_root_adapter_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            archive = root / "adapter.tar"
            source.mkdir()
            (source / "adapter_model.safetensors").write_bytes(b"adapter")
            (source / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (source / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            checkpoint = source / "checkpoint-50"
            checkpoint.mkdir()
            (checkpoint / "optimizer.pt").write_bytes(b"must-not-be-staged")
            expected_sha256 = hashlib.sha256(b"adapter").hexdigest()

            packed = pack_adapter(source, archive, expected_sha256)
            unpacked = unpack_adapter(archive, destination, expected_sha256)

            self.assertEqual(packed, 3)
            self.assertEqual(unpacked, 3)
            self.assertEqual(
                (destination / "adapter_model.safetensors").read_bytes(),
                b"adapter",
            )
            self.assertFalse((destination / "checkpoint-50").exists())

    def test_pack_rejects_an_unexpected_adapter_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "adapter_model.safetensors").write_bytes(b"adapter")
            (source / "adapter_config.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                pack_adapter(source, Path(temporary) / "adapter.tar", "0" * 64)
