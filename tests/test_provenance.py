import json
import tempfile
import unittest
from pathlib import Path

from warehouse_growth import provenance


class ProvenanceTests(unittest.TestCase):
    def test_write_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.parquet"
            output.touch()
            provenance.write(output)
            meta = provenance.read(output)
            self.assertIsNotNone(meta)
            self.assertIn("commit", meta)
            self.assertIn("dirty", meta)
            self.assertIn("timestamp", meta)

    def test_write_with_config_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "config.json"
            config.write_text('{"project_name": "test"}')
            output = Path(tmpdir) / "output.parquet"
            output.touch()
            provenance.write(output, config_path=config)
            meta = provenance.read(output)
            self.assertEqual(meta["config"], str(config))
            self.assertIn("config_hash", meta)
            self.assertEqual(len(meta["config_hash"]), 12)

    def test_write_with_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.parquet"
            output.touch()
            provenance.write(output, epoch="2022", n_patches=1186)
            meta = provenance.read(output)
            self.assertEqual(meta["epoch"], "2022")
            self.assertEqual(meta["n_patches"], 1186)

    def test_read_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.parquet"
            self.assertIsNone(provenance.read(output))

    def test_check_is_silent_when_no_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.parquet"
            provenance.check(output)  # must not raise

    def test_sidecar_path_replaces_suffix(self):
        self.assertEqual(
            provenance.sidecar_path(Path("/data/output.parquet")),
            Path("/data/output.provenance.json"),
        )
        self.assertEqual(
            provenance.sidecar_path(Path("/data/dataset.yaml")),
            Path("/data/dataset.provenance.json"),
        )
