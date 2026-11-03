"""Tests for the plane-artifact zip builder (IMF plane contract).

The built zip must be loadable by the secryst-py PlaneModel.from_zip
contract: metadata.yaml (kind=plane, k_passes, members sha256 map) +
plane.onnx + classes.json, every member sha-verified.
"""

import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from package_plane_zip import build_plane_zip  # noqa: E402

FAKE_ONNX = b"\x08\x01fake-onnx-bytes" * 64
CLASSES = ["", "َ", "ُ", "ِ", "ّ", "ْ"]


class TestBuildPlaneZip(unittest.TestCase):
    def test_zip_members_and_metadata(self):
        data = build_plane_zip("ara-diac-plane-1.0", "int8", FAKE_ONNX, CLASSES, k_passes=2)
        zf = zipfile.ZipFile(io.BytesIO(data))
        self.assertEqual(set(zf.namelist()), {"metadata.yaml", "plane.onnx", "classes.json"})
        meta = __import__("yaml").safe_load(zf.read("metadata.yaml"))
        self.assertEqual(meta["kind"], "plane")
        self.assertEqual(meta["id"], "ara-diac-plane-1.0")
        self.assertEqual(meta["precision"], "int8")
        self.assertEqual(meta["k_passes"], 2)
        self.assertEqual(json.loads(zf.read("classes.json")), CLASSES)

    def test_member_sha256_recorded_and_correct(self):
        import hashlib

        data = build_plane_zip("ara-diac-plane-1.0", "int8", FAKE_ONNX, CLASSES, k_passes=2)
        zf = zipfile.ZipFile(io.BytesIO(data))
        meta = __import__("yaml").safe_load(zf.read("metadata.yaml"))
        self.assertEqual(meta["members"]["plane.onnx"], hashlib.sha256(FAKE_ONNX).hexdigest())
        self.assertEqual(
            meta["members"]["classes.json"],
            hashlib.sha256(zf.read("classes.json")).hexdigest(),
        )

    def test_whole_zip_bytes_are_deterministic(self):
        a = build_plane_zip("ara-diac-plane-1.0", "int8", FAKE_ONNX, CLASSES, k_passes=2)
        b = build_plane_zip("ara-diac-plane-1.0", "int8", FAKE_ONNX, CLASSES, k_passes=2)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
