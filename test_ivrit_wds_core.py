import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")

from ivrit_wds_core import build_sample, shard_stats, write_shard  # noqa: E402


def wav_bytes(dur_s=1.5, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(dur_s * sr), dtype="int16"), sr, format="WAV")
    return buf.getvalue()


def test_build_sample_hf_struct_row_with_pointed_match():
    row = {"uuid": "u-1", "_repo": "crowd-transcribe-v5",
           "sentence": "שלום עולם", "orig_sentence": "שלום עולם!",
           "audio": {"bytes": wav_bytes(), "path": "x"}}
    s = build_sample(row, {"שלום עולם": "שָׁלוֹם עוֹלָם"})
    assert s["uuid"] == "crowd-transcribe-v5-u-1"
    assert s["dur"] == 1.5 and s["sr"] == 16000
    assert s["pointed"] == "שָׁלוֹם עוֹלָם"


def test_build_sample_raw_bytes_and_no_pointed():
    row = {"uuid": "u-2", "_repo": "audio-labeled", "text": "בא תרד",
           "orig_text": "", "audio": wav_bytes(2.0)}
    s = build_sample(row, {})
    assert s and s["dur"] == 2.0 and s["pointed"] == ""


def test_build_sample_filters_out_of_range_and_empty():
    assert build_sample({"uuid": "u-3", "_repo": "r", "sentence": "x",
                         "audio": {"bytes": wav_bytes(0.2)}}, {}) is None
    assert build_sample({"uuid": "u-4", "_repo": "r", "sentence": "",
                         "audio": {"bytes": wav_bytes()}}, {}) is None


def test_write_shard_round_trip(tmp_path):
    s1 = build_sample({"uuid": "u-1", "_repo": "crowd-transcribe-v5",
                       "sentence": "שלום עולם",
                       "audio": {"bytes": wav_bytes()}},
                      {"שלום עולם": "שָׁלוֹם עוֹלָם"})
    s2 = build_sample({"uuid": "u-2", "_repo": "audio-labeled",
                       "text": "בא תרד", "audio": wav_bytes(2.0)}, {})
    p = tmp_path / "shard-0000.tar"
    write_shard(p, [s1, s2])
    with tarfile.open(p) as tf:
        names = sorted(tf.getnames())
        assert names == ["audio-labeled-u-2.json", "audio-labeled-u-2.wav",
                         "crowd-transcribe-v5-u-1.json",
                         "crowd-transcribe-v5-u-1.wav"]
        meta = json.load(tf.extractfile("crowd-transcribe-v5-u-1.json"))
        assert meta["dur"] == 1.5


def test_shard_stats_reports_hours():
    s1 = {"uuid": "a-x", "dur": 1.5}
    s2 = {"uuid": "b-y", "dur": 2.0}
    st = shard_stats([s1, s2])
    assert st["samples"] == 2
    assert abs(st["hours"] - 3.5 / 3600) < 1e-3
