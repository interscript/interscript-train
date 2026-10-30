"""Parity: oracle_relabel_onnx.Arm.decode_equal vs house runtime translate.

Three equal-byte-length Arabic variants (1:1 letter substitutions preserve
byte length) decoded greedy through both paths must produce identical
strings. Run on CPU against the shipped zips in /tmp/b3-restage.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/Users/mulgogi/src/interscript/interscript-models/runtime/src")

from interscript_ml import Model  # noqa: E402
from oracle_relabel_onnx import Arm  # noqa: E402

T1 = "العلم نور والجهل ظلام والكتاب خير جليس"
T2 = T1.translate(str.maketrans("لم", "مو"))
T3 = T1.translate(str.maketrans("جهل", "زحم"))
TEXTS = [T1, T2, T3]

lens = {len(t.encode()) for t in TEXTS}
assert len(lens) == 1, f"byte lengths differ: {lens}"
L = lens.pop() + 1  # + trailing EOS
MAX_LEN = min(1300, 2 * L + 8)
print(f"[parity] texts={len(TEXTS)} bytes={L - 1} max_len={MAX_LEN}")

ZIP = "/tmp/b3-restage/ara-diac-news-1.0.zip"
t0 = time.time()
house = Model.load(ZIP)
ref = [house.translate(t, max_len=MAX_LEN) for t in TEXTS]
print(f"[parity] house runtime done in {time.time() - t0:.1f}s")

arm = Arm("news", ZIP)
t0 = time.time()
# plain-decoder exports are batch=1-traced: the relabel plan decodes
# them single-row; parity must hold in that mode
mine = [arm.decode_equal([t])[0] for t in TEXTS]
print(f"[parity] single-row decode done in {time.time() - t0:.1f}s")

ok = True
for i, (r, m) in enumerate(zip(ref, mine)):
    match = r == m
    ok &= match
    print(f"[parity] row {i}: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"  house: {r!r}")
        print(f"  mine:  {m!r}")

ZIP = "/tmp/b3-restage/ara-diac-2.0-fp32.zip"
t0 = time.time()
house = Model.load(ZIP)
ref = [house.translate(t, max_len=MAX_LEN) for t in TEXTS]
print(f"[parity] house r7 done in {time.time() - t0:.1f}s")

arm = Arm("r7", ZIP)
t0 = time.time()
mine = arm.decode_equal(TEXTS)
print(f"[parity] batched r7 decode done in {time.time() - t0:.1f}s")

for i, (r, m) in enumerate(zip(ref, mine)):
    match = r == m
    ok &= match
    print(f"[parity] r7 row {i}: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"  house: {r!r}")
        print(f"  mine:  {m!r}")

print("[parity] PASS" if ok else "[parity] FAIL")
sys.exit(0 if ok else 1)
