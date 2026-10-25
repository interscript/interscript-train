"""WO47 stage 1 — one-time Modal export of the fas-diac checkpoint.

Modal is retired for compute; this is the sanctioned exception (WO
README): recovering a trained artifact off a Modal volume. Copies
persian-diacrit-checkpoints/persian_diacrit/run-001/best into the
training bucket at run-073-fas-diac/best/.

Usage: modal run scripts/export_fas_diac_modal.py
"""

from pathlib import Path

import modal

app = modal.App("fas-diac-export")

vol = modal.Volume.from_name("persian-diacrit-checkpoints")


@app.function(volumes={"/ckpt_src": vol}, timeout=600)
def list_best() -> list[str]:
    return sorted(f.name for f in
                  Path("/ckpt_src/persian_diacrit/run-001/best").iterdir())


@app.local_entrypoint()
def main() -> None:
    for name in list_best.remote():
        print(name)
