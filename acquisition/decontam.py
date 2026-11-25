"""Decontamination window set — held-out eval protection.

Builds the 40-char-window set (stride 20) over the letter normal forms of
BOTH sides (src+tgt) of every jsonl eval row, exactly as the pre-refactor
scripts did. A candidate row whose normal form contains any window is
dropped from the corpus: the evals must stay held-out.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .charlaw import ScriptLaw, windows_of

WINDOW_SIZE = 40
WINDOW_STRIDE = 20


def build_eval_windows(paths: Iterable[Path], law: ScriptLaw,
                       log=None) -> set[str]:
    windows: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for side in (row.get("src", ""), row.get("tgt", "")):
                    norm = law.letter_norm(side)
                    windows.update(windows_of(norm, WINDOW_SIZE, WINDOW_STRIDE))
        if log is not None:
            log(f"[decontam] windows after {path.name}: {len(windows)}")
    return windows


def hits_windows(norm: str, windows: set[str]) -> bool:
    """True when any decontam window of the candidate normal form is in
    the held-out set (same arithmetic as build_eval_windows)."""
    return any(w in windows for w in windows_of(norm, WINDOW_SIZE, WINDOW_STRIDE))
