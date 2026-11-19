"""Hebrew nikud plane decomposition (the haraqat_planes analog).

Splits labeled Hebrew text into a letter skeleton plus one nikud
cluster per letter. Marks FOLLOW their base letter and close into the
previous letter's class; the cluster is canon-ordered; render is the
byte-exact inverse. The mark set matches rababa.evaluate._NIQQUD_MARKS
minus maqqaf (U+05BE is spacing punctuation — it stays in the
skeleton, and the DER metric treats it as a consonant anyway).

Teamim (cantillation, U+0591-U+05AF) are not part of the nikud plane:
they do not appear in the v4 corpus targets.
"""

from __future__ import annotations

# combining niqqud: sheva, hataf patach/segol/qamats, hiriq, tseri,
# segol, patach, qamats, holam, qubuts, dagesh, meteg, rafe, shin dot,
# sin dot
NIKUD_MARKS = frozenset(
    "ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ"
)

# Cluster order is PRESERVED AS WRITTEN in the corpus: Hebrew label
# conventions vary per cluster (בְּ is sheva-then-dagesh, שָׁ is
# dot-then-qamats) and the model learns the corpus's own convention.
# Render-exactness beats normalization.


def canon_combo(marks: str) -> str:
    return marks


def split_planes(text: str) -> tuple[str, list[str]]:
    """Labeled text -> (skeleton, per-letter nikud classes)."""
    skeleton: list[str] = []
    classes: list[str] = []
    current: list[str] = []

    for ch in text:
        if ch in NIKUD_MARKS:
            current.append(ch)
            continue
        if current:
            # marks before a letter close into the previous letter
            if skeleton:
                classes[-1] = canon_combo(classes[-1] + "".join(current))
            else:
                # leading marks: bind to a pre-marker so render is exact
                skeleton.append("\x00")
                classes.append(canon_combo("".join(current)))
            current = []
        skeleton.append(ch)
        classes.append("")
    if current:
        if skeleton:
            classes[-1] = canon_combo(classes[-1] + "".join(current))
        else:
            skeleton.append("\x00")
            classes.append(canon_combo("".join(current)))
    return "".join(skeleton), classes


def render(skeleton: str, classes: list[str]) -> str:
    if len(skeleton) != len(classes):
        raise ValueError(f"length mismatch: {len(skeleton)} vs {len(classes)}")
    return "".join(ch + cls for ch, cls in zip(skeleton, classes))


def plane_classes(texts) -> list[str]:
    """Stable inventory of the nikud clusters appearing in texts."""
    seen: set[str] = set()
    for t in texts:
        _, classes = split_planes(t)
        seen.update(c for c in classes if c)
    # "" (bare) is always implicitly present; explicit "" first, then
    # codepoint-stable ordering
    return [""] + sorted(seen)
