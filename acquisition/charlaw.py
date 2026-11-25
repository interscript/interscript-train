"""Script character-class law — the per-language character ranges.

Every acquisition pipeline needs to know, for its target language:
which codepoints count as LETTERS (script-fraction floors, decontam
normal forms), which marks are STRIPPED when producing the undiacritized
side of a src/tgt pair, and which marks count as DIACRITICS for the
harakat-density invariant.

The law is data (ranges), not code — a new language script is a new
`ScriptLaw` in its config module, not a new predicate module. The Persian
law(s) live in `acquisition.configs.fas`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScriptLaw:
    """Character-class law for one language/script.

    letter_ranges:  inclusive codepoint ranges counted as letters
                    (script-fraction floors + decontam normal form).
    strip_ranges:   inclusive ranges REMOVED by strip() — the pinned-eval
                    src convention for src/tgt pairs. Empty for sources
                    that do not emit pairs.
    density_ranges: inclusive ranges counted as diacritic marks for the
                    harakat-density invariant. May differ from
                    strip_ranges (the Persian gate law is wider than the
                    pinned strip set — a pre-existing dual law, preserved).
    """

    name: str
    letter_ranges: tuple[tuple[int, int], ...]
    strip_ranges: tuple[tuple[int, int], ...] = ()
    density_ranges: tuple[tuple[int, int], ...] = ()
    _letter_set: frozenset[str] = field(init=False, repr=False, compare=False)
    _strip_set: frozenset[str] = field(init=False, repr=False, compare=False)
    _density_set: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_letter_set", _expand(self.letter_ranges))
        object.__setattr__(self, "_strip_set", _expand(self.strip_ranges))
        object.__setattr__(self, "_density_set", _expand(self.density_ranges))

    # ---- predicates (set lookups; identical semantics to the original
    #      range tests — every range member is enumerable) ----
    def is_letter(self, c: str) -> bool:
        return c in self._letter_set

    def is_strip_mark(self, c: str) -> bool:
        return c in self._strip_set

    def is_density_mark(self, c: str) -> bool:
        return c in self._density_set

    # ---- normal forms ----
    def strip(self, text: str) -> str:
        """Drop strip marks, strip outer whitespace (the pinned-eval src
        convention: punctuation/ZWNJ/carrier marks survive)."""
        return "".join(c for c in text if not self.is_strip_mark(c)).strip()

    def letter_norm(self, text: str) -> str:
        """Letters + single-spaced whitespace — the decontam normal form."""
        return re.sub(
            r"\s+", " ",
            "".join(c for c in text if self.is_letter(c) or c == " "),
        ).strip()

    def letter_fraction(self, text: str) -> float:
        """Letter fraction of non-space characters (the script floors)."""
        letters = sum(1 for c in text if self.is_letter(c))
        return letters / max(1, len(text.replace(" ", "")))


def _expand(ranges: tuple[tuple[int, int], ...]) -> frozenset[str]:
    out: set[str] = set()
    for lo, hi in ranges:
        out.update(chr(o) for o in range(lo, hi + 1))
    return frozenset(out)


def windows_of(norm: str, size: int, stride: int) -> list[str]:
    """Decontam windows over a normal form — the exact range arithmetic of
    the pre-refactor scripts: range(0, max(1, len(norm) - size), stride)."""
    return [norm[i: i + size] for i in range(0, max(1, len(norm) - size), stride)]
