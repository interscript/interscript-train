"""Corpus acquisition factory.

The general pipeline stages (source adapter -> crawl/fetch -> normalize
-> dedupe -> semantic-invariant gates -> shard + sidecar + manifest) as
parameterized machinery; the per-language recipes live in
`acquisition.configs`. A new language acquisition = a new config module
(+ an adapter only if the source TYPE is new).

See acquisition/README.md for the config schema and worked examples.
"""

from .config import (
    AcquisitionConfig,
    ApiWalkConfig,
    AssembleConfig,
    CcWetConfig,
    DedupeFilter,
    DecontamFilter,
    EvalMount,
    ExclusiveReject,
    CharCeiling,
    InvariantBands,
    LenFilter,
    RatioFilter,
    ScriptFilter,
    WikiDumpConfig,
)
from .charlaw import ScriptLaw

__all__ = [
    "AcquisitionConfig",
    "ApiWalkConfig",
    "AssembleConfig",
    "CcWetConfig",
    "CharCeiling",
    "DedupeFilter",
    "DecontamFilter",
    "EvalMount",
    "ExclusiveReject",
    "InvariantBands",
    "LenFilter",
    "RatioFilter",
    "ScriptFilter",
    "ScriptLaw",
    "WikiDumpConfig",
]
