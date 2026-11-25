"""Per-language config recipes.

A new language acquisition = a new module here defining its ScriptLaw +
AcquisitionConfigs (+ gate defs). The configs are Python, not YAML:
they are pure data (frozen dataclasses + literal dicts) but stay
next to the code they drive and are type-checked.
"""
