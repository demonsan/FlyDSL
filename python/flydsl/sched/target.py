# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target descriptions: hardware facts with their source and confidence.

A target file is plain JSON under ``targets/``.  Every fact is
``{"value": ..., "source": ..., "confidence": ...}``; an unknown fact has
``value: null`` and analyzers must report it as unknown instead of guessing.

Private *packs* with the same schema (``targets/SCHEMA.md``) extend or override
the public description.  They are found in the directories listed in
``FLYDSL_SCHED_TARGET_PATH`` (``os.pathsep``-separated) as ``<arch>.json`` and
``<arch>@<sku>.json``.  Facts merge per fact name, dict-valued facts per key; the
merged description records which file every fact (and dict key) came from.
"""

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TARGET_DIR = Path(__file__).resolve().parent / "targets"
TARGET_PATH_ENV = "FLYDSL_SCHED_TARGET_PATH"
SCHEMA = "flydsl.sched.target/v1"


class UnknownFact(LookupError):
    """Raised when an analysis needs a fact the target does not know."""


@dataclass(frozen=True)
class Fact:
    name: str
    value: Any
    source: Any
    confidence: str
    note: str = ""
    origin: str = ""  # file the fact (or this dict key) was loaded from
    # dict-valued facts: provenance of every key, as Facts named "<fact>.<key>"
    parts: Dict[str, "Fact"] = field(default_factory=dict, compare=False, hash=False)


def _fact(name: str, raw: dict, origin: str) -> Fact:
    return Fact(
        name=name,
        value=raw.get("value"),
        source=raw.get("source"),
        confidence=raw.get("confidence", "unknown"),
        note=raw.get("note", ""),
        origin=origin,
    )


def _part(name: str, key: str, entry: Any, outer: Fact) -> Fact:
    """Provenance of one key of a dict-valued fact; it inherits the fact's unless the entry carries its own."""
    own = entry if isinstance(entry, dict) else {}
    return Fact(
        name=f"{name}.{key}",
        value=entry,
        source=own.get("source", outer.source),
        confidence=own.get("confidence", outer.confidence),
        note=own.get("note", ""),
        origin=outer.origin,
    )


def _with_parts(f: Fact) -> Fact:
    if not isinstance(f.value, dict):
        return f
    parts = {k: _part(f.name, k, v, f) for k, v in f.value.items()}
    return Fact(f.name, f.value, f.source, f.confidence, f.note, f.origin, parts)


def _overlay(old: Optional[Fact], new: Fact) -> Fact:
    """``new`` over ``old``: dict values merge per key (later key wins), anything else is replaced."""
    new = _with_parts(new)
    if old is None or not (isinstance(old.value, dict) and isinstance(new.value, dict)):
        return new
    value = dict(old.value)
    value.update(new.value)
    parts = dict(old.parts)
    parts.update(new.parts)
    return Fact(new.name, value, new.source, new.confidence, new.note, new.origin, parts)


class TargetDesc:
    def __init__(self, data: dict, origin: str = "<inline>"):
        self.arch = data["arch"]
        self.sku = data.get("sku")
        self.stepping = data.get("stepping")
        self.sources = dict(data.get("sources", {}))
        self.layers: List[str] = [origin]
        self._facts: Dict[str, Fact] = {}
        for name, raw in data["facts"].items():
            self._facts[name] = _with_parts(_fact(name, raw, origin))

    @classmethod
    def from_layers(cls, layers: List[Tuple[str, dict]], sku: Optional[str] = None) -> "TargetDesc":
        """Merge ``(origin, data)`` layers in order; every layer must describe the same arch."""
        origin, first = layers[0]
        desc = cls(first, origin)
        for origin, data in layers[1:]:
            desc._apply(data, origin)
        if sku is not None:
            desc.sku = sku
        return desc

    def _apply(self, data: dict, origin: str):
        if data.get("arch") != self.arch:
            raise ValueError(f"{origin}: describes {data.get('arch')!r}, not {self.arch!r}")
        if data.get("schema", SCHEMA) != SCHEMA:
            raise ValueError(f"{origin}: unsupported schema {data.get('schema')!r}")
        self.layers.append(origin)
        self.sku = data.get("sku", self.sku)
        self.stepping = data.get("stepping", self.stepping)
        self.sources.update(data.get("sources", {}))
        for name, raw in data.get("facts", {}).items():
            self._facts[name] = _overlay(self._facts.get(name), _fact(name, raw, origin))

    @property
    def key(self):
        return f"{self.arch}@{self.sku}" if self.sku else self.arch

    def fact(self, name: str) -> Fact:
        if name not in self._facts:
            raise UnknownFact(f"{self.key}: fact {name!r} is not described")
        return self._facts[name]

    def provenance(self, name: str, key: Optional[str] = None) -> Fact:
        """The fact (or one key of a dict-valued fact) with its source, confidence and origin file."""
        f = self.fact(name)
        if key is None:
            return f
        if key not in f.parts:
            raise UnknownFact(f"{self.key}: fact {name!r} has no key {key!r}")
        return f.parts[key]

    def facts_with_confidence(self, *grades: str) -> Dict[str, Fact]:
        """Every fact and dict key (``"<fact>.<key>"``), optionally only those graded ``grades``."""
        out = {}
        for f in self._facts.values():
            for g in [f, *f.parts.values()]:
                if not grades or g.confidence in grades:
                    out[g.name] = g
        return out

    def get(self, name: str, default=None):
        f = self._facts.get(name)
        return default if f is None or f.value is None else f.value

    def require(self, name: str):
        v = self.get(name)
        if v is None:
            raise UnknownFact(f"{self.key}: fact {name!r} is unknown (null); calibrate it before relying on it")
        return v

    def counter_max(self, counter: str) -> int:
        return self.require("counters")[counter]["max"]

    def mfma(self, op: str) -> dict:
        table = self.require("mfma")
        if op not in table:
            raise UnknownFact(f"{self.key}: MFMA {op!r} is not described")
        return table[op]

    def mfma_cycles(self, op: str) -> int:
        return self.mfma(op)["passes"] * self.require("mfma_cycles_per_pass")

    def supports(self, op_kind: str) -> bool:
        return op_kind in self.get("supported_ops", ())

    def sched_mask(self, cls: str) -> int:
        table = self.require("sched_group_mask")
        if cls not in table:
            raise UnknownFact(f"{self.key}: no sched_group_barrier mask for class {cls!r}")
        return table[cls]


def pack_dirs() -> List[Path]:
    """Directories named by ``FLYDSL_SCHED_TARGET_PATH``, in order."""
    return [Path(p) for p in os.environ.get(TARGET_PATH_ENV, "").split(os.pathsep) if p]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _load(base: str, sku: Optional[str], dirs: Tuple[str, ...]) -> TargetDesc:
    layers = []
    public = _TARGET_DIR / f"{base}.json"
    if public.exists():
        layers.append((f"builtin:{public.name}", _read(public)))
    for d in map(Path, dirs):
        p = d / f"{base}.json"
        if p.exists():
            layers.append((str(p), _read(p)))
    if not layers:
        known = sorted({p.stem for d in [_TARGET_DIR, *map(Path, dirs)] for p in d.glob("*.json") if "@" not in p.stem})
        raise UnknownFact(f"no target description for {base!r} (known: {known}); refusing to substitute another arch")
    if layers[0][1].get("arch") != base:
        raise ValueError(f"{layers[0][0]}: describes {layers[0][1].get('arch')!r}, not {base!r}")
    # sku packs: the requested sku, else the sku the arch-level layers declare
    want = sku or next((d.get("sku") for _, d in reversed(layers) if d.get("sku")), None)
    if want:
        for d in map(Path, dirs):
            p = d / f"{base}@{want}.json"
            if p.exists():
                layers.append((str(p), _read(p)))
    return TargetDesc.from_layers(layers, sku=sku)


def load_target(arch: str) -> TargetDesc:
    """Load the description for ``arch`` (``"gfx950"``, ``"gfx950@mi355x"``); exact arch match only.

    The public description is overlaid with the packs found in ``FLYDSL_SCHED_TARGET_PATH``;
    the result is cached per (arch, sku, search path).
    """
    spec = arch.split(":", 1)[0].lower()
    base, _, sku = spec.partition("@")
    return _load(base, sku or None, tuple(str(d) for d in pack_dirs()))


load_target.cache_clear = _load.cache_clear
