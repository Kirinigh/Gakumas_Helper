"""Mappings supplied by a caller for an audited UIObs projection."""
from __future__ import annotations

from typing import Mapping
from dataclasses import field, dataclass


@dataclass(frozen=True)
class ShadowMappings:
    card_forms: Mapping[str, tuple[str, int, str]]
    card_upgrades: Mapping[str, str]
    custom_forms: Mapping[str, tuple[tuple[str, int], ...]]
    drink_ids: Mapping[str, str]
    schedules: Mapping[tuple[str, str, int, bool], tuple]
    grow_effects: Mapping[str, tuple[tuple[int, str], ...]]
    item_ids: frozenset[str] = frozenset()
    observed_only_items: Mapping[str, str] = field(default_factory=dict)
    support_deltas: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
