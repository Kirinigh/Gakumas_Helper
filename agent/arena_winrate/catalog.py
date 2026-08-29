"""Resolve clicked skill-card details against the version-matched engine data."""

from __future__ import annotations

import re
import json
import itertools
from typing import Any
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Mapping, Sequence


class ArenaCatalogError(ValueError):
    """Raised when clicked detail text cannot resolve one exact engine ID."""


@dataclass(frozen=True)
class CustomizationObservation:
    name: str
    count: int
    detail_text: str = ""


@dataclass(frozen=True)
class EffectiveCustomizationResolution:
    """One authoritative detail resolution and how the badge count was used."""

    customizations: Mapping[str, int]
    resolved_count: int
    source: str
    observed_badge_count: int
    badge_count_match: bool
    evidence_mode: str = "negative_dependent"


@dataclass(frozen=True)
class GenericCostAmbiguityResolution:
    """The only two exact states left after detail, differing by generic cost."""

    generic_customization_id: int
    unenhanced_customizations: Mapping[str, int]
    enhanced_customizations: Mapping[str, int]


@dataclass(frozen=True)
class CustomizationEvidenceInventoryEntry:
    """One legal catalog state and every evidence carrier it may require.

    The inventory is generated from the fixed engine catalog.  It deliberately
    classifies unsupported detail DSL as ``unsupported_fail_closed`` instead
    of silently treating a missing matcher as negative evidence.
    """

    card_id: int
    expected_count: int
    group: tuple[tuple[str, int], ...]
    carriers: tuple[str, ...]
    unsupported_customization_ids: tuple[int, ...] = ()


_TEXT_NORMALIZATION_REPLACEMENTS = {
    "值": "値",
    "增": "増",
    "查": "査",
    "絕": "絶",
    "％": "%",
    "⁺": "+",
    "＋": "+",
    "﹢": "+",
    "（": "(",
    "）": ")",
    "ァ": "ア",
    "ベ": "べ",
    "！": "!",
    "？": "?",
}
_anchor_variant_sources: defaultdict[str, list[str]] = defaultdict(list)
for _source_character, _normalized_character in _TEXT_NORMALIZATION_REPLACEMENTS.items():
    _anchor_variant_sources[_normalized_character].append(_source_character)
_TEXT_ANCHOR_CHARACTER_VARIANTS = {
    normalized: tuple(dict.fromkeys((normalized, *sources)))
    for normalized, sources in _anchor_variant_sources.items()
}


def _normalise_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    compact = "".join(value.split())
    return compact.translate(str.maketrans(_TEXT_NORMALIZATION_REPLACEMENTS))


def _contains_exact_integer_token(
    compact: str,
    prefix: str,
    value: int,
    suffix: str = "",
) -> bool:
    """Match one rendered integer without accepting it as a longer number."""

    return (
        re.search(
            rf"{re.escape(prefix)}{re.escape(str(value))}"
            rf"(?!\d|[.,．，。]\d){re.escape(suffix)}",
            compact,
        )
        is not None
    )


def _exact_title_anchor_branch(value: str) -> str:
    return "".join(
        (
            re.escape(character)
            if len(variants := _TEXT_ANCHOR_CHARACTER_VARIANTS.get(character, (character,)))
            == 1
            else rf"(?:{'|'.join(re.escape(variant) for variant in variants)})"
        )
        for character in value
    )


def _selected_level_patch(value: object, level: int) -> str:
    """Return one level body, or the whole single-level patch."""

    text = str(value or "")
    if level < 1:
        return ""
    markers = tuple(re.finditer(r"level:([1-9]\d*)\s*\{", text))
    if not markers:
        return text if level == 1 else ""
    marker = next((item for item in markers if int(item.group(1)) == level), None)
    if marker is None:
        return ""
    start = marker.end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return ""


def _first_integer_increment(value: object, field: str) -> int | None:
    match = re.search(rf"(?<![.A-Za-z]){re.escape(field)}\+=(-?[0-9]+)", str(value or ""))
    return int(match.group(1)) if match else None


def _all_integer_increments(value: object, field: str) -> tuple[int, ...]:
    return tuple(
        int(match.group(1))
        for match in re.finditer(
            rf"(?<![.A-Za-z]){re.escape(field)}\+=(-?[0-9]+)",
            str(value or ""),
        )
    )


def _growth_increment(value: object, field: str, level: int) -> int | None:
    selected = _selected_level_patch(value, level)
    match = re.search(rf"g\.{re.escape(field)}\+=(-?[0-9]+)", selected)
    return int(match.group(1)) if match else None


class ArenaEntityCatalog:
    """A strict lookup view over bundled ``gakumas-data`` JSON files."""

    DATA_DIRECTORY = Path("node_modules") / "gakumas-data" / "json"
    _PURE_GENERIC_COST_EFFECT = re.compile(
        r"at:prestage\{target:this\{g\.cost\+=([1-9][0-9]*);?\}\}"
    )
    _GENERIC_COST_EMPTY_FIELDS = (
        "forceInitialHand",
        "conditions",
        "cost",
        "actions",
        "limit",
    )

    def __init__(
        self,
        skill_cards: Sequence[Mapping[str, Any]],
        customizations: Sequence[Mapping[str, Any]],
        p_items: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._cards = tuple(dict(row) for row in skill_cards)
        self._p_items = tuple(dict(row) for row in p_items)
        self._customizations = {int(row["id"]): dict(row) for row in customizations}
        if len(self._customizations) != len(customizations):
            raise ArenaCatalogError("bundled customization IDs must be unique positive integers")
        self._cards_by_id = {
            int(row["id"]): row
            for row in self._cards
            if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool)
        }
        if len(self._cards_by_id) != len(self._cards) or any(card_id < 1 for card_id in self._cards_by_id):
            raise ArenaCatalogError("bundled skill-card IDs must be unique positive integers")
        self._skill_card_title_aliases = self._build_skill_card_title_aliases()
        self._p_items_by_id = {
            int(row["id"]): row
            for row in self._p_items
            if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool)
        }
        if len(self._p_items_by_id) != len(self._p_items) or any(
            p_item_id < 1 for p_item_id in self._p_items_by_id
        ):
            raise ArenaCatalogError("bundled P-item IDs must be unique positive integers")
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for card in self._cards:
            by_name[_normalise_text(card.get("name"))].append(card)
        self._cards_by_name = {name: tuple(rows) for name, rows in by_name.items()}
        self._validate_customization_index()

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path) -> "ArenaEntityCatalog":
        root = Path(bundle_dir) / cls.DATA_DIRECTORY
        try:
            cards = json.loads((root / "skill_cards.json").read_text(encoding="utf-8"))
            customizations = json.loads((root / "customizations.json").read_text(encoding="utf-8"))
            p_items = json.loads((root / "p_items.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArenaCatalogError(f"bundled entity catalog is unavailable or invalid: {root}") from error
        if not all(isinstance(rows, list) for rows in (cards, customizations, p_items)):
            raise ArenaCatalogError("bundled entity catalogs must be JSON arrays")
        return cls(cards, customizations, p_items)

    def confirm_clicked_p_item_candidates(
        self,
        detail_text: str,
        *,
        candidate_p_item_ids: Sequence[int],
    ) -> int:
        """Resolve one clicked P-item title within the rendered-icon candidates."""

        candidate_ids = tuple(dict.fromkeys(candidate_p_item_ids))
        matches = self.clicked_p_item_candidate_matches(
            detail_text,
            candidate_p_item_ids=candidate_ids,
        )
        if len(matches) != 1:
            raise ArenaCatalogError(
                "clicked P-item detail must resolve exactly once within "
                f"icon candidates {candidate_ids!r}"
            )
        return matches[0]

    def clicked_p_item_candidate_matches(
        self,
        detail_text: str,
        *,
        candidate_p_item_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Return candidate IDs whose fixed title is visible in clicked detail text."""

        compact = _normalise_text(detail_text)
        candidate_ids = tuple(dict.fromkeys(candidate_p_item_ids))
        if not candidate_ids or any(
            isinstance(p_item_id, bool) or not isinstance(p_item_id, int) or p_item_id < 1
            for p_item_id in candidate_ids
        ):
            raise ArenaCatalogError("icon prediction contains no valid P-item candidate IDs")
        candidates = [
            item
            for p_item_id in candidate_ids
            if (item := self._p_items_by_id.get(p_item_id)) is not None
            and _normalise_text(item.get("name"))
            and _normalise_text(item.get("name")) in compact
        ]
        if candidates:
            longest = max(len(_normalise_text(item.get("name"))) for item in candidates)
            candidates = [
                item
                for item in candidates
                if len(_normalise_text(item.get("name"))) == longest
            ]
        if not candidates:
            # P-item detail titles are short, isolated OCR rows. Recover one
            # substitution only when the complete normalized row is equal in
            # length, preserves the upgraded ``+`` suffix, is not another
            # exact catalog title, and has exactly one match across the entire
            # fixed P-item catalog. The visual candidate family remains a
            # mandatory second boundary; insertions/deletions stay rejected.
            exact_titles = {
                _normalise_text(item.get("name"))
                for item in self._p_items
                if _normalise_text(item.get("name"))
            }
            candidate_id_set = set(candidate_ids)
            recovered: list[dict[str, Any]] = []
            lines = {
                normalized
                for line in str(detail_text or "").splitlines()
                if (normalized := _normalise_text(line))
            }
            for line in lines:
                core = line[:-1] if line.endswith("+") else line
                if len(core) < 4 or line in exact_titles:
                    continue
                global_matches = [
                    item
                    for item in self._p_items
                    if (title := _normalise_text(item.get("name")))
                    and len(title) == len(line)
                    and title.endswith("+") == line.endswith("+")
                    and sum(left != right for left, right in zip(title, line, strict=True))
                    == 1
                ]
                if (
                    len(global_matches) == 1
                    and int(global_matches[0]["id"]) in candidate_id_set
                ):
                    recovered.append(global_matches[0])
            candidates = list(
                {
                    int(item["id"]): item
                    for item in recovered
                }.values()
            )
        return tuple(int(item["id"]) for item in candidates)

    def resolve_skill_card(self, name: str, *, upgraded: bool) -> int:
        candidates = self._cards_by_name.get(_normalise_text(name), ())
        candidates = tuple(card for card in candidates if card.get("upgraded") is upgraded)
        if len(candidates) != 1:
            raise ArenaCatalogError(
                f"clicked skill-card detail must resolve exactly once: name={name!r}, upgraded={upgraded}"
            )
        card_id = candidates[0].get("id")
        if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id < 1:
            raise ArenaCatalogError("resolved skill-card ID is invalid")
        return card_id

    def skill_card_title(self, card_id: int) -> str:
        """Return the fixed detail title for one catalogued skill-card ID."""

        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        title = str(card.get("name", "")).strip()
        if not title:
            raise ArenaCatalogError(f"skill-card {card_id} has no fixed title")
        return title

    def skill_card_title_anchor_pattern(self, card_id: int) -> str:
        """Return one exact OCR anchor pattern with shared upgrade-mark aliases."""

        aliases = self.skill_card_title_aliases(card_id)
        branches = []
        for alias in aliases:
            if alias.endswith("+"):
                branches.append(
                    rf"{_exact_title_anchor_branch(alias[:-1])}(?:\+|⁺|＋|﹢)?"
                )
            else:
                branches.append(_exact_title_anchor_branch(alias))
        return rf"^(?:{'|'.join(branches)})$"

    def skill_card_title_aliases(self, card_id: int) -> tuple[str, ...]:
        """Return the exact title and any globally unique one-glyph-loss alias."""

        if card_id not in self._cards_by_id:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        return self._skill_card_title_aliases[card_id]

    def _build_skill_card_title_aliases(self) -> dict[int, tuple[str, ...]]:
        full_title_owners: defaultdict[str, set[int]] = defaultdict(set)
        dropped_title_owners: defaultdict[str, set[int]] = defaultdict(set)
        normalized_titles: dict[int, str] = {}
        for card_id, card in self._cards_by_id.items():
            title = _normalise_text(card.get("name"))
            if not title:
                raise ArenaCatalogError(f"skill-card {card_id} has no fixed title")
            normalized_titles[card_id] = title
            full_title_owners[title].add(card_id)
            dropped = title[1:]
            dropped_core = dropped[:-1] if dropped.endswith("+") else dropped
            if len(dropped_core) >= 4:
                dropped_title_owners[dropped].add(card_id)
        aliases: dict[int, tuple[str, ...]] = {}
        for card_id, title in normalized_titles.items():
            values = [title]
            dropped = title[1:]
            if (
                dropped_title_owners.get(dropped) == {card_id}
                and dropped not in full_title_owners
            ):
                values.append(dropped)
            aliases[card_id] = tuple(values)
        return aliases

    def _skill_card_title_match_length(
        self,
        card: Mapping[str, Any],
        detail_text: str,
        *,
        allow_missing_lead_alias: bool,
    ) -> int:
        card_id = int(card["id"])
        aliases = self._skill_card_title_aliases[card_id]
        compact = _normalise_text(detail_text)
        matches = [len(aliases[0])] if aliases[0] in compact else []
        if allow_missing_lead_alias:
            # A one-glyph-loss alias is deliberately narrower than the exact
            # title path: it must occupy one complete OCR line.  This prevents
            # effect text or an unrelated longer title from satisfying the
            # recovery merely because it contains the same substring.
            lines = {
                normalized
                for line in str(detail_text or "").splitlines()
                if (normalized := _normalise_text(line))
            }
            matches.extend(
                len(alias)
                for alias in aliases[1:]
                if alias in lines
            )
        return max(matches, default=0)

    def _normal_title_has_upgraded_mark(
        self,
        card: Mapping[str, Any],
        compact: str,
    ) -> bool:
        card_id = int(card["id"])
        return any(
            not alias.endswith("+") and f"{alias}+" in compact
            for alias in self._skill_card_title_aliases[card_id]
        )

    def skill_card_candidates_for_plan(
        self,
        candidate_card_ids: Sequence[int],
        *,
        plan: str,
    ) -> tuple[int, ...]:
        """Apply the stage's authoritative plan constraint to visual candidates."""

        if plan not in {"sense", "logic", "anomaly"}:
            raise ArenaCatalogError(f"unsupported contest plan: {plan!r}")
        candidate_ids = tuple(dict.fromkeys(candidate_card_ids))
        if not candidate_ids or any(
            isinstance(card_id, bool) or not isinstance(card_id, int) or card_id < 1
            for card_id in candidate_ids
        ):
            raise ArenaCatalogError(
                "visual identity contains no valid skill-card candidate IDs"
            )
        matches = []
        for card_id in candidate_ids:
            card = self._cards_by_id.get(card_id)
            if card is None:
                raise ArenaCatalogError(
                    f"visual identity references unknown skill card {card_id}"
                )
            card_plan = str(card.get("plan", ""))
            if card_plan in {plan, "free"}:
                matches.append(card_id)
        return tuple(matches)

    def confirm_clicked_skill_card(self, detail_text: str, *, predicted_card_id: int) -> int:
        """Require the clicked detail name and icon prediction to name the same ID."""

        return self.confirm_clicked_skill_card_candidates(
            detail_text,
            candidate_card_ids=(predicted_card_id,),
        )

    def confirm_clicked_skill_card_candidates(
        self,
        detail_text: str,
        *,
        candidate_card_ids: Sequence[int],
    ) -> int:
        """Resolve one clicked title within the recognizer's visual-family IDs.

        Customized arena overlays obscure the card's normal/``+`` marker, so
        The arena custom-card recognizer deliberately returns a visual family rather than an exact
        business ID.  The clicked title is the only place where that pair may
        be resolved.  IDs outside the retrieved family are never accepted.
        """

        compact = _normalise_text(detail_text)
        candidate_ids = tuple(dict.fromkeys(candidate_card_ids))
        if not candidate_ids or any(
            isinstance(card_id, bool) or not isinstance(card_id, int) or card_id < 1
            for card_id in candidate_ids
        ):
            raise ArenaCatalogError("icon prediction contains no valid skill-card candidate IDs")
        candidates = [
            card
            for card in self._cards
            if card.get("id") in candidate_ids
            and self._skill_card_title_match_length(
                card,
                detail_text,
                allow_missing_lead_alias=True,
            )
            > 0
        ]
        if not candidates:
            raise ArenaCatalogError(
                f"clicked skill-card text disagrees with icon candidates {candidate_ids!r}"
            )
        longest = max(
            self._skill_card_title_match_length(
                card,
                detail_text,
                allow_missing_lead_alias=True,
            )
            for card in candidates
        )
        candidates = [
            card
            for card in candidates
            if self._skill_card_title_match_length(
                card,
                detail_text,
                allow_missing_lead_alias=True,
            )
            == longest
        ]
        if len(candidates) == 1:
            if (
                candidates[0].get("upgraded") is False
                and self._normal_title_has_upgraded_mark(candidates[0], compact)
            ):
                raise ArenaCatalogError(
                    f"clicked skill-card text disagrees with icon candidates {candidate_ids!r}"
                )
        if len(candidates) > 1:
            names = {_normalise_text(card.get("name")) for card in candidates}
            upgraded_states = {card.get("upgraded") for card in candidates}
            if len(names) == 1 and upgraded_states == {False, True}:
                name = next(iter(names))
                title_is_upgraded = f"{name}+" in compact
                candidates = [card for card in candidates if card.get("upgraded") is title_is_upgraded]
        if len(candidates) != 1:
            raise ArenaCatalogError(
                f"clicked skill-card detail must resolve exactly once within icon candidates {candidate_ids!r}"
            )
        return int(candidates[0]["id"])

    def confirm_clicked_customizable_skill_card(
        self,
        detail_text: str,
        *,
        expected_count: int,
    ) -> int:
        """Resolve one clicked customizable card from its authoritative title.

        This is a fail-closed fallback for cases where the icon recognizer's
        Top-K family missed the card.  It is only valid after a positive badge
        count has already authorized the detail click, and only considers
        bundled cards that admit at least one legal group with that count.
        """

        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 1
        ):
            raise ArenaCatalogError(
                "clicked customizable-card fallback requires a positive badge count"
            )
        compact = _normalise_text(detail_text)
        candidates = [
            card
            for card in self._cards
            if self._skill_card_title_match_length(
                card,
                detail_text,
                allow_missing_lead_alias=False,
            )
            > 0
            and self.legal_customization_groups(
                int(card["id"]),
                expected_count=expected_count,
            )
        ]
        if candidates:
            longest = max(
                self._skill_card_title_match_length(
                    card,
                    detail_text,
                    allow_missing_lead_alias=False,
                )
                for card in candidates
            )
            candidates = [
                card
                for card in candidates
                if self._skill_card_title_match_length(
                    card,
                    detail_text,
                    allow_missing_lead_alias=False,
                )
                == longest
            ]
        if len(candidates) != 1:
            raise ArenaCatalogError(
                "clicked customizable skill-card title must resolve exactly once "
                f"for badge count {expected_count}"
            )
        return int(candidates[0]["id"])

    def confirm_clicked_customizable_skill_card_without_badge_count(
        self,
        detail_text: str,
    ) -> int:
        """Resolve a unique customizable title without trusting its face digit.

        The caller must already have authorized a bounded detail click from
        independent badge-presence evidence. This fallback only repairs an
        icon-family miss; it never decides whether a card should be clicked.
        """

        compact = _normalise_text(detail_text)
        candidates = [
            card
            for card in self._cards
            if self._skill_card_title_match_length(
                card,
                detail_text,
                allow_missing_lead_alias=False,
            )
            > 0
            and self.available_customization_ids(int(card["id"]))
        ]
        if candidates:
            longest = max(
                self._skill_card_title_match_length(
                    card,
                    detail_text,
                    allow_missing_lead_alias=False,
                )
                for card in candidates
            )
            candidates = [
                card
                for card in candidates
                if self._skill_card_title_match_length(
                    card,
                    detail_text,
                    allow_missing_lead_alias=False,
                )
                == longest
            ]
        if len(candidates) > 1:
            names = {_normalise_text(card.get("name")) for card in candidates}
            upgraded_states = {card.get("upgraded") for card in candidates}
            if len(names) == 1 and upgraded_states == {False, True}:
                name = next(iter(names))
                title_is_upgraded = f"{name}+" in compact
                candidates = [
                    card
                    for card in candidates
                    if card.get("upgraded") is title_is_upgraded
                ]
        if len(candidates) != 1:
            raise ArenaCatalogError(
                "clicked customizable skill-card title must resolve exactly once "
                "without a badge-count constraint"
            )
        return int(candidates[0]["id"])

    def available_customization_ids(self, card_id: int) -> tuple[int, ...]:
        """Return the version-matched engine customization IDs for one card."""

        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(f"skill-card ID is absent from the bundled catalog: {card_id}")
        return tuple(
            int(value)
            for value in str(card.get("availableCustomizations", "")).split(",")
            if value
        )

    def legal_customization_groups(
        self,
        card_id: int,
        *,
        expected_count: int,
    ) -> tuple[dict[str, int], ...]:
        """Enumerate engine-ready groups whose selected levels sum to the badge count."""

        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
            raise ArenaCatalogError("expected customization count must be a non-negative integer")
        available = self.available_customization_ids(card_id)
        if not available:
            return ({},) if expected_count == 0 else ()
        levels = [range(int(self._customizations[customization_id]["max"]) + 1) for customization_id in available]
        return tuple(
            {
                str(customization_id): count
                for customization_id, count in zip(available, counts, strict=True)
                if count
            }
            for counts in itertools.product(*levels)
            if sum(counts) == expected_count
        )

    def customization_resolution_requires_confirmation(
        self,
        card_id: int,
        *,
        expected_count: int,
    ) -> bool:
        """Return whether one badge count still admits multiple engine states.

        A single OCR read is sufficient when the static catalog proves that the
        observed count maps to exactly one customization group.  When two or
        more groups share that count, the final-effect OCR is the only
        discriminator and must be confirmed from a fresh frame.
        """

        return len(
            self.legal_customization_groups(
                card_id,
                expected_count=expected_count,
            )
        ) > 1

    def customization_evidence_inventory(
        self,
    ) -> tuple[CustomizationEvidenceInventoryEntry, ...]:
        """Enumerate every legal zero and positive customization state.

        This is the G1 coverage surface for the fixed ``gakumas-data``
        revision.  A state is never left unclassified: unknown detail DSL is
        explicitly assigned the fail-closed carrier, while generic cost and
        structural omissions are separated from ordinary rendered effects.
        """

        entries: list[CustomizationEvidenceInventoryEntry] = []
        for card_id, card in sorted(self._cards_by_id.items()):
            available = self.available_customization_ids(card_id)
            if not available:
                continue
            maximum_total = sum(
                int(self._customizations[customization_id]["max"])
                for customization_id in available
            )
            unsupported = tuple(
                customization_id
                for customization_id in available
                if self._effective_detail_matcher(card, customization_id) is None
            )
            for expected_count in range(maximum_total + 1):
                legal_groups = self.legal_customization_groups(
                    card_id,
                    expected_count=expected_count,
                )
                for group in legal_groups:
                    carriers: set[str] = set()
                    if len(legal_groups) == 1:
                        carriers.add("catalog_unique")
                    if unsupported:
                        carriers.add("unsupported_fail_closed")
                    for customization_id in available:
                        carrier = self._customization_evidence_carrier(
                            card,
                            customization_id,
                            group.get(str(customization_id), 0),
                            group,
                        )
                        carriers.add(carrier)
                    entries.append(
                        CustomizationEvidenceInventoryEntry(
                            card_id=card_id,
                            expected_count=expected_count,
                            group=tuple(sorted(group.items())),
                            carriers=tuple(sorted(carriers)),
                            unsupported_customization_ids=unsupported,
                        )
                    )
        return tuple(entries)

    def _customization_evidence_carrier(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        level: int,
        group: Mapping[str, int],
    ) -> str:
        definition = self._customizations[customization_id]
        if self._effective_detail_matcher(card, customization_id) is None:
            return "unsupported_fail_closed"
        if self._pure_generic_cost_delta(customization_id) is not None:
            return "card_face_generic_cost"
        if _normalise_text(definition.get("actions")) == "upgradeHand":
            return "badge_constraint_only"
        if level == 0:
            return (
                "visible_baseline"
                if self._customization_has_visible_baseline(
                    card,
                    customization_id,
                )
                else "no_direct_zero_evidence"
            )
        if level > 0:
            descriptor = self._typed_cost_descriptor(card)
            if descriptor is not None and customization_id in descriptor[2]:
                if (
                    self._effective_typed_cost(
                        group,
                        base_value=descriptor[1],
                        deltas=descriptor[2],
                    )
                    == 0
                ):
                    return "structural_omission"
            if definition.get("limit") == 0 and card.get("limit") == 1:
                return "structural_omission"
        return "detail_positive"

    def _customization_has_visible_baseline(
        self,
        card: Mapping[str, Any],
        customization_id: int,
    ) -> bool:
        definition = self._customizations[customization_id]
        descriptor = self._typed_cost_descriptor(card)
        if descriptor is not None and customization_id in descriptor[2]:
            return True
        scalar_fields = (
            "goodConditionTurns",
            "goodImpressionTurns",
            "perfectConditionTurns",
            "concentration",
            "genki",
            "motivation",
            "fullPowerCharge",
            "halfCostTurns",
            "score",
        )
        maximum = int(definition["max"])
        for field in scalar_fields:
            if any(
                _growth_increment(definition.get("effects"), field, level)
                is not None
                for level in range(1, maximum + 1)
            ):
                return _first_integer_increment(card.get("actions"), field) is not None
        return definition.get("limit") == 0 and card.get("limit") == 1

    def generic_cost_hypotheses(
        self,
        card_id: int,
    ) -> tuple[int, int] | None:
        """Return the only two card-face costs allowed by the fixed DSL."""

        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        generic = tuple(
            (customization_id, delta)
            for customization_id in self.available_customization_ids(card_id)
            if (delta := self._pure_generic_cost_delta(customization_id)) is not None
        )
        if not generic:
            return None
        if len(generic) != 1:
            raise ArenaCatalogError(
                f"card {card_id} has multiple generic-cost customizations "
                f"{tuple(customization_id for customization_id, _delta in generic)!r}"
            )
        _, delta = generic[0]
        base_match = re.fullmatch(
            r"(?:cost|stamina)-=([0-9]+)",
            str(card.get("cost", "")),
        )
        if base_match is None:
            raise ArenaCatalogError(
                f"card {card_id} generic-cost DSL has no visible card-face hypothesis"
            )
        base_value = int(base_match.group(1))
        customized_value = max(0, base_value - delta)
        if customized_value == base_value:
            raise ArenaCatalogError(
                f"card {card_id} generic-cost hypotheses are not distinguishable"
            )
        return base_value, customized_value

    def validate_generic_cost_fallback_pair(
        self,
        card_id: int,
        *,
        generic_customization_id: int,
        unenhanced: Mapping[str, int],
        enhanced: Mapping[str, int],
        cost_hypotheses: Sequence[int],
        observed_badge_count: int | None,
    ) -> None:
        """Revalidate a serialized generic-cost bound against this catalog."""

        mappings_valid = all(
            isinstance(group, Mapping)
            and all(
                isinstance(key, str)
                and key.isdigit()
                and str(int(key)) == key
                and not isinstance(level, bool)
                and isinstance(level, int)
                and level > 0
                for key, level in group.items()
            )
            for group in (unenhanced, enhanced)
        )
        hypotheses_valid = (
            isinstance(cost_hypotheses, Sequence)
            and not isinstance(cost_hypotheses, (str, bytes))
            and len(cost_hypotheses) == 2
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value >= 0
                for value in cost_hypotheses
            )
        )
        observed_valid = observed_badge_count is None or (
            not isinstance(observed_badge_count, bool)
            and isinstance(observed_badge_count, int)
            and observed_badge_count > 0
        )
        if (
            isinstance(card_id, bool)
            or not isinstance(card_id, int)
            or card_id < 1
            or isinstance(generic_customization_id, bool)
            or not isinstance(generic_customization_id, int)
            or generic_customization_id < 1
            or not mappings_valid
            or not hypotheses_valid
            or not observed_valid
        ):
            raise ArenaCatalogError("generic-cost fallback pair has invalid types")

        available = self.available_customization_ids(card_id)
        generic_ids = tuple(
            customization_id
            for customization_id in available
            if self._pure_generic_cost_delta(customization_id) is not None
        )
        if generic_ids != (generic_customization_id,):
            raise ArenaCatalogError(
                f"card {card_id} does not bind generic-cost customization {generic_customization_id}"
            )
        expected_hypotheses = self.generic_cost_hypotheses(card_id)
        if expected_hypotheses is None or tuple(cost_hypotheses) != expected_hypotheses:
            raise ArenaCatalogError(
                f"card {card_id} generic-cost hypotheses do not match the catalog"
            )
        generic_key = str(generic_customization_id)
        expected_enhanced = dict(unenhanced)
        expected_enhanced[generic_key] = 1
        if generic_key in unenhanced or dict(enhanced) != expected_enhanced:
            raise ArenaCatalogError(
                f"card {card_id} fallback pair is not the catalog generic-cost bit"
            )
        for label, group in (("unenhanced", unenhanced), ("enhanced", enhanced)):
            legal = self.legal_customization_groups(
                card_id,
                expected_count=sum(group.values()),
            )
            if dict(group) not in legal:
                raise ArenaCatalogError(
                    f"card {card_id} {label} fallback state is not catalog-legal"
                )
        if observed_badge_count is not None and observed_badge_count not in {
            sum(unenhanced.values()),
            sum(enhanced.values()),
        }:
            raise ArenaCatalogError(
                f"card {card_id} observed badge count is outside the fallback pair"
            )

    def card_face_cost_descriptor(self, card_id: int) -> tuple[str, int] | None:
        """Return the rendered generic/stamina cost kind and base value."""

        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        match = re.fullmatch(
            r"(cost|stamina)-=([0-9]+)",
            str(card.get("cost", "")),
        )
        if match is None:
            return None
        return match.group(1), int(match.group(2))

    def generic_cost_card_ids(self) -> tuple[int, ...]:
        """Return every card whose fixed DSL changes the card-face cost."""

        return tuple(
            card_id
            for card_id in sorted(self._cards_by_id)
            if self.generic_cost_hypotheses(card_id) is not None
        )

    def certify_card_face_generic_cost(
        self,
        card_id: int,
        *,
        resolved: Mapping[str, int],
        frame_values: Sequence[int],
    ) -> str:
        """Require three stable card-face reads to agree with one legal group."""

        hypotheses = self.generic_cost_hypotheses(card_id)
        if hypotheses is None:
            raise ArenaCatalogError(
                f"card {card_id} has no generic-cost customization to certify"
            )
        if len(frame_values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in frame_values
        ):
            raise ArenaCatalogError(
                "generic-cost evidence requires three non-negative integer reads"
            )
        if len(set(frame_values)) != 1:
            raise ArenaCatalogError(
                f"generic-cost evidence disagreed across stable frames: {tuple(frame_values)!r}"
            )
        generic_ids = tuple(
            customization_id
            for customization_id in self.available_customization_ids(card_id)
            if self._pure_generic_cost_delta(customization_id) is not None
        )
        selected_level = int(resolved.get(str(generic_ids[0]), 0))
        if selected_level not in (0, 1):
            raise ArenaCatalogError(
                f"card {card_id} generic-cost level is invalid: {selected_level}"
            )
        expected_value = hypotheses[selected_level]
        if frame_values[0] != expected_value:
            raise ArenaCatalogError(
                f"card {card_id} card-face cost {frame_values[0]} disagrees with "
                f"resolved generic-cost level {selected_level} (expected {expected_value})"
            )
        return "card_face_generic_cost_unique"

    def effective_customization_evidence_mode(
        self,
        card_id: int,
        detail_text: str,
        *,
        expected_count: int,
        resolved: Mapping[str, int],
    ) -> str:
        """Classify whether visible positive effects uniquely prove a group.

        ``catalog_unique`` needs no effect discrimination because the trusted
        badge total has one legal group. ``positive_unique`` is selected by
        one or more explicitly rendered customized values or explicitly
        rendered unchanged baseline values that exclude another legal group. A
        ``negative_dependent`` result still relies on a missing or unchanged
        row and therefore needs an independent effect-ROI read before runtime
        use.
        """

        legal_groups = self.legal_customization_groups(
            card_id,
            expected_count=expected_count,
        )
        normalized_resolved = {
            str(key): int(value)
            for key, value in resolved.items()
            if int(value) > 0
        }
        if normalized_resolved not in legal_groups:
            raise ArenaCatalogError(
                f"resolved customizations are not legal for card {card_id} "
                f"and badge count {expected_count}: {normalized_resolved!r}"
            )
        if len(legal_groups) == 1:
            return "catalog_unique"

        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        compact = _normalise_text(detail_text)
        typed_cost_candidates = self._visible_typed_cost_groups(
            card,
            compact,
            legal_groups,
        )
        if typed_cost_candidates == (normalized_resolved,):
            return "positive_unique"
        visible_levels: dict[int, frozenset[int]] = {}
        for customization_id in self.available_customization_ids(card_id):
            matcher = self._effective_detail_matcher(card, customization_id)
            if matcher is None:
                return "negative_dependent"
            maximum = int(self._customizations[customization_id]["max"])
            zero_matches = matcher(compact, 0)
            visible_levels[customization_id] = frozenset(
                (
                    (0,)
                    if zero_matches
                    and self._zero_signature_is_visible(
                        card,
                        customization_id,
                        compact,
                    )
                    else ()
                )
                + tuple(
                    level
                    for level in range(1, maximum + 1)
                    if not zero_matches
                    and matcher(compact, level)
                    and self._positive_signature_is_visible(
                        card,
                        customization_id,
                        compact,
                        level,
                    )
                )
            )
        if not any(visible_levels.values()):
            return "negative_dependent"
        candidates = tuple(
            group
            for group in legal_groups
            if all(
                not visible_levels[customization_id]
                or group.get(str(customization_id), 0)
                in visible_levels[customization_id]
                for customization_id in visible_levels
            )
        )
        return (
            "positive_unique"
            if candidates == (normalized_resolved,)
            else "negative_dependent"
        )

    def certify_structural_omission_evidence(
        self,
        card_id: int,
        *,
        full_detail_text: str,
        effect_roi_text: str,
        expected_count: int,
        resolved: Mapping[str, int],
        generic_cost_frame_values: Sequence[int] | None = None,
        effect_roi_title_bound: bool = False,
    ) -> str:
        """Certify selected UI-defined omissions from two covered OCR views.

        A typed cost reduced to zero and a removed once-per-stage footer are
        intentionally omitted by the contest detail UI. Absence alone is not
        evidence: both the full detail OCR and the independently enlarged
        effect ROI must contain the card title and every derivable invariant
        action anchor when structural omission evidence is required. The full
        detail pass binds the exact card title. When the caller used a fixed
        ROI, that ROI must repeat the title before absence inside the effect
        panel can be evidence. A ROI geometrically constructed from the unique
        exact title anchor may instead carry ``effect_roi_title_bound=True``;
        invariant action anchors are still mandatory. Every selected
        customization must then be either one of
        those certified omissions, have its own explicitly rendered positive
        signature, or be backed by an independently certified three-frame
        card-face generic-cost value. Other invisible effects (for example
        ``upgradeHand`` without a visible positive signature) remain
        fail-closed.
        """

        legal_groups = self.legal_customization_groups(
            card_id,
            expected_count=expected_count,
        )
        normalized_resolved = {
            str(key): int(value)
            for key, value in resolved.items()
            if int(value) > 0
        }
        if normalized_resolved not in legal_groups:
            raise ArenaCatalogError(
                f"resolved customizations are not legal for card {card_id} "
                f"and badge count {expected_count}: {normalized_resolved!r}"
            )
        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        full_compact = _normalise_text(full_detail_text)
        roi_compact = _normalise_text(effect_roi_text)
        title = _normalise_text(card.get("name"))
        anchors = self._detail_coverage_anchors(card, normalized_resolved)
        descriptor = self._typed_cost_descriptor(card)
        typed_label = descriptor[0] if descriptor is not None else None
        typed_deltas = descriptor[2] if descriptor is not None else {}
        effective_typed_cost = (
            self._effective_typed_cost(
                normalized_resolved,
                base_value=descriptor[1],
                deltas=typed_deltas,
            )
            if descriptor is not None
            else None
        )
        selected_generic_costs = tuple(
            int(customization_text)
            for customization_text, level in normalized_resolved.items()
            if int(level) > 0
            and self._pure_generic_cost_delta(int(customization_text)) is not None
        )
        selected_structural_omissions = tuple(
            customization_id
            for customization_text, level in normalized_resolved.items()
            if int(level) > 0
            for customization_id in (int(customization_text),)
            if (
                customization_id in typed_deltas
                and effective_typed_cost == 0
            )
            or (
                self._customizations[customization_id].get("limit") == 0
                and card.get("limit") == 1
            )
        )
        if not anchors and (
            selected_structural_omissions or not selected_generic_costs
        ):
            raise ArenaCatalogError(
                f"card {card_id} lacks an invariant action anchor for omission coverage"
            )
        evidence_sources = (
            ("full detail", full_compact),
            ("effect ROI", roi_compact),
        )
        if not title or title not in full_compact:
            raise ArenaCatalogError(
                f"card {card_id} full detail OCR lacks the exact title anchor"
            )
        roi_upgraded_title = (
            title.endswith("+")
            and title in full_compact
            and title[:-1] in roi_compact
        )
        if (
            selected_structural_omissions
            and title not in roi_compact
            and not roi_upgraded_title
            and not effect_roi_title_bound
        ):
            raise ArenaCatalogError(
                f"card {card_id} effect ROI OCR lacks the exact title anchor"
            )
        # The two OCR passes cover the same accepted detail image at different
        # scales. The enlarged effect ROI remains the authoritative coverage
        # channel. The full-screen pass may miss a row only when that complete
        # row is independently present inside the ROI. Never concatenate source
        # boundaries, which could fabricate an anchor from unrelated fragments.
        roi_missing = tuple(
            anchor
            for anchor in anchors
            if not self._detail_roi_coverage_anchor_present(
                card,
                normalized_resolved,
                anchor,
                full_compact=full_compact,
                roi_compact=roi_compact,
            )
        )
        counted_roi_coverage = any(
            self._detail_counted_score_anchor_present(
                card,
                normalized_resolved,
                anchor,
                roi_compact,
            )
            for anchor in anchors
        )
        if roi_missing and not counted_roi_coverage:
            raise ArenaCatalogError(
                f"card {card_id} effect ROI OCR lacks invariant anchors {roi_missing!r}"
            )
        uncovered_full_missing = tuple(
            anchor
            for anchor in anchors
            if not self._detail_coverage_anchor_present(
                card,
                normalized_resolved,
                anchor,
                full_compact,
            )
            and not self._detail_coverage_anchor_present(
                card,
                normalized_resolved,
                anchor,
                roi_compact,
            )
        )
        if uncovered_full_missing:
            raise ArenaCatalogError(
                f"card {card_id} covered detail OCR lacks invariant anchors "
                f"{uncovered_full_missing!r}"
            )
        certified_omissions = 0
        certified_generic_costs = 0
        for customization_text, level in normalized_resolved.items():
            customization_id = int(customization_text)
            definition = self._customizations[customization_id]
            generic_cost = self._pure_generic_cost_delta(customization_id) is not None
            if generic_cost and generic_cost_frame_values is not None:
                self.certify_card_face_generic_cost(
                    card_id,
                    resolved=normalized_resolved,
                    frame_values=generic_cost_frame_values,
                )
                certified_generic_costs += 1
                continue
            typed_cost_omitted = (
                customization_id in typed_deltas
                and effective_typed_cost == 0
                and typed_label is not None
                and typed_label not in full_compact
                and typed_label not in roi_compact
            )
            once_footer_omitted = (
                definition.get("limit") == 0
                and card.get("limit") == 1
                and "試験・ステージ中1回" not in full_compact
                and "試験・ステージ中1回" not in roi_compact
            )
            if typed_cost_omitted or once_footer_omitted:
                certified_omissions += 1
                continue
            matcher = self._effective_detail_matcher(card, customization_id)
            channel_has_positive_signature = matcher is not None and any(
                not matcher(compact, 0)
                and matcher(compact, int(level))
                and self._positive_signature_is_visible(
                    card,
                    customization_id,
                    compact,
                    int(level),
                )
                for _source_name, compact in evidence_sources
            )
            if not channel_has_positive_signature:
                raise ArenaCatalogError(
                    f"card {card_id} selected customization {customization_id} "
                    "has neither a certified structural omission nor an explicit "
                    "positive signature"
                )
        if certified_omissions < 1 and certified_generic_costs < 1:
            raise ArenaCatalogError(
                f"card {card_id} has no selected external evidence to certify"
            )
        if certified_omissions and certified_generic_costs:
            return "external_evidence_unique"
        if certified_generic_costs:
            return "card_face_generic_cost_unique"
        return "structural_omission_unique"

    def _visible_typed_cost_groups(
        self,
        card: Mapping[str, Any],
        compact: str,
        legal_groups: Sequence[Mapping[str, int]],
    ) -> tuple[Mapping[str, int], ...]:
        descriptor = self._typed_cost_descriptor(card)
        if descriptor is None:
            return ()
        label, base_value, deltas = descriptor
        match = re.search(rf"{re.escape(label)}([0-9]+)", compact)
        if match is None:
            return ()
        observed = int(match.group(1))
        return tuple(
            group
            for group in legal_groups
            if self._effective_typed_cost(
                group,
                base_value=base_value,
                deltas=deltas,
            )
            == observed
        )

    def _typed_cost_descriptor(
        self,
        card: Mapping[str, Any],
    ) -> tuple[str, int, Mapping[int, int]] | None:
        typed_cost = re.fullmatch(
            r"(concentration|cost|stamina|fullPowerCharge|goodConditionTurns|goodImpressionTurns|motivation|perfectConditionTurns)-=([0-9]+)",
            str(card.get("cost", "")),
        )
        if typed_cost is None:
            return None
        labels = {
            "concentration": "集中消費",
            "cost": "体力消費",
            "stamina": "体力消費",
            "fullPowerCharge": "全力値消費",
            "goodConditionTurns": "好調消費",
            "goodImpressionTurns": "好印象消費",
            "motivation": "やる気消費",
            "perfectConditionTurns": "絶好調消費",
        }
        deltas: dict[int, int] = {}
        for customization_id in (
            int(value)
            for value in str(card.get("availableCustomizations", "")).split(",")
            if value
        ):
            match = re.search(
                r"g\.typedCost\+=([0-9]+)",
                str(self._customizations[customization_id].get("effects", "")),
            )
            if match is not None:
                deltas[customization_id] = int(match.group(1))
        if not deltas:
            return None
        return labels[typed_cost.group(1)], int(typed_cost.group(2)), deltas

    @staticmethod
    def _effective_typed_cost(
        group: Mapping[str, int],
        *,
        base_value: int,
        deltas: Mapping[int, int],
    ) -> int:
        return max(
            0,
            base_value
            - sum(
                delta * int(group.get(str(customization_id), 0))
                for customization_id, delta in deltas.items()
            ),
        )

    def _detail_coverage_anchors(
        self,
        card: Mapping[str, Any],
        resolved: Mapping[str, int],
    ) -> tuple[str, ...]:
        actions = str(card.get("actions", ""))
        anchors: list[str] = []
        scalar_labels = {
            "goodConditionTurns": ("好調", "ターン"),
            "goodImpressionTurns": ("好印象+", ""),
            "perfectConditionTurns": ("絶好調", "ターン"),
            "concentration": ("集中+", ""),
            "genki": ("元気+", ""),
            "motivation": ("やる気+", ""),
            "fullPowerCharge": ("全力値+", ""),
            "score": ("スコア+", ""),
        }
        for field, (prefix, suffix) in scalar_labels.items():
            base_values = _all_integer_increments(actions, field)
            if not base_values:
                continue
            selected_delta = 0
            for customization_text, level in resolved.items():
                delta = _growth_increment(
                    self._customizations[int(customization_text)].get("effects"),
                    field,
                    int(level),
                )
                if delta is not None:
                    selected_delta += delta
            anchors.extend(
                f"{prefix}{value + selected_delta}{suffix}"
                for value in base_values
            )
        score_by_genki = re.search(
            r"score\+=genki\*([0-9]+(?:\.[0-9]+)?)",
            actions,
        )
        if score_by_genki is not None:
            percent = int(round(float(score_by_genki.group(1)) * 100))
            anchors.append(f"元気の{percent}%分スコア")
        if re.search(
            r"(?:^|[;{])cardUsesRemaining\+=1(?:[;}]|$)",
            _normalise_text(actions),
        ):
            # Narrow detail OCR may split or drop the trailing ``追加`` while
            # preserving the action row's stable semantic prefix and +1.
            # Other title/action/limit anchors are still required together.
            anchors.append("スキルカード使用数")
        if re.search(r"(?:^|;)turnsRemaining\+=1(?:;|$)", _normalise_text(actions)):
            anchors.append("ターン追加")
        if re.search(r"(?:^|;)drawCard(?:;|$)", _normalise_text(actions)):
            anchors.append("スキルカードを")
        stance = re.search(r"(?:^|;)setStance\((strength|preservation)\)", actions)
        if stance is not None:
            anchors.append("強気に変更" if stance.group(1) == "strength" else "温存に変更")
        enthusiasm = re.search(r"setEnthusiasmBonus\(([0-9]+)\)", actions)
        if enthusiasm is not None:
            anchors.append(f"熱意追加+{int(enthusiasm.group(1))}")
        if re.search(r"target:all\s*\{\s*g\.cost\+=([0-9]+)", actions):
            anchors.append("すべてのスキルカードの")
        return tuple(dict.fromkeys(anchors))

    def _detail_coverage_anchor_present(
        self,
        card: Mapping[str, Any],
        resolved: Mapping[str, int],
        anchor: str,
        compact: str,
    ) -> bool:
        """Accept an exact row or the UI's counted rendering of repeated rows."""

        if anchor in compact:
            return True
        if anchor == "スキルカード使用数":
            return re.search(r"スキルカ(?:ード)?[^+]{0,8}\+1", compact) is not None
        if anchor == "ターン追加":
            return "ターン追力" in compact
        return self._detail_counted_score_anchor_present(
            card,
            resolved,
            anchor,
            compact,
        )

    def _detail_roi_coverage_anchor_present(
        self,
        card: Mapping[str, Any],
        resolved: Mapping[str, int],
        anchor: str,
        *,
        full_compact: str,
        roi_compact: str,
    ) -> bool:
        """Accept one bounded ROI-only OCR intrusion backed by exact full text.

        The enlarged effect crop can include a neighbouring one-glyph counter
        between a scalar label and its signed value (for example
        ``集中1+5``).  This is coverage evidence only when the same accepted
        frame's full-detail OCR contains the exact scalar anchor.  The inserted
        glyph must be exactly one digit before ``+``; a changed value, a
        multi-digit insertion, or any non-scalar anchor remains rejected.
        """

        if self._detail_coverage_anchor_present(card, resolved, anchor, roi_compact):
            return True
        scalar = re.fullmatch(
            r"(好調|好印象|絶好調|集中|元気|やる気|全力値|スコア)"
            r"\+([0-9]+)(ターン)?",
            anchor,
        )
        if scalar is None:
            return False
        if re.search(rf"{re.escape(anchor)}(?![0-9])", full_compact) is None:
            return False
        prefix, value, suffix = scalar.groups()
        return (
            re.search(
                rf"{re.escape(prefix)}[0-9]\+{re.escape(value)}"
                rf"{re.escape(suffix or '')}(?![0-9])",
                roi_compact,
            )
            is not None
        )

    def _detail_counted_score_anchor_present(
        self,
        card: Mapping[str, Any],
        resolved: Mapping[str, int],
        anchor: str,
        compact: str,
    ) -> bool:
        """Recognize the UI's explicit ``N 回`` compaction of repeated scores."""

        score = re.fullmatch(r"スコア\+([0-9]+)", anchor)
        if score is None:
            return False
        actions = str(card.get("actions", ""))
        selected_delta = 0
        for customization_text, level in resolved.items():
            delta = _growth_increment(
                self._customizations[int(customization_text)].get("effects"),
                "score",
                int(level),
            )
            if delta is not None:
                selected_delta += delta
        effective_scores = tuple(
            value + selected_delta
            for value in _all_integer_increments(actions, "score")
        )
        expected_value = int(score.group(1))
        repetitions = effective_scores.count(expected_value)
        if repetitions < 2:
            return False
        return re.search(
            rf"(?:スコア)?\+{expected_value}[（(]?{repetitions}回",
            compact,
        ) is not None

    def _positive_signature_is_visible(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        compact: str,
        level: int,
    ) -> bool:
        """Reject matcher branches that become true only because text is absent."""

        definition = self._customizations[customization_id]
        typed_cost = re.fullmatch(
            r"(concentration|cost|stamina|fullPowerCharge|goodConditionTurns|goodImpressionTurns|motivation|perfectConditionTurns)-=([0-9]+)",
            str(card.get("cost", "")),
        )
        cost_delta = re.search(
            r"g\.typedCost\+=([0-9]+)",
            str(definition.get("effects", "")),
        )
        if typed_cost and cost_delta:
            labels = {
                "concentration": "集中消費",
                "cost": "体力消費",
                "stamina": "体力消費",
                "fullPowerCharge": "全力値消費",
                "goodConditionTurns": "好調消費",
                "goodImpressionTurns": "好印象消費",
                "motivation": "やる気消費",
                "perfectConditionTurns": "絶好調消費",
            }
            expected = max(
                0,
                int(typed_cost.group(2)) - int(cost_delta.group(1)) * level,
            )
            if expected == 0:
                return _contains_exact_integer_token(
                    compact,
                    labels[typed_cost.group(1)],
                    0,
                )
        if "g.scoreTimes+=1" in str(definition.get("effects", "")):
            base_scores = _all_integer_increments(
                card.get("actions"),
                "score",
            )
            return any(
                re.search(
                    rf"(?:スコア|スコア値増加)\+{score}"
                    rf"(?![0-9]|[.,．，。][0-9])\(2回\)",
                    compact,
                )
                is not None
                for score in base_scores
            )
        if definition.get("limit") == 0 and card.get("limit") == 1:
            return "試験・ステージ中1回" in compact
        return True

    def _zero_signature_is_visible(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        compact: str,
    ) -> bool:
        """Accept level zero only when the unchanged value is rendered.

        This is not an absence test.  It covers catalog alternatives where a
        baseline number is visibly present, such as ``やる気+5`` proving that
        a ``やる気+`` growth customization was not selected.  Invisible
        actions, missing rows and generic cost changes remain unproven.
        """

        definition = self._customizations[customization_id]
        typed_cost = re.fullmatch(
            r"(concentration|cost|stamina|fullPowerCharge|goodConditionTurns|goodImpressionTurns|motivation|perfectConditionTurns)-=([0-9]+)",
            str(card.get("cost", "")),
        )
        cost_delta = re.search(
            r"g\.typedCost\+=([0-9]+)",
            str(definition.get("effects", "")),
        )
        if typed_cost and cost_delta:
            labels = {
                "concentration": "集中消費",
                "cost": "体力消費",
                "stamina": "体力消費",
                "fullPowerCharge": "全力値消費",
                "goodConditionTurns": "好調消費",
                "goodImpressionTurns": "好印象消費",
                "motivation": "やる気消費",
                "perfectConditionTurns": "絶好調消費",
            }
            return _contains_exact_integer_token(
                compact,
                labels[typed_cost.group(1)],
                int(typed_cost.group(2)),
            )

        scalar_labels = {
            "goodConditionTurns": ("好調", "ターン"),
            "goodImpressionTurns": ("好印象+", ""),
            "perfectConditionTurns": ("絶好調", "ターン"),
            "concentration": ("集中+", ""),
            "genki": ("元気+", ""),
            "motivation": ("やる気+", ""),
            "fullPowerCharge": ("全力値+", ""),
            "halfCostTurns": ("消費体力減少", "ターン"),
            "score": ("スコア+", ""),
        }
        for field, (prefix, suffix) in scalar_labels.items():
            maximum = int(definition["max"])
            if not any(
                _growth_increment(definition.get("effects"), field, level)
                is not None
                for level in range(1, maximum + 1)
            ):
                continue
            base_value = _first_integer_increment(card.get("actions"), field)
            return base_value is not None and _contains_exact_integer_token(
                compact,
                prefix,
                base_value,
                suffix,
            )

        percentage_growths = (
            (
                "scoreByGenki",
                "genki",
                "元気の",
            ),
            (
                "scoreByMotivation",
                "motivation",
                "やる気の",
            ),
            (
                "scoreByGoodImpressionTurns",
                "goodImpressionTurns",
                "好印象の",
            ),
        )
        for growth_field, action_field, prefix in percentage_growths:
            if not any(
                re.search(
                    rf"g\.{re.escape(growth_field)}\+=",
                    _selected_level_patch(definition.get("effects"), level),
                )
                for level in range(1, int(definition["max"]) + 1)
            ):
                continue
            base = re.search(
                rf"score\+={re.escape(action_field)}\*([0-9]+(?:\.[0-9]+)?)",
                str(card.get("actions", "")),
            )
            if base is None:
                return False
            return _contains_exact_integer_token(
                compact,
                prefix,
                int(round(float(base.group(1)) * 100)),
                "%分スコア",
            )

        if "g.stanceLevel+=1" in str(definition.get("effects", "")):
            base_stance = re.search(
                r"setStance\((strength|preservation)\)",
                str(card.get("actions", "")),
            )
            if base_stance is None:
                return False
            label = "強気" if base_stance.group(1) == "strength" else "温存"
            return (
                f"{label}に変更" in compact and f"{label}2段階目に変更" not in compact
            )

        if definition.get("limit") == 0 and card.get("limit") == 1:
            return "試験・ステージ中1回" in compact
        return False

    def observe_customizations_from_clicked_text(
        self,
        card_id: int,
        detail_text: str,
    ) -> tuple[CustomizationObservation, ...]:
        """Extract only card-eligible customization rows from a clicked detail."""

        import re

        compact = _normalise_text(detail_text)
        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(f"skill-card ID is absent from the bundled catalog: {card_id}")
        available = tuple(
            int(value)
            for value in str(card.get("availableCustomizations", "")).split(",")
            if value
        )
        if not available:
            return ()
        if "カスタマイズ" not in compact:
            raise ArenaCatalogError("clicked skill-card detail is missing the customization section anchor")
        observations: list[CustomizationObservation] = []
        seen_names: set[str] = set()
        for customization_id in available:
            definition = self._customizations.get(customization_id)
            if definition is None:
                raise ArenaCatalogError(f"customization {customization_id} is absent from the bundled catalog")
            name = _normalise_text(definition.get("name"))
            if not name or name not in compact or name in seen_names:
                continue
            seen_names.add(name)
            maximum = definition.get("max")
            if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
                raise ArenaCatalogError(f"customization {customization_id} has an invalid maximum")
            count = 1
            if maximum > 1:
                tail = compact[compact.index(name) + len(name) :][:20]
                match = re.search(r"(?:Lv\.?|レベル|×|x)([1-9]\d*)", tail, re.IGNORECASE)
                if match is None:
                    raise ArenaCatalogError(
                        f"clicked customization {customization_id} requires an explicit level/count"
                    )
                count = int(match.group(1))
            observations.append(CustomizationObservation(name, count, compact))
        return tuple(observations)

    def resolve_customizations(
        self,
        card_id: int,
        observations: Sequence[CustomizationObservation],
    ) -> dict[str, int]:
        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(f"skill-card ID is absent from the bundled catalog: {card_id}")
        available = tuple(
            int(value)
            for value in str(card.get("availableCustomizations", "")).split(",")
            if value
        )
        resolved: dict[str, int] = {}
        for observation in observations:
            if isinstance(observation.count, bool) or observation.count < 1:
                raise ArenaCatalogError("customization count must be a positive integer")
            candidates = [
                customization_id
                for customization_id in available
                if _normalise_text(self._customizations.get(customization_id, {}).get("name"))
                == _normalise_text(observation.name)
            ]
            candidates = self._disambiguate_same_name(candidates, observation.detail_text)
            if len(candidates) != 1:
                raise ArenaCatalogError(
                    f"clicked customization detail must resolve exactly once for card {card_id}: {observation.name!r}"
                )
            customization_id = candidates[0]
            maximum = self._customizations[customization_id].get("max")
            if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= observation.count <= maximum:
                raise ArenaCatalogError(
                    f"customization {customization_id} count {observation.count} exceeds bundled maximum"
                )
            key = str(customization_id)
            if key in resolved:
                raise ArenaCatalogError(f"customization {customization_id} appears more than once in clicked details")
            resolved[key] = observation.count
        return resolved

    def resolve_effective_customizations(
        self,
        card_id: int,
        detail_text: str,
        *,
        expected_count: int,
        allow_positive_completion: bool = True,
    ) -> dict[str, int]:
        """Resolve a clicked contest card from its final, already-applied effect text.

        Contest details do not expose customization names.  This method only
        accepts cards whose every legal customization has a supported visible
        signature, enumerates every legal combination matching the badge count,
        and requires exactly one combination to agree with the final text.
        Unsupported or non-unique cards fail closed.
        """

        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
            raise ArenaCatalogError("expected customization count must be a non-negative integer")
        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(f"skill-card ID is absent from the bundled catalog: {card_id}")
        available = tuple(
            int(value)
            for value in str(card.get("availableCustomizations", "")).split(",")
            if value
        )
        if not available:
            if expected_count:
                raise ArenaCatalogError(
                    f"card {card_id} cannot have {expected_count} customizations in the bundled catalog"
                )
            return {}
        matchers = {
            customization_id: self._effective_detail_matcher(card, customization_id)
            for customization_id in available
        }
        unsupported = [customization_id for customization_id, matcher in matchers.items() if matcher is None]
        if unsupported:
            raise ArenaCatalogError(
                f"card {card_id} has unsupported final-effect signatures for customizations {unsupported}"
            )

        compact = _normalise_text(detail_text)
        if expected_count == 0:
            if all(matchers[customization_id](compact, 0) for customization_id in available):
                return {}
            raise ArenaCatalogError(
                f"final skill-card effects do not match the uncustomized form of card {card_id}"
            )

        legal_groups = self.legal_customization_groups(
            card_id,
            expected_count=expected_count,
        )
        strict_candidates = [
            group
            for group in legal_groups
            if all(
                matchers[customization_id](
                    compact,
                    group.get(str(customization_id), 0),
                )
                for customization_id in available
            )
        ]

        # Prefer a group proved by visible customized values. Missing text is
        # unknown, not evidence for level zero: full-screen contest OCR can
        # omit one effect row while preserving the other rows. The trusted
        # badge count and finite catalog close the remaining degrees of
        # freedom. A matcher that also accepts level zero is non-discriminating
        # and contributes no positive constraint. The strict path is retained
        # for catalog effects whose stable UI representation is intentionally
        # an omitted/unchanged row, but the backend confirms all multi-solution
        # results from a fresh frame before they can enter an engine payload.
        positive_levels: dict[int, frozenset[int]] = {}
        for customization_id in available:
            matcher = matchers[customization_id]
            maximum = int(self._customizations[customization_id]["max"])
            zero_matches = matcher(compact, 0)
            positive_levels[customization_id] = frozenset(
                level
                for level in range(1, maximum + 1)
                if not zero_matches and matcher(compact, level)
            )
        positive_evidence_count = sum(bool(levels) for levels in positive_levels.values())
        positive_candidates = [
            group
            for group in legal_groups
            if all(
                not positive_levels[customization_id]
                or group.get(str(customization_id), 0)
                in positive_levels[customization_id]
                for customization_id in available
            )
        ]
        if len(strict_candidates) == 1:
            return strict_candidates[0]
        if (
            allow_positive_completion
            and positive_evidence_count
            and len(positive_candidates) == 1
        ):
            return positive_candidates[0]
        raise ArenaCatalogError(
            f"final skill-card effects must resolve exactly once for card {card_id} "
            f"and badge count {expected_count}; "
            f"strict_candidates={strict_candidates!r}; "
            f"positive_candidates={positive_candidates!r}"
        )

    def resolve_effective_customizations_without_badge_count(
        self,
        card_id: int,
        detail_text: str,
    ) -> dict[str, int]:
        """Infer one positive badge count from an authorized clicked detail.

        This narrow fallback is only for clean-reference-positive cards whose
        bounded digit reads remained unreadable.  It enumerates the finite
        catalog counts and still requires one unique final-effect match.
        """

        available = self.available_customization_ids(card_id)
        if not available:
            raise ArenaCatalogError(
                f"card {card_id} cannot infer a positive customization count"
            )
        maximum_total = sum(int(self._customizations[item]["max"]) for item in available)
        matches: list[dict[str, int]] = []
        for expected_count in range(1, maximum_total + 1):
            try:
                resolved = self.resolve_effective_customizations(
                    card_id,
                    detail_text,
                    expected_count=expected_count,
                    allow_positive_completion=False,
                )
            except ArenaCatalogError:
                continue
            if sum(resolved.values()) == expected_count:
                matches.append(resolved)
        unique = {tuple(sorted(item.items())): item for item in matches}
        if len(unique) != 1:
            raise ArenaCatalogError(
                f"clicked card {card_id} must resolve exactly one positive customization count"
            )
        return next(iter(unique.values()))

    def resolve_effective_customizations_with_generic_cost_evidence(
        self,
        card_id: int,
        detail_text: str,
        *,
        generic_cost_frame_values: Sequence[int],
    ) -> dict[str, int]:
        """Resolve one positive group from detail plus the same card's face cost.

        The face badge does not supply the count.  Generic-cost effects are
        rendered only on the source card face, so detail text alone can leave
        two otherwise identical groups.  Three stable source-frame reads close
        that degree of freedom without adding another UI action.
        """

        if self.generic_cost_hypotheses(card_id) is None:
            raise ArenaCatalogError(
                f"card {card_id} has no generic-cost customization evidence"
            )
        available = self.available_customization_ids(card_id)
        maximum_total = sum(
            int(self._customizations[item]["max"])
            for item in available
        )
        matches: list[dict[str, int]] = []
        for expected_count in range(1, maximum_total + 1):
            try:
                resolved = self.resolve_effective_customizations(
                    card_id,
                    detail_text,
                    expected_count=expected_count,
                    allow_positive_completion=False,
                )
                self.certify_card_face_generic_cost(
                    card_id,
                    resolved=resolved,
                    frame_values=generic_cost_frame_values,
                )
            except ArenaCatalogError:
                continue
            if sum(resolved.values()) == expected_count:
                matches.append(resolved)
        unique = {tuple(sorted(item.items())): item for item in matches}
        if len(unique) != 1:
            raise ArenaCatalogError(
                f"clicked card {card_id} detail and card-face cost must resolve "
                "exactly one positive customization state"
            )
        return next(iter(unique.values()))

    def resolve_clicked_customizations(
        self,
        card_id: int,
        detail_text: str,
        *,
        observed_badge_count: int,
        generic_cost_frame_values: Sequence[int] | None = None,
    ) -> EffectiveCustomizationResolution:
        """Resolve detail truth before using the fallible card-face digit.

        A unique positive final-effect combination is authoritative. The
        observed badge count is used only when the detail admits more than one
        legal positive combination. This keeps count-only disambiguation for
        effects that are not visibly rendered while preventing a wrong face
        digit from discarding a unique detail result.
        """

        if (
            isinstance(observed_badge_count, bool)
            or not isinstance(observed_badge_count, int)
            or observed_badge_count < 1
        ):
            raise ArenaCatalogError(
                "clicked customization resolution requires a positive observed badge count"
            )
        customizations = None
        source = ""
        if generic_cost_frame_values is not None:
            try:
                customizations = (
                    self.resolve_effective_customizations_with_generic_cost_evidence(
                        card_id,
                        detail_text,
                        generic_cost_frame_values=generic_cost_frame_values,
                    )
                )
                source = "detail_card_face_unique"
            except ArenaCatalogError:
                pass
        if customizations is None:
            try:
                customizations = self.resolve_effective_customizations_without_badge_count(
                    card_id,
                    detail_text,
                )
                source = "detail_unique"
            except ArenaCatalogError:
                customizations = self.resolve_effective_customizations(
                    card_id,
                    detail_text,
                    expected_count=observed_badge_count,
                )
                source = "badge_constrained"
        resolved_count = sum(int(value) for value in customizations.values())
        if resolved_count < 1:
            raise ArenaCatalogError(
                f"clicked card {card_id} did not resolve a positive customization state"
            )
        return EffectiveCustomizationResolution(
            customizations=dict(customizations),
            resolved_count=resolved_count,
            source=source,
            observed_badge_count=observed_badge_count,
            badge_count_match=resolved_count == observed_badge_count,
            evidence_mode=self.effective_customization_evidence_mode(
                card_id,
                detail_text,
                expected_count=resolved_count,
                resolved=customizations,
            ),
        )

    def resolve_effective_customizations_unconstrained(
        self,
        card_id: int,
        detail_text: str,
    ) -> dict[str, int]:
        """Resolve zero or one unique positive combination without badge input."""

        available = self.available_customization_ids(card_id)
        if not available:
            return {}
        maximum_total = sum(int(self._customizations[item]["max"]) for item in available)
        matches: list[dict[str, int]] = []
        for expected_count in range(0, maximum_total + 1):
            try:
                resolved = self.resolve_effective_customizations(
                    card_id,
                    detail_text,
                    expected_count=expected_count,
                    allow_positive_completion=False,
                )
            except ArenaCatalogError:
                continue
            if sum(resolved.values()) == expected_count:
                matches.append(resolved)
        unique = {tuple(sorted(item.items())): item for item in matches}
        if len(unique) != 1:
            raise ArenaCatalogError(
                f"clicked card {card_id} must resolve exactly one zero-or-positive customization state"
            )
        return next(iter(unique.values()))

    def resolve_generic_cost_ambiguity(
        self,
        card_id: int,
        detail_text: str,
        *,
        observed_badge_count: int | None,
    ) -> GenericCostAmbiguityResolution:
        """Prove that one unreadable card-face cost is the only open bit.

        The result is deliberately a pair rather than an inferred truth.  It
        is valid only when the fixed catalog, settled detail text and every
        non-cost customization leave exactly two legal states.  Those states
        must differ solely by one single-level ``g.cost+=`` customization.
        """

        if observed_badge_count is not None and (
            isinstance(observed_badge_count, bool)
            or not isinstance(observed_badge_count, int)
            or observed_badge_count < 1
        ):
            raise ArenaCatalogError(
                "generic-cost ambiguity requires a positive badge count or no count"
            )
        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        available = self.available_customization_ids(card_id)
        generic_ids = tuple(
            customization_id
            for customization_id in available
            if self._pure_generic_cost_delta(customization_id) is not None
        )
        if len(generic_ids) != 1:
            raise ArenaCatalogError(
                f"card {card_id} must have exactly one generic-cost customization"
            )
        generic_id = generic_ids[0]
        if int(self._customizations[generic_id]["max"]) != 1:
            raise ArenaCatalogError(
                f"card {card_id} generic-cost customization must be single-level"
            )
        # This also proves that the fixed DSL exposes two distinct card-face
        # hypotheses.  The caller separately proves that their visual carrier
        # was attempted but remained inconclusive.
        self.generic_cost_hypotheses(card_id)

        matchers = {
            customization_id: self._effective_detail_matcher(
                card,
                customization_id,
            )
            for customization_id in available
        }
        unsupported = tuple(
            customization_id
            for customization_id, matcher in matchers.items()
            if matcher is None
        )
        if unsupported:
            raise ArenaCatalogError(
                f"card {card_id} has unsupported final-effect signatures for customizations {unsupported!r}"
            )
        compact = _normalise_text(detail_text)
        maximum_total = sum(
            int(self._customizations[customization_id]["max"])
            for customization_id in available
        )
        candidates = tuple(
            group
            for expected_count in range(maximum_total + 1)
            for group in self.legal_customization_groups(
                card_id,
                expected_count=expected_count,
            )
            if all(
                matchers[customization_id](
                    compact,
                    group.get(str(customization_id), 0),
                )
                for customization_id in available
            )
        )
        if len(candidates) != 2:
            raise ArenaCatalogError(
                f"card {card_id} detail must leave exactly two generic-cost states; candidates={candidates!r}"
            )

        by_generic_level: dict[int, dict[str, int]] = {}
        non_cost_states: set[tuple[tuple[str, int], ...]] = set()
        for candidate in candidates:
            generic_level = int(candidate.get(str(generic_id), 0))
            if generic_level not in (0, 1) or generic_level in by_generic_level:
                raise ArenaCatalogError(
                    f"card {card_id} detail does not leave one state per generic-cost level"
                )
            normalized = {
                str(key): int(value)
                for key, value in candidate.items()
                if int(value) > 0
            }
            by_generic_level[generic_level] = normalized
            non_cost_states.add(
                tuple(
                    sorted(
                        (key, value)
                        for key, value in normalized.items()
                        if key != str(generic_id)
                    )
                )
            )
        if set(by_generic_level) != {0, 1} or len(non_cost_states) != 1:
            raise ArenaCatalogError(
                f"card {card_id} detail ambiguity is not limited to generic cost"
            )
        if observed_badge_count is not None and observed_badge_count not in {
            sum(candidate.values()) for candidate in by_generic_level.values()
        }:
            raise ArenaCatalogError(
                f"card {card_id} badge count {observed_badge_count} is outside the two generic-cost states"
            )

        # Missing non-cost rows are not reclassified as zero.  Every other
        # selected level must have a visible positive signature, and every
        # zero level must have an explicitly rendered baseline.  This keeps
        # the bounded assumption strictly on the generic-cost bit.
        representative = by_generic_level[0]
        for customization_id in available:
            if customization_id == generic_id:
                continue
            level = int(representative.get(str(customization_id), 0))
            visible = (
                self._positive_signature_is_visible(
                    card,
                    customization_id,
                    compact,
                    level,
                )
                if level > 0
                else self._zero_signature_is_visible(
                    card,
                    customization_id,
                    compact,
                )
            )
            if not visible:
                raise ArenaCatalogError(
                    f"card {card_id} non-cost customization {customization_id} level {level} lacks an explicit detail signature"
                )

        return GenericCostAmbiguityResolution(
            generic_customization_id=generic_id,
            unenhanced_customizations=dict(by_generic_level[0]),
            enhanced_customizations=dict(by_generic_level[1]),
        )

    def _validate_customization_index(self) -> None:
        """Reject a mismatched or incomplete fixed ``gakumas-data`` snapshot."""

        if any(customization_id < 1 for customization_id in self._customizations):
            raise ArenaCatalogError("bundled customization IDs must be unique positive integers")
        referenced: set[int] = set()
        for card_id, card in self._cards_by_id.items():
            raw_ids = tuple(
                value
                for value in str(card.get("availableCustomizations", "")).split(",")
                if value
            )
            try:
                customization_ids = tuple(int(value) for value in raw_ids)
            except ValueError as error:
                raise ArenaCatalogError(
                    f"card {card_id} has a non-integer customization reference"
                ) from error
            if len(customization_ids) != len(set(customization_ids)):
                raise ArenaCatalogError(f"card {card_id} repeats a customization ID")
            missing = [value for value in customization_ids if value not in self._customizations]
            if missing:
                raise ArenaCatalogError(f"card {card_id} references missing customizations {missing!r}")
            referenced.update(customization_ids)
        for customization_id, definition in self._customizations.items():
            maximum = definition.get("max")
            if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
                raise ArenaCatalogError(f"customization {customization_id} has an invalid maximum")
            self._pure_generic_cost_delta(customization_id)
        unreferenced = sorted(set(self._customizations) - referenced)
        if unreferenced:
            raise ArenaCatalogError(f"bundled customizations are unreferenced: {unreferenced!r}")

    def _pure_generic_cost_delta(self, customization_id: int) -> int | None:
        """Return one pure card-face cost delta or reject an impure definition."""

        definition = self._customizations.get(customization_id)
        if definition is None:
            raise ArenaCatalogError(
                f"customization {customization_id} is absent from the bundled catalog"
            )
        effects = definition.get("effects")
        compact_effects = _normalise_text(effects)
        if "g.cost+=" not in compact_effects:
            return None
        empty_fields = tuple(
            field
            for field in self._GENERIC_COST_EMPTY_FIELDS
            if (value := definition.get(field)) is not None
            and (not isinstance(value, str) or bool(value.strip()))
        )
        maximum = definition.get("max")
        match = self._PURE_GENERIC_COST_EFFECT.fullmatch(compact_effects)
        if (
            definition.get("type") != "cost"
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum != 1
            or empty_fields
            or match is None
        ):
            raise ArenaCatalogError(
                f"customization {customization_id} generic-cost DSL is not a pure "
                "single-level card-face cost change"
            )
        return int(match.group(1))

    def _effective_detail_matcher(
        self,
        card: Mapping[str, Any],
        customization_id: int,
    ) -> Any:
        definition = self._customizations.get(customization_id)
        if definition is None:
            raise ArenaCatalogError(
                f"customization {customization_id} is absent from the bundled catalog"
            )

        typed_cost = re.fullmatch(
            r"(concentration|cost|stamina|fullPowerCharge|goodConditionTurns|goodImpressionTurns|motivation|perfectConditionTurns)-=([0-9]+)",
            str(card.get("cost", "")),
        )
        cost_delta = re.search(
            r"g\.typedCost\+=([0-9]+)", str(definition.get("effects", ""))
        )
        if typed_cost and cost_delta:
            labels = {
                "concentration": "集中消費",
                "cost": "体力消費",
                "stamina": "体力消費",
                "fullPowerCharge": "全力値消費",
                "goodConditionTurns": "好調消費",
                "goodImpressionTurns": "好印象消費",
                "motivation": "やる気消費",
                "perfectConditionTurns": "絶好調消費",
            }
            label = labels[typed_cost.group(1)]
            base_value = int(typed_cost.group(2))
            delta = int(cost_delta.group(1))

            def match_typed_cost(compact: str, count: int) -> bool:
                expected = max(0, base_value - delta * count)
                if expected == 0:
                    return label not in compact or _contains_exact_integer_token(
                        compact,
                        label,
                        0,
                    )
                return _contains_exact_integer_token(
                    compact,
                    label,
                    expected,
                )

            return match_typed_cost

        if "g.scoreTimes+=1" in str(definition.get("effects", "")):
            customization_effects = str(definition.get("effects", ""))
            base_score_values = _all_integer_increments(
                card.get("actions"),
                "score",
            )
            base_growth_score = (
                re.search(
                    r"@grow\s+.*?target:this\s*\{\s*g\.score\+=([0-9]+)",
                    str(card.get("effects", "")),
                )
                if "@grow" in customization_effects
                else None
            )

            def match_score_times(compact: str, count: int) -> bool:
                if base_growth_score is None:
                    if not base_score_values:
                        return False
                    repeated = "(2回)" in compact or "（2回）" in compact
                    return repeated is (count > 0)
                score = int(base_growth_score.group(1))
                base_visible = _contains_exact_integer_token(
                    compact,
                    "スコア値増加+",
                    score,
                )
                times_visible = (
                    _contains_exact_integer_token(
                        compact,
                        "スコア上昇回数増加+",
                        1,
                    )
                    and "2回まで" in compact
                )
                return (
                    base_visible and times_visible
                    if count > 0
                    else base_visible and not times_visible
                )

            return match_score_times

        scalar_labels = {
            "goodConditionTurns": ("好調", "ターン"),
            "goodImpressionTurns": ("好印象+", ""),
            "perfectConditionTurns": ("絶好調", "ターン"),
            "concentration": ("集中+", ""),
            "genki": ("元気+", ""),
            "motivation": ("やる気+", ""),
            "fullPowerCharge": ("全力値+", ""),
            "halfCostTurns": ("消費体力減少", "ターン"),
            "score": ("スコア+", ""),
        }
        for field, (prefix, suffix) in scalar_labels.items():
            maximum = int(definition["max"])
            deltas = {
                level: _growth_increment(definition.get("effects"), field, level)
                for level in range(1, maximum + 1)
            }
            if not any(value is not None for value in deltas.values()):
                continue
            base_value = _first_integer_increment(card.get("actions"), field)
            if base_value is None:
                return None

            def match_scalar_growth(
                compact: str,
                count: int,
                *,
                base_value: int = base_value,
                deltas: Mapping[int, int | None] = deltas,
                prefix: str = prefix,
                suffix: str = suffix,
            ) -> bool:
                delta = 0 if count == 0 else deltas.get(count)
                if delta is None:
                    return False
                return _contains_exact_integer_token(
                    compact,
                    prefix,
                    base_value + delta,
                    suffix,
                )

            return match_scalar_growth

        simple_action_labels = {
            "goodConditionTurns": ("好調", "ターン"),
            "goodImpressionTurns": ("好印象+", ""),
            "perfectConditionTurns": ("絶好調", "ターン"),
            "concentration": ("集中+", ""),
            "genki": ("元気+", ""),
            "motivation": ("やる気+", ""),
            "fullPowerCharge": ("全力値+", ""),
            "halfCostTurns": ("消費体力減少", "ターン"),
            "turnsRemaining": ("ターン追加+", ""),
            "score": ("スコア+", ""),
        }
        for field, (prefix, suffix) in simple_action_labels.items():
            added_value = _first_integer_increment(definition.get("actions"), field)
            if added_value is None:
                continue
            token = f"{prefix}{added_value}{suffix}"
            base_has_same_token = _first_integer_increment(card.get("actions"), field) == added_value

            def match_added_action(
                compact: str,
                count: int,
                *,
                prefix: str = prefix,
                added_value: int = added_value,
                suffix: str = suffix,
                base_has_same_token: bool = base_has_same_token,
            ) -> bool:
                if base_has_same_token:
                    return True
                visible = _contains_exact_integer_token(
                    compact,
                    prefix,
                    added_value,
                    suffix,
                )
                return visible is (count > 0)

            return match_added_action

        if definition.get("forceInitialHand") is True:

            def match_force_initial_hand(compact: str, count: int) -> bool:
                visible = "試験・ステージ開始時手札に入る" in compact
                return visible is (count > 0)

            return match_force_initial_hand

        card_use_delta = re.fullmatch(
            r"cardUsesRemaining\+=([0-9]+)",
            _normalise_text(definition.get("actions")),
        )
        if card_use_delta:
            added_uses = int(card_use_delta.group(1))
            base_has_same_addition = (
                re.search(
                    rf"(?:^|;)cardUsesRemaining\+={added_uses}(?:;|$)",
                    _normalise_text(card.get("actions")),
                )
                is not None
            )

            def match_card_use_addition(compact: str, count: int) -> bool:
                if base_has_same_addition:
                    return True
                # Full-screen contest OCR may interleave the right-hand +N
                # value with parameter columns. The label itself is absent
                # from the base card and therefore remains the stable signal.
                visible = "スキルカード使用数追加" in compact
                return visible is (count > 0)

            return match_card_use_addition

        condition = str(definition.get("conditions", ""))
        condition_match = re.fullmatch(
            r"@usable if:(goodConditionTurns|goodImpressionTurns|genki|preservationTimes)>=([0-9]+)",
            condition,
        )
        if condition_match:
            field, custom_threshold_text = condition_match.groups()
            base_match = re.search(
                rf"@usable if:{re.escape(field)}>=([0-9]+)",
                str(card.get("conditions", "")),
            )
            if base_match is None:
                return None
            base_threshold = int(base_match.group(1))
            custom_threshold = int(custom_threshold_text)

            def match_condition_threshold(compact: str, count: int) -> bool:
                expected = custom_threshold if count > 0 else base_threshold
                if field == "goodConditionTurns":
                    return f"好調が{expected}ターン以上" in compact
                if field == "goodImpressionTurns":
                    return f"好印象が{expected}以上" in compact
                if field == "preservationTimes":
                    return f"温存になった回数が{expected}回以上" in compact
                return f"元気が{expected}以上" in compact

            return match_condition_threshold

        actions = str(definition.get("actions", ""))
        trigger_condition_match = re.fullmatch(
            r"@trigger if:(goodImpressionTurns)>=([0-9]+)",
            actions,
        )
        if trigger_condition_match:
            field, custom_threshold_text = trigger_condition_match.groups()
            base_match = re.search(
                rf"@trigger if:{re.escape(field)}>=([0-9]+)",
                str(card.get("actions", "")),
            )
            if base_match is None:
                return None
            base_threshold = int(base_match.group(1))
            custom_threshold = int(custom_threshold_text)

            def match_trigger_condition(compact: str, count: int) -> bool:
                expected = custom_threshold if count > 0 else base_threshold
                return f"好印象が{expected}以上" in compact

            return match_trigger_condition

        direct_good_impression_multiplier = re.search(
            r"@trigger at:goodImpressionTurnsIncreased\s*\{.*?"
            r"score\+=goodImpressionTurns\*([0-9]+(?:\.[0-9]+)?)",
            actions,
        )
        if direct_good_impression_multiplier:
            percent = int(round(float(direct_good_impression_multiplier.group(1)) * 100))

            def match_direct_good_impression_multiplier(compact: str, count: int) -> bool:
                visible = f"{percent}%分スコア" in compact
                if percent >= 100:
                    # Narrow contest OCR may lose the hundreds digit while
                    # preserving the distinguishing fractional percentage.
                    visible = visible or f"{percent % 100}%分スコア" in compact
                return visible is (count > 0)

            return match_direct_good_impression_multiplier

        added_good_impression_multiplier = re.fullmatch(
            r"score\+=goodImpressionTurns\*([0-9]+(?:\.[0-9]+)?)",
            actions,
        )
        if added_good_impression_multiplier:
            percent = int(
                round(float(added_good_impression_multiplier.group(1)) * 100)
            )

            def match_added_good_impression_multiplier(
                compact: str,
                count: int,
            ) -> bool:
                visible = f"好印象の{percent}%分スコア" in compact
                return visible is (count > 0)

            return match_added_good_impression_multiplier

        direct_motivation_multiplier = re.fullmatch(
            r"score\+=motivation\*([0-9]+(?:\.[0-9]+)?)",
            actions,
        )
        if direct_motivation_multiplier:
            percent = int(round(float(direct_motivation_multiplier.group(1)) * 100))

            def match_direct_motivation_multiplier(compact: str, count: int) -> bool:
                visible = f"やる気の{percent}%分スコア" in compact
                return visible is (count > 0)

            return match_direct_motivation_multiplier

        direct_genki_multiplier = re.fullmatch(
            r"score\+=genki\*([0-9]+(?:\.[0-9]+)?)",
            actions,
        )
        if direct_genki_multiplier:
            percent = int(round(float(direct_genki_multiplier.group(1)) * 100))

            def match_direct_genki_multiplier(compact: str, count: int) -> bool:
                visible = f"元気の{percent}%分スコア" in compact
                return visible is (count > 0)

            return match_direct_genki_multiplier

        coefficient_levels: dict[int, float] = {}
        for level in range(1, int(definition["max"]) + 1):
            selected = _selected_level_patch(actions, level)
            match = re.search(r"score\+=genki\*([0-9]+(?:\.[0-9]+)?)", selected)
            if match:
                coefficient_levels[level] = float(match.group(1))
        if coefficient_levels:
            base_match = re.search(
                r"score\+=genki\*([0-9]+(?:\.[0-9]+)?)",
                str(card.get("actions", "")),
            )
            if base_match is None:
                return None
            base_coefficient = float(base_match.group(1))

            def match_genki_coefficient(compact: str, count: int) -> bool:
                coefficient = base_coefficient if count == 0 else coefficient_levels.get(count)
                if coefficient is None:
                    return False
                percent = int(round(coefficient * 100))
                return f"元気の{percent}%分スコア" in compact

            return match_genki_coefficient

        score_by_genki_levels: dict[int, float] = {}
        for level in range(1, int(definition["max"]) + 1):
            selected = _selected_level_patch(definition.get("effects"), level)
            match = re.search(r"g\.scoreByGenki\+=([0-9]+(?:\.[0-9]+)?)", selected)
            if match:
                score_by_genki_levels[level] = float(match.group(1))
        if score_by_genki_levels:
            base_match = re.search(
                r"score\+=genki\*([0-9]+(?:\.[0-9]+)?)",
                str(card.get("actions", "")),
            )
            if base_match is None:
                return None
            base_coefficient = float(base_match.group(1))

            def match_score_by_genki_growth(compact: str, count: int) -> bool:
                increment = 0.0 if count == 0 else score_by_genki_levels.get(count)
                if increment is None:
                    return False
                percent = int(round((base_coefficient + increment) * 100))
                return f"元気の{percent}%分スコア" in compact

            return match_score_by_genki_growth

        score_by_motivation_levels: dict[int, float] = {}
        for level in range(1, int(definition["max"]) + 1):
            selected = _selected_level_patch(definition.get("effects"), level)
            match = re.search(
                r"g\.scoreByMotivation\+=([0-9]+(?:\.[0-9]+)?)",
                selected,
            )
            if match:
                score_by_motivation_levels[level] = float(match.group(1))
        if score_by_motivation_levels:
            base_match = re.search(
                r"score\+=motivation\*([0-9]+(?:\.[0-9]+)?)",
                str(card.get("actions", "")),
            )
            if base_match is None:
                return None
            base_coefficient = float(base_match.group(1))

            def match_score_by_motivation_growth(
                compact: str,
                count: int,
            ) -> bool:
                increment = (
                    0.0
                    if count == 0
                    else score_by_motivation_levels.get(count)
                )
                if increment is None:
                    return False
                percent = int(round((base_coefficient + increment) * 100))
                return f"やる気の{percent}%分スコア" in compact

            return match_score_by_motivation_growth

        score_by_good_impression_levels: dict[int, float] = {}
        for level in range(1, int(definition["max"]) + 1):
            selected = _selected_level_patch(definition.get("effects"), level)
            match = re.search(
                r"g\.scoreByGoodImpressionTurns\+=([0-9]+(?:\.[0-9]+)?)",
                selected,
            )
            if match:
                score_by_good_impression_levels[level] = float(match.group(1))
        if score_by_good_impression_levels:
            base_match = re.search(
                r"score\+=goodImpressionTurns\*([0-9]+(?:\.[0-9]+)?)",
                str(card.get("actions", "")),
            )
            if base_match is None:
                return None
            base_coefficient = float(base_match.group(1))

            def match_score_by_good_impression_growth(
                compact: str,
                count: int,
            ) -> bool:
                increment = (
                    0.0
                    if count == 0
                    else score_by_good_impression_levels.get(count)
                )
                if increment is None:
                    return False
                percent = int(round((base_coefficient + increment) * 100))
                return f"好印象の{percent}%分スコア" in compact

            return match_score_by_good_impression_growth

        hand_growth_levels: dict[int, int] = {}
        for level in range(1, int(definition["max"]) + 1):
            selected = _selected_level_patch(actions, level)
            match = re.search(r"target:hand\s*\{\s*g\.score\+=([0-9]+)", selected)
            if match:
                hand_growth_levels[level] = int(match.group(1))
        if hand_growth_levels:

            def match_hand_growth(compact: str, count: int) -> bool:
                target_visible = "手札の" in compact
                if count == 0:
                    return not (target_visible and "スコア値増加+" in compact)
                expected = hand_growth_levels.get(count)
                return (
                    expected is not None
                    and target_visible
                    and _contains_exact_integer_token(
                        compact,
                        "スコア値増加+",
                        expected,
                    )
                )

            return match_hand_growth

        held_growth_levels: dict[int, int] = {}
        for level in range(1, int(definition["max"]) + 1):
            selected = _selected_level_patch(actions, level)
            match = re.search(r"target:held\s*\{\s*g\.score\+=([0-9]+)", selected)
            if match:
                held_growth_levels[level] = int(match.group(1))
        if held_growth_levels:

            def match_held_growth(compact: str, count: int) -> bool:
                target_visible = "保留" in compact
                if count == 0:
                    return not (target_visible and "スコア値増加+" in compact)
                expected = held_growth_levels.get(count)
                return (
                    expected is not None
                    and target_visible
                    and _contains_exact_integer_token(
                        compact,
                        "スコア値増加+",
                        expected,
                    )
                )

            return match_held_growth

        all_growth_levels: dict[int, int] = {}
        for level in range(1, int(definition["max"]) + 1):
            selected = _selected_level_patch(actions, level)
            match = re.search(r"target:all\s*\{\s*g\.score\+=([0-9]+)", selected)
            if match:
                all_growth_levels[level] = int(match.group(1))
        if all_growth_levels:

            def match_all_growth(compact: str, count: int) -> bool:
                # Keep the distinctive suffix because the narrow OCR may
                # confuse the leading ス with 複 while preserving the value.
                marker = "コア値増加+"
                if count == 0:
                    return marker not in compact
                expected = all_growth_levels.get(count)
                return expected is not None and _contains_exact_integer_token(
                    compact,
                    marker,
                    expected,
                )

            return match_all_growth

        if "holdSelected[deck]" in actions:

            def match_hold_from_deck(compact: str, count: int) -> bool:
                visible = all(token in compact for token in ("山札", "選択", "保留"))
                return visible is (count > 0)

            return match_hold_from_deck

        if re.search(r"(?:^|[;{]\s*)setStance\(strength\)", actions):

            def match_strength_add(compact: str, count: int) -> bool:
                visible = "強気に変更" in compact
                return visible is (count > 0)

            return match_strength_add

        if _normalise_text(actions) == "setStance(preservation)":

            def match_preservation_add(compact: str, count: int) -> bool:
                visible = "温存に変更" in compact
                return visible is (count > 0)

            return match_preservation_add

        if "at:turn { setStance(preservation); limit:1 }" in actions:

            def match_next_turn_preservation(compact: str, count: int) -> bool:
                visible = "次のターン、温存に変更" in compact
                return visible is (count > 0)

            return match_next_turn_preservation

        if "g.stanceLevel+=1" in str(definition.get("effects", "")):
            base_stance = re.search(r"setStance\((strength|preservation)\)", str(card.get("actions", "")))
            if base_stance is None:
                return None
            label = "強気" if base_stance.group(1) == "strength" else "温存"

            def match_stance_level(compact: str, count: int) -> bool:
                advanced = f"{label}2段階目に変更" in compact
                base = f"{label}に変更" in compact
                return advanced if count > 0 else base and not advanced

            return match_stance_level

        if _normalise_text(definition.get("actions")) == "upgradeHand":

            def match_upgrade_hand(compact: str, count: int) -> bool:
                del compact, count
                # The contest detail does not render this applied action.  It
                # remains resolvable only by the other visible alternatives
                # plus the independently read badge total.
                return True

            return match_upgrade_hand

        if self._pure_generic_cost_delta(customization_id) is not None:

            def match_unrendered_generic_cost(compact: str, count: int) -> bool:
                del compact, count
                # Generic cost is carried by the card icon rather than a
                # stable OCR line in this detail layout.  Other alternatives
                # must make the legal group unique.
                return True

            return match_unrendered_generic_cost

        if definition.get("limit") == 0 and card.get("limit") == 1:

            def match_removed_once_limit(compact: str, count: int) -> bool:
                once_visible = "試験・ステージ中1回" in compact
                return once_visible is (count == 0)

            return match_removed_once_limit
        return None

    @staticmethod
    def _disambiguate_same_name(candidates: list[int], detail_text: str) -> list[int]:
        if len(candidates) <= 1:
            return candidates
        compact = _normalise_text(detail_text)
        # The fixed catalog has one same-card/name collision: 44 affects held
        # cards and 45 affects the hand.  The clicked detail must expose one of
        # these targets; otherwise the reader refuses to infer the ID.
        if set(candidates) == {44, 45}:
            if "保留" in compact and "手札" not in compact:
                return [44]
            if "手札" in compact and "保留" not in compact:
                return [45]
        return candidates
