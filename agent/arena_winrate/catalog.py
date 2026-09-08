"""Resolve clicked skill-card details against the version-matched engine data."""

from __future__ import annotations

import re
import json
import itertools
from typing import Any
from pathlib import Path
from collections import defaultdict
from dataclasses import replace, dataclass
from collections.abc import Mapping, Sequence

from ._p_item_detail_text import PItemDetailTextIndex, PItemDetailTextResult


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
    "～": "~",
}
# The fixed RIS catalog and the authoritative in-game/API titles differ for a
# small, already-frozen set of skill-card names.  The card-gallery training
# pipeline has to apply the same four mappings before joining its official
# rows.  Keep the runtime title resolver on that exact bounded contract as
# well; generic edit-distance or one-glyph insertion recovery would allow an
# unrelated card title to satisfy a detail click.
_SKILL_CARD_RENDERED_TITLE_REPLACEMENTS = {
    "ストレッチ談義": "ストレッチ談議",
    "愛をこめて": "愛を込めて",
    "月明りに包まれて": "月明かりに包まれて",
    "夢と現境界線": "夢と現の境界線",
}
_anchor_variant_sources: defaultdict[str, list[str]] = defaultdict(list)
for _source_character, _normalized_character in _TEXT_NORMALIZATION_REPLACEMENTS.items():
    _anchor_variant_sources[_normalized_character].append(_source_character)
_TEXT_ANCHOR_CHARACTER_VARIANTS = {
    normalized: tuple(dict.fromkeys((normalized, *sources)))
    for normalized, sources in _anchor_variant_sources.items()
}
_OCR_NUMERIC_TOKEN_BOUNDARY = "\u2063"
# Keep separately reconstructed OCR views lexically isolated.  A plain newline
# is removed by ``_normalise_effect_text`` and could otherwise manufacture a
# token from the end of one view and the start of another.
_OCR_EFFECT_VIEW_BOUNDARY = "\u241e"
_ADDED_NUMERIC_EFFECT_LABELS = {
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
_ADDED_NUMERIC_EFFECT_MARKERS = re.compile(
    "|".join(
        re.escape(prefix)
        for prefix in sorted(
            # Consume fixedGenki as its own complete marker, so its inner
            # 元気+ cannot be validated against an ordinary genki domain.
            ("固定元気+", *(value[0] for value in _ADDED_NUMERIC_EFFECT_LABELS.values())),
            key=len, reverse=True,
        )
    )
)
_ADDED_NUMERIC_INTEGER = re.compile(r"-?[0-9]+(?!\d|[.,．，。]\d)")


def _normalise_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    compact = "".join(value.split())
    return compact.translate(str.maketrans(_TEXT_NORMALIZATION_REPLACEMENTS))


def _normalise_skill_card_title_text(value: object) -> str:
    """Normalize only fixed catalog-to-rendered skill-card title aliases."""

    # The same rendered middle dot is returned as U+00B7 by title OCR and
    # stored as U+30FB in the catalog. Apply the equivalence to both indexed
    # titles and queries here, without changing effect text or P-item titles.
    compact = _normalise_text(value).replace("·", "・")
    for source, rendered in _SKILL_CARD_RENDERED_TITLE_REPLACEMENTS.items():
        compact = compact.replace(source, rendered)
    return compact


def _normalise_effect_text(value: object) -> str:
    """Compact effect OCR without joining independent numeric lines."""

    if not isinstance(value, str):
        return ""
    translated = value.translate(str.maketrans(_TEXT_NORMALIZATION_REPLACEMENTS))
    # OCR lines are independent lexical tokens.  Joining every whitespace
    # boundary can turn a valid value followed by an unrelated numeric line
    # (for example ``集中消費1\n2848``) into the false token ``12848``.  Keep
    # digit-to-digit newline boundaries explicit. Ordinary spaces and text
    # fragments still use the historical whitespace-tolerant normalization.
    protected = re.sub(
        r"(?<=\d)[^\S\r\n]*(?:\r\n|\r|\n)"
        r"(?:[^\S\r\n]*(?:\r\n|\r|\n))*[^\S\r\n]*(?=\d)",
        _OCR_NUMERIC_TOKEN_BOUNDARY,
        translated,
    )
    compact = "".join(protected.split())
    return compact


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
    # Title queries already equate these middle dots. Match either raw OCR
    # glyph here too, without changing shared effect or P-item normalization.
    return "".join(
        (
            re.escape(character)
            if len(variants := (
                ("・", "·") if character in ("・", "·")
                else _TEXT_ANCHOR_CHARACTER_VARIANTS.get(character, (character,))
            ))
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


def _all_literal_integer_increments(value: object, field: str) -> tuple[int, ...]:
    """Return only assignments whose complete right-hand side is an integer."""

    compact = _normalise_text(value)
    return tuple(
        int(match.group(1))
        for match in re.finditer(
            rf"(?<![.A-Za-z]){re.escape(field)}\+=(-?[0-9]+)(?=;|}}|$)",
            compact,
        )
    )


def _growth_increment(value: object, field: str, level: int) -> int | None:
    selected = _selected_level_patch(value, level)
    match = re.search(rf"g\.{re.escape(field)}\+=(-?[0-9]+)", selected)
    return int(match.group(1)) if match else None


def _direct_prestage_target_this_increment(
    value: object,
    field: str,
    level: int,
) -> int | None:
    """Return one positive direct card-growth increment, or fail closed.

    ``scoreTimes`` seeded on the card applies to the first score action.  The
    same token inside ``@grow``, another timing scope, or another target has a
    different meaning and cannot certify that rendered action row.  The fixed
    catalog's direct patches are deliberately required to be exactly one
    ``at:prestage -> target:this -> g.FIELD+=N`` body.
    """

    selected = _normalise_text(_selected_level_patch(value, level))
    match = re.fullmatch(
        rf"at:prestage\{{target:this\{{g\.{re.escape(field)}\+="
        r"([1-9][0-9]*);?\}\};?",
        selected,
    )
    return int(match.group(1)) if match is not None else None


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
        self._added_numeric_effect_domain_cache: dict[
            int, tuple[dict[str, tuple[int, ...]], dict[str, str]]
        ] = {}
        self._p_items = tuple(dict(row) for row in p_items)
        self._p_item_detail_text_index: PItemDetailTextIndex | None = None
        self._p_item_detail_text_index_loaded = False
        self._p_item_detail_text_index_error: str | None = None
        self._p_item_detail_fuzzy_titles: dict[int, tuple[tuple[str, int], ...]] | None = None
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
        by_display_title: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for card in self._cards:
            by_display_title[self._skill_card_display_title(card)].append(card)
        self._cards_by_display_title = {
            title: tuple(owners) for title, owners in by_display_title.items()
        }
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
            by_name[_normalise_skill_card_title_text(card.get("name"))].append(card)
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

    def _added_numeric_effect_domains(
        self,
        card_id: int,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
        """Prove complete domains only for optional literal-added action rows.

        Base-field references and other same-field DSL keep their established
        resolver paths. In particular, the first integer of an expression is
        never treated as the expression's complete set of rendered values.
        The cache belongs to this version-matched catalog instance only.
        """
        cached = self._added_numeric_effect_domain_cache.get(card_id)
        if cached is not None:
            return cached
        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(f"skill-card ID is absent from the bundled catalog: {card_id}")
        definitions = tuple(
            self._customizations[identifier]
            for identifier in self.available_customization_ids(card_id)
        )
        domains: dict[str, tuple[int, ...]] = {}
        exclusions: dict[str, str] = {}
        dsl_fields = ("conditions", "cost", "actions", "effects")
        for field, (prefix, _) in _ADDED_NUMERIC_EFFECT_LABELS.items():
            direct_assignment = re.compile(rf"(?<![.A-Za-z0-9_]){field}\+=")
            if not any(direct_assignment.search(_normalise_text(row.get("actions"))) for row in definitions):
                continue
            reference = re.compile(rf"(?<![A-Za-z0-9_]){field}(?![A-Za-z0-9_])")
            if any(reference.search(str(card.get(key) or "")) for key in dsl_fields):
                exclusions[field] = "base_mentions_field"
                continue
            if prefix in _normalise_skill_card_title_text(card.get("name")):
                exclusions[field] = "effect_marker_in_card_title"
                continue
            values: set[int] = set()
            for definition in definitions:
                if not any(reference.search(str(definition.get(key) or "")) for key in dsl_fields):
                    continue
                literal = re.fullmatch(
                    rf"{field}\+=(-?[0-9]+);?",
                    _normalise_text(definition.get("actions")),
                )
                if literal is None or any(
                    reference.search(str(definition.get(key) or ""))
                    for key in ("conditions", "cost", "effects")
                ):
                    exclusions[field] = "complex_same_field_customization"
                    break
                if definition.get("max") != 1:
                    exclusions[field] = "multi_level_added_action"
                    break
                values.add(int(literal.group(1)))
            if field not in exclusions and values:
                domains[field] = tuple(sorted(values))
        result = (domains, exclusions)
        self._added_numeric_effect_domain_cache[card_id] = result
        return result

    def _validate_added_numeric_effects(self, card_id: int, detail_text: str) -> None:
        """Reject corrupt present optional rows without redefining absent rows."""
        domains, _ = self._added_numeric_effect_domains(card_id)
        if not domains or not isinstance(detail_text, str):
            return
        by_marker = {
            _ADDED_NUMERIC_EFFECT_LABELS[field][0]: (
                field, _ADDED_NUMERIC_EFFECT_LABELS[field][1], values,
            )
            for field, values in domains.items()
        }
        for view_index, view in enumerate(detail_text.split(_OCR_EFFECT_VIEW_BOUNDARY)):
            compact = _normalise_effect_text(view)
            # The longest marker wins: 絶好調 is not 好調, and 固定元気+ is
            # not 元気+. A marker outside the optional domains stays untouched.
            for occurrence in _ADDED_NUMERIC_EFFECT_MARKERS.finditer(compact):
                domain = by_marker.get(occurrence.group())
                if domain is None:
                    continue
                field, suffix, values = domain
                tail = compact[occurrence.end():]
                number = _ADDED_NUMERIC_INTEGER.match(tail)
                if (
                    number is None
                    or number.group() != str(int(number.group()))
                    or int(number.group()) not in values
                    or not tail[number.end():].startswith(suffix)
                ):
                    raise ArenaCatalogError(
                        "malformed added numeric effect: "
                        f"card {card_id}, field {field}, view {view_index}, "
                        f"marker {occurrence.group()!r}, payload {tail[:32]!r}, "
                        f"allowed_values={values!r}, suffix={suffix!r}"
                    )

    def p_item_business_ids(self) -> tuple[int, ...]:
        """Return every P-item business ID declared by the active engine data."""

        return tuple(sorted(self._p_items_by_id))

    def arena_p_item_reference_required_ids(
        self,
        *,
        plan: str | None = None,
    ) -> tuple[int, ...]:
        """Return P-items that can affect an arena-stage simulation.

        The engine EntityBank arena selector requests ``modes: ["stage"]``;
        its coverage and generated-suite tests use the same reachability rule.
        Produce-mode rewards and H.I.F. fixtures therefore stay outside the
        contestant P-item reference domain.
        """

        return tuple(
            sorted(
                int(item["id"])
                for item in self._p_items
                if item.get("sourceType") in {"pIdol", "support"}
                and item.get("mode") == "stage"
                and (
                    plan is None
                    or plan == "free"
                    or item.get("plan") in {None, "", "free", plan}
                )
            )
        )

    def _get_p_item_detail_text_index(self) -> PItemDetailTextIndex | None:
        """Build the full active-catalog text relation once, only when needed."""

        if not self._p_item_detail_text_index_loaded:
            self._p_item_detail_text_index_loaded = True
            try:
                self._p_item_detail_text_index = PItemDetailTextIndex(self._p_items)
            except (OSError, ValueError) as error:
                # A missing asset must not cause repeated filesystem work for
                # each detail. It also must not invalidate image-only reads.
                self._p_item_detail_text_index_error = f"{type(error).__name__}: {error}"
        return self._p_item_detail_text_index

    def p_item_detail_title_family_ids(
        self,
        title_text: str,
        *,
        plan: str | None = None,
        allow_one_substitution: bool = False,
    ) -> tuple[int, ...]:
        """Find the complete title family in the active catalog, before images.

        Exact titles never depend on icon candidates. The optional legacy OCR
        substitution recovery retains its equal-length, suffix-preserving,
        globally unique restriction; its caller still owns the old additional
        visual boundary for that approximate-title path only.
        """

        index = self._get_p_item_detail_text_index()
        if index is None:
            return ()
        family = index.family_ids_for_title(title_text, plan=plan)
        if family or not allow_one_substitution:
            return family
        recovered = self._recover_p_item_title_one_substitution(title_text)
        return index.family_ids_for_title(recovered, plan=plan) if recovered else ()

    def _recover_p_item_title_one_substitution(self, title_text: str) -> str | None:
        """Keep the original global title ambiguity rule for legacy recovery."""

        title = _normalise_text(title_text)
        core = title[:-1] if title.endswith("+") else title
        if len(core) < 4:
            return None
        if self._p_item_detail_fuzzy_titles is None:
            by_length: defaultdict[int, list[tuple[str, int]]] = defaultdict(list)
            for item in self._p_items:
                normalized = _normalise_text(item.get("name"))
                if normalized:
                    by_length[len(normalized)].append((normalized, int(item["id"])))
            self._p_item_detail_fuzzy_titles = {
                length: tuple(rows) for length, rows in by_length.items()
            }
        same_length = self._p_item_detail_fuzzy_titles.get(len(title), ())
        if any(expected == title for expected, _ in same_length):
            return None
        recovered = [
            expected
            for expected, _ in same_length
            if expected.endswith("+") == title.endswith("+")
            and sum(a != b for a, b in zip(expected, title, strict=True)) == 1
        ]
        if len(recovered) != 1:
            return None
        return recovered[0]

    def resolve_clicked_p_item_text(
        self,
        atoms: Sequence[Mapping[str, Any]],
        *,
        title_index: int,
        source_frames: Any = (),
        plan: str | None = None,
        candidate_p_item_ids: Sequence[int] = (),
        visual_tiebreak_p_item_ids: Sequence[int] = (),
        title_text: str | None = None,
        detail_text: str | None = None,
        allow_one_substitution: bool = False,
    ) -> PItemDetailTextResult:
        """Resolve attributed detail text without image candidate preselection.

        Production supplies same-frame atoms and source observations. The text
        adapter exists for callers that historically have no geometry; it is
        never used to replace failed geometry from a real observation.
        """

        index = self._get_p_item_detail_text_index()
        title_recovery = None
        if index is not None and allow_one_substitution:
            observed_title = title_text if not atoms else None
            if (
                atoms
                and isinstance(title_index, int)
                and not isinstance(title_index, bool)
                and 0 <= title_index < len(atoms)
            ):
                observed_title = str(atoms[title_index].get("text", ""))
            if observed_title and not index.family_ids_for_title(observed_title, plan=plan):
                recovered = self._recover_p_item_title_one_substitution(observed_title)
                if recovered and index.family_ids_for_title(recovered, plan=plan):
                    title_recovery = {"observed": observed_title, "recovered": recovered}
                    if atoms:
                        substituted_atoms = list(atoms)
                        substituted_atoms[title_index] = {
                            **atoms[title_index],
                            "text": recovered,
                        }
                        atoms = substituted_atoms
                    else:
                        title_text = recovered
        if index is None:
            result = PItemDetailTextResult(
                status="unknown",
                ids=(),
                reason="reference_unavailable",
                diagnostics={"load_error": self._p_item_detail_text_index_error},
            )
        elif not atoms and title_text is not None and detail_text is not None:
            result = index.match_text(title_text, detail_text, plan=plan)
        else:
            result = index.match(
                atoms,
                title_index=title_index,
                source_frames=source_frames,
                plan=plan,
            )
        return replace(
            result,
            diagnostics={
                **result.diagnostics,
                "visual_candidate_ids_diagnostic_only": tuple(candidate_p_item_ids),
                "visual_tiebreak_ids_diagnostic_only": tuple(visual_tiebreak_p_item_ids),
                "title_one_substitution_recovery": title_recovery,
            },
        )

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
        candidate_id_set = set(candidate_ids)
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
            visible_base_titles = {
                _normalise_text(item.get("name"))
                for item in candidates
                if not _normalise_text(item.get("name")).endswith("+")
            }
            if visible_base_titles:
                candidates.extend(
                    item
                    for item in self._p_items
                    if int(item["id"]) in candidate_id_set
                    and _normalise_text(item.get("name"))
                    in {f"{title}+" for title in visible_base_titles}
                )
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

    @staticmethod
    def _half_stamina_upgrade_hand_p_item_descriptor(
        item: Mapping[str, Any],
    ) -> tuple[int, int] | None:
        """Parse the fixed half-stamina hand-upgrade P-item DSL family.

        A base/``+`` title is deliberately ambiguous when OCR omits the final
        plus sign.  This descriptor is intentionally narrower than a generic
        effect parser: every trigger, condition, action and limit operand must
        match the version-pinned catalog family before rendered effect text can
        distinguish its two numeric states.
        """

        effects = _normalise_text(item.get("effects"))
        match = re.fullmatch(
            r"at:afterStartOfTurn\{"
            r"if:stamina<=maxStamina\*0\.5\{"
            r"concentration\+=([1-9][0-9]*);"
            r"costReduction\+=([1-9][0-9]*);"
            r"upgradeHand"
            r"\};limit:1\}",
            effects,
        )
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    def clicked_p_item_effect_detail_matches(
        self,
        detail_text: str,
        *,
        candidate_p_item_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Disambiguate one supported base/``+`` pair from complete effects.

        Title-only matching must continue to keep both states when the final
        ``+`` is absent.  A unique result is returned only for the fixed DSL
        family above and only when the same complete detail view proves its
        trigger, condition, both numeric operands, hand upgrade and once-per-
        stage footer.  Partial or unsupported details remain unknown.
        """

        candidate_ids = tuple(dict.fromkeys(candidate_p_item_ids))
        if len(candidate_ids) != 2 or any(
            isinstance(p_item_id, bool)
            or not isinstance(p_item_id, int)
            or p_item_id < 1
            for p_item_id in candidate_ids
        ):
            return ()
        candidates = tuple(
            self._p_items_by_id.get(p_item_id)
            for p_item_id in candidate_ids
        )
        if any(item is None for item in candidates):
            return ()
        typed_candidates = tuple(item for item in candidates if item is not None)
        base_items = tuple(
            item for item in typed_candidates if not bool(item.get("upgraded"))
        )
        upgraded_items = tuple(
            item for item in typed_candidates if bool(item.get("upgraded"))
        )
        if len(base_items) != 1 or len(upgraded_items) != 1:
            return ()
        base_title = _normalise_text(base_items[0].get("name"))
        upgraded_title = _normalise_text(upgraded_items[0].get("name"))
        if (
            not base_title
            or base_title.endswith("+")
            or upgraded_title != f"{base_title}+"
        ):
            return ()
        descriptors = {
            int(item["id"]): self._half_stamina_upgrade_hand_p_item_descriptor(item)
            for item in typed_candidates
        }
        if any(descriptor is None for descriptor in descriptors.values()):
            return ()
        if len(set(descriptors.values())) != len(descriptors):
            return ()

        compact = _normalise_effect_text(detail_text)
        if not all(
            anchor in compact
            for anchor in (
                "ターン開始後",
                "体力が50%以下の場合",
                "手札をすべて試験・ステージ中強化",
                "試験・ステージ内1回",
            )
        ):
            return ()
        matches = []
        for p_item_id, descriptor in descriptors.items():
            if descriptor is None:
                continue
            concentration, cost_reduction = descriptor
            if _contains_exact_integer_token(
                compact,
                "集中+",
                concentration,
            ) and _contains_exact_integer_token(
                compact,
                "消費体力削減",
                cost_reduction,
            ):
                matches.append(p_item_id)
        return tuple(matches)

    def clicked_p_item_global_detail_matches(
        self,
        detail_text: str,
        *,
        candidate_p_item_ids: Sequence[int],
        visual_tiebreak_p_item_ids: Sequence[int] = (),
        unrepresented_p_item_ids: Sequence[int] = (),
        allow_one_substitution: bool = False,
    ) -> tuple[int, ...]:
        """Resolve a clicked title across a broad catalog-drift candidate set.

        Unlike the rendered-icon candidate path, this recovery requires the
        complete normalized title to occupy one OCR line.  That lets a newer
        arena P-item absent from the fixed gallery resolve by its authoritative
        engine-data title without allowing effect prose elsewhere in the
        detail panel to masquerade as the title. ``unrepresented_p_item_ids``
        also blocks visual tie-breaking for provisional references whose
        pixels cannot yet prove a base/``+`` identity.
        """

        candidate_ids = tuple(dict.fromkeys(candidate_p_item_ids))
        if not candidate_ids or any(
            isinstance(p_item_id, bool) or not isinstance(p_item_id, int) or p_item_id < 1
            for p_item_id in candidate_ids
        ):
            raise ArenaCatalogError("detail recovery contains no valid P-item candidate IDs")
        candidate_id_set = set(candidate_ids)
        visual_tiebreak_ids = frozenset(visual_tiebreak_p_item_ids)
        visual_tiebreak_blocked_ids = frozenset(unrepresented_p_item_ids)
        for label, values in (
            ("visual tiebreak", visual_tiebreak_ids),
            ("visual tiebreak blocked", visual_tiebreak_blocked_ids),
        ):
            if any(
                isinstance(p_item_id, bool)
                or not isinstance(p_item_id, int)
                or p_item_id < 1
                for p_item_id in values
            ):
                raise ArenaCatalogError(
                    f"detail recovery contains invalid {label} P-item IDs"
                )
        lines = {
            normalized
            for line in str(detail_text or "").splitlines()
            if (normalized := _normalise_text(line))
        }
        candidates = [
            item
            for p_item_id in candidate_ids
            if (item := self._p_items_by_id.get(p_item_id)) is not None
            and (title := _normalise_text(item.get("name")))
            and title in lines
        ]
        if candidates:
            longest = max(len(_normalise_text(item.get("name"))) for item in candidates)
            candidates = [
                item
                for item in candidates
                if len(_normalise_text(item.get("name"))) == longest
            ]
            direct_candidate_ids = {int(item["id"]) for item in candidates}
            visible_base_titles = {
                _normalise_text(item.get("name"))
                for item in candidates
                if not _normalise_text(item.get("name")).endswith("+")
            }
            if visible_base_titles:
                candidates.extend(
                    item
                    for item in self._p_items
                    if int(item["id"]) in candidate_id_set
                    and _normalise_text(item.get("name"))
                    in {f"{title}+" for title in visible_base_titles}
                )
            if len(candidates) > 1 and not any(
                int(item["id"]) in visual_tiebreak_blocked_ids
                for item in candidates
            ):
                visual_matches = [
                    item
                    for item in candidates
                    if int(item["id"]) in visual_tiebreak_ids
                    and int(item["id"]) in direct_candidate_ids
                ]
                if len(visual_matches) == 1:
                    candidates = visual_matches
            return tuple(int(item["id"]) for item in candidates)
        if not allow_one_substitution:
            return ()

        exact_titles = {
            _normalise_text(item.get("name"))
            for item in self._p_items
            if _normalise_text(item.get("name"))
        }
        recovered: list[dict[str, Any]] = []
        for line in lines:
            core = line[:-1] if line.endswith("+") else line
            if len(core) < 4 or line in exact_titles:
                continue
            matches = [
                item
                for item in self._p_items
                if (title := _normalise_text(item.get("name")))
                and len(title) == len(line)
                and title.endswith("+") == line.endswith("+")
                and sum(
                    left != right
                    for left, right in zip(title, line, strict=True)
                )
                == 1
            ]
            if (
                len(matches) == 1
                and int(matches[0]["id"]) in candidate_id_set
            ):
                recovered.append(matches[0])
        recovered = list({int(item["id"]): item for item in recovered}.values())
        if len(recovered) != 1:
            return ()
        recovered_item = recovered[0]
        recovered_title = _normalise_text(recovered_item.get("name"))
        if not recovered_title.endswith("+"):
            plus_siblings = [
                item
                for p_item_id in candidate_ids
                if (item := self._p_items_by_id.get(p_item_id)) is not None
                and _normalise_text(item.get("name")) == f"{recovered_title}+"
            ]
            if plus_siblings:
                recovered.extend(plus_siblings)
        if len(recovered) > 1 and not any(
            int(item["id"]) in visual_tiebreak_blocked_ids
            for item in recovered
        ):
            visual_matches = [
                item
                for item in recovered
                if int(item["id"]) in visual_tiebreak_ids
                and item is recovered_item
            ]
            if len(visual_matches) == 1:
                recovered = visual_matches
        return tuple(int(item["id"]) for item in recovered)

    def resolve_skill_card(self, name: str, *, upgraded: bool) -> int:
        candidates = self._cards_by_name.get(
            _normalise_skill_card_title_text(name),
            (),
        )
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

    def skill_card_business_ids(self) -> tuple[int, ...]:
        """Return every skill-card business ID declared by the active catalog."""

        return tuple(sorted(self._cards_by_id))

    def skill_card_source_type(self, card_id: int) -> str:
        """Return the authoritative deck-source class for one skill card."""

        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        source_type = card.get("sourceType")
        if not isinstance(source_type, str) or not source_type.strip():
            raise ArenaCatalogError(
                f"skill-card {card_id} has no valid sourceType"
            )
        return source_type.strip()

    @staticmethod
    def normalize_skill_card_title_text(value: object) -> str:
        """Expose the catalog's sole title normalization contract to readers."""

        return _normalise_skill_card_title_text(value)

    @staticmethod
    def _skill_card_display_title(card: Mapping[str, Any]) -> str:
        """Return the normalized title rendered by the detail overlay."""

        title = _normalise_skill_card_title_text(card.get("name"))
        if card.get("upgraded") is True and not title.endswith("+"):
            return f"{title}+"
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
        """Return the exact title and globally unique bounded OCR aliases."""

        if card_id not in self._cards_by_id:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        return self._skill_card_title_aliases[card_id]

    def _build_skill_card_title_aliases(self) -> dict[int, tuple[str, ...]]:
        full_title_owners: defaultdict[str, set[int]] = defaultdict(set)
        weak_title_owners: defaultdict[str, set[int]] = defaultdict(set)
        normalized_titles: dict[int, str] = {}
        terminal_note_aliases: dict[int, str] = {}
        for card_id, card in self._cards_by_id.items():
            title = _normalise_skill_card_title_text(card.get("name"))
            if not title:
                raise ArenaCatalogError(f"skill-card {card_id} has no fixed title")
            normalized_titles[card_id] = title
            full_title_owners[title].add(card_id)
            full_title_owners[self._skill_card_display_title(card)].add(card_id)
            dropped = title[1:]
            dropped_core = dropped[:-1] if dropped.endswith("+") else dropped
            if len(dropped_core) >= 4:
                weak_title_owners[dropped].add(card_id)
                if title.startswith("一"):
                    # A leading horizontal stroke can be rendered by OCR as
                    # ASCII '-'. Keep this a complete-title alias; ordinary
                    # minus signs and exact-title rebinding stay unchanged.
                    weak_title_owners[f"-{dropped}"].add(card_id)
            # A settled detail can omit its terminal music note even when the
            # same transaction's opening frame read it. Keep every other
            # character, including the rendered upgrade mark, and require the
            # complete OCR line inside the existing visual candidate family.
            note = re.fullmatch(
                r"([^♪+]{3,})♪(\+?)", self._skill_card_display_title(card)
            )
            if note is not None:
                note_alias = "".join(note.groups())
                terminal_note_aliases[card_id] = note_alias
                weak_title_owners[note_alias].add(card_id)
        aliases: dict[int, tuple[str, ...]] = {}
        for card_id, title in normalized_titles.items():
            values = [title]
            dropped = title[1:]
            if (
                weak_title_owners.get(dropped) == {card_id}
                and dropped not in full_title_owners
            ):
                values.append(dropped)
            stroke_alias = f"-{dropped}"
            if (
                title.startswith("一")
                and weak_title_owners.get(stroke_alias) == {card_id}
                and stroke_alias not in full_title_owners
            ):
                values.append(stroke_alias)
            note_alias = terminal_note_aliases.get(card_id)
            if (
                note_alias is not None
                and weak_title_owners.get(note_alias) == {card_id}
                and note_alias not in full_title_owners
            ):
                values.append(note_alias)
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
        compact = _normalise_skill_card_title_text(detail_text)
        matches = [len(aliases[0])] if aliases[0] in compact else []
        if allow_missing_lead_alias:
            # A bounded OCR alias is deliberately narrower than the exact
            # title path: it must occupy one complete OCR line.  This prevents
            # effect text or an unrelated longer title from satisfying the
            # recovery merely because it contains the same substring.
            lines = {
                normalized
                for line in str(detail_text or "").splitlines()
                if (normalized := _normalise_skill_card_title_text(line))
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

        compact = _normalise_skill_card_title_text(detail_text)
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
            names = {
                _normalise_skill_card_title_text(card.get("name"))
                for card in candidates
            }
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

    def confirm_clicked_skill_card_by_exact_title(self, detail_text: str) -> int:
        """Resolve one catalog card from one exact clicked-detail title line.

        The fixed visual gallery can lag behind the active RIS catalog.  In
        that case a newly added card may be projected onto an older visual
        family even though its clicked detail exposes an authoritative title.
        Rebinding is allowed only for one authoritative normalized display-title
        row selected by the caller and only when that row resolves to exactly
        one catalog ID. Embedded substrings, lossy aliases, multiple OCR rows,
        and duplicate display titles remain fail-closed.

        Some upstream rows keep the same raw ``name`` for the normal and
        upgraded IDs.  The detail overlay renders ``+`` for the upgraded row,
        so the active catalog's ``upgraded`` flag is part of the display-title
        key instead of relying on a fixed list of known card IDs.
        """

        lines = tuple(
            normalized
            for line in str(detail_text or "").splitlines()
            if (normalized := _normalise_skill_card_title_text(line))
        )
        if len(lines) != 1:
            raise ArenaCatalogError(
                "clicked skill-card exact title requires one authoritative OCR line"
            )
        (title_line,) = lines
        candidates = self._cards_by_display_title.get(title_line, ())
        if len(candidates) != 1:
            raise ArenaCatalogError(
                "clicked skill-card exact title must resolve exactly once "
                "across the active catalog"
            )
        card_id = candidates[0].get("id")
        if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id < 1:
            raise ArenaCatalogError("resolved skill-card ID is invalid")
        return card_id

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
        compact = _normalise_skill_card_title_text(detail_text)
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

        compact = _normalise_skill_card_title_text(detail_text)
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
            names = {
                _normalise_skill_card_title_text(card.get("name"))
                for card in candidates
            }
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

    def maximum_customization_count(self, card_id: int) -> int:
        """Return the largest badge count representable for one card."""

        return sum(
            int(self._customizations[customization_id]["max"])
            for customization_id in self.available_customization_ids(card_id)
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
            if self._upgrade_hand_detail_signature(card, "") is None:
                return "badge_constraint_only"
            return "detail_positive" if level > 0 else "no_direct_zero_evidence"
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
        if self._named_action_threshold_descriptor(card, customization_id) is not None:
            return True
        if (
            "g.scoreByGoodImpressionTurns+=" in str(definition.get("effects", ""))
            and "score+=goodImpressionTurns*" not in str(card.get("actions", ""))
        ):
            return self._good_impression_score_coefficient(card) is not None
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
        self._validate_added_numeric_effects(card_id, detail_text)

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
        compact = _normalise_effect_text(detail_text)
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
                        resolved=normalized_resolved,
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
        card-face generic-cost value. A customization indistinguishable from
        an identical base action (for example, another ``upgradeHand`` when
        the base already contains it) remains fail-closed.
        """
        self._validate_added_numeric_effects(card_id, full_detail_text)
        self._validate_added_numeric_effects(card_id, effect_roi_text)

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
        full_compact = _normalise_effect_text(full_detail_text)
        roi_compact = _normalise_effect_text(effect_roi_text)
        full_title_compact = _normalise_skill_card_title_text(full_detail_text)
        roi_title_compact = _normalise_skill_card_title_text(effect_roi_text)
        title = _normalise_skill_card_title_text(card.get("name"))
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
        if not title or title not in full_title_compact:
            raise ArenaCatalogError(
                f"card {card_id} full detail OCR lacks the exact title anchor"
            )
        roi_upgraded_title = (
            title.endswith("+")
            and title in full_title_compact
            and title[:-1] in roi_title_compact
        )
        if (
            selected_structural_omissions
            and title not in roi_title_compact
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
                    resolved=normalized_resolved,
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

    @staticmethod
    def _literal_score_good_condition_annotation(
        card: Mapping[str, Any],
        base_score: int,
    ) -> str:
        """Bind a rendered annotation to its adjacent, sole literal score action."""

        actions = _normalise_text(card.get("actions"))
        if len(re.findall(r"(?<![.A-Za-z0-9_])score\+=", actions)) != 1:
            return ""
        multiplier = re.search(
            r"(?:^|;)goodConditionTurnsMultiplier=([1-9][0-9]*(?:\.[0-9]+)?);"
            rf"score\+={base_score}(?=;|$)",
            actions,
        )
        if multiplier is None:
            return ""
        return f"(好調効果を{multiplier.group(1)}倍適用)"

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
        selected_delta = self._selected_direct_growth_increment(resolved, "score")
        effective_scores = tuple(
            value + selected_delta
            for value in _all_literal_integer_increments(actions, "score")
        )
        expected_value = int(score.group(1))
        base_repetitions = effective_scores.count(expected_value)

        # The active engine seeds ``scoreTimes`` on the card, applies its extra
        # hits to the first score action, then consumes it. Identical rendered
        # score actions therefore collapse to ``base count + D``: two existing
        # ``score+=N`` actions plus ``g.scoreTimes+=1`` render as ``3回``.
        score_times_increment = self._selected_direct_growth_increment(
            resolved,
            "scoreTimes",
        )
        repetitions = base_repetitions + (
            score_times_increment
            if effective_scores and expected_value == effective_scores[0]
            else 0
        )
        if repetitions < 2:
            return False
        if (
            re.search(
                rf"(?:スコア)?\+{expected_value}[（(]?{repetitions}回(?!まで)",
                compact,
            )
            is not None
        ):
            return True
        if "好調効果を" not in compact:
            return False
        annotation = self._literal_score_good_condition_annotation(
            card,
            expected_value - selected_delta,
        )
        return bool(
            annotation
            and re.search(
                rf"(?:スコア)?\+{expected_value}{re.escape(annotation)}"
                rf"[（(]?{repetitions}回(?!まで)",
                compact,
            )
        )

    def _selected_direct_growth_increment(
        self,
        resolved: Mapping[str, int],
        field: str,
    ) -> int:
        return sum(
            delta
            for customization_text, level in resolved.items()
            if (
                delta := _direct_prestage_target_this_increment(
                    self._customizations[int(customization_text)].get("effects"),
                    field,
                    int(level),
                )
            )
            is not None
        )

    def _positive_signature_is_visible(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        compact: str,
        level: int,
        *,
        resolved: Mapping[str, int] | None = None,
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
        customization_effects = str(definition.get("effects", ""))
        if re.search(r"g\.scoreTimes\+=([1-9][0-9]*)", customization_effects):
            if self._dynamic_full_power_score_times_descriptor(
                card,
                customization_id,
            ) is not None:
                return self._dynamic_full_power_score_times_level_is_visible(
                    card,
                    customization_id,
                    compact,
                    level,
                )
            base_scores = _all_literal_integer_increments(
                card.get("actions"),
                "score",
            )
            if "@grow" not in customization_effects:
                selected = (
                    dict(resolved)
                    if resolved is not None
                    else {str(customization_id): int(level)}
                )
                selected_score_delta = self._selected_direct_growth_increment(
                    selected,
                    "score",
                )
                if not base_scores:
                    return False
                return self._detail_counted_score_anchor_present(
                    card,
                    selected,
                    f"スコア+{base_scores[0] + selected_score_delta}",
                    compact,
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

    def _view_merge_positive_signature_is_visible(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        compact: str,
        level: int,
    ) -> bool:
        """Recognize positive rows without changing the legacy fallback surface."""

        definition = self._customizations[customization_id]
        applied_score = self._good_condition_applied_score_descriptor(
            card,
            customization_id,
        )
        if applied_score is not None:
            base_score, multiplier, coefficient = applied_score
            return self._good_condition_applied_score_level_is_visible(
                compact,
                level,
                base_score=base_score,
                multiplier=multiplier,
                coefficient=coefficient,
            )
        direct_score_times = _direct_prestage_target_this_increment(
            definition.get("effects"),
            "scoreTimes",
            level,
        )
        if direct_score_times is not None:
            if self._dynamic_full_power_score_times_descriptor(
                card,
                customization_id,
            ) is not None:
                return self._dynamic_full_power_score_times_level_is_visible(
                    card,
                    customization_id,
                    compact,
                    level,
                )
            base_scores = _all_literal_integer_increments(
                card.get("actions"),
                "score",
            )
            if not base_scores:
                return False
            card_id = int(card["id"])
            available = self.available_customization_ids(card_id)
            maximum_total = sum(
                int(self._customizations[item]["max"]) for item in available
            )
            return any(
                self._detail_counted_score_anchor_present(
                    card,
                    group,
                    f"スコア+{base_scores[0] + self._selected_direct_growth_increment(group, 'score')}",
                    compact,
                )
                for expected_count in range(1, maximum_total + 1)
                for group in self.legal_customization_groups(
                    card_id,
                    expected_count=expected_count,
                )
                if int(group.get(str(customization_id), 0)) == level
            )
        if _normalise_text(definition.get("actions")) == "upgradeHand":
            return self._upgrade_hand_detail_signature(card, compact) is True
        if self._pure_generic_cost_delta(customization_id) is not None:
            return False
        for field in (
            "goodConditionTurns",
            "goodImpressionTurns",
            "perfectConditionTurns",
            "concentration",
            "genki",
            "motivation",
            "fullPowerCharge",
            "halfCostTurns",
            "turnsRemaining",
            "score",
        ):
            added_value = _first_integer_increment(definition.get("actions"), field)
            if (
                added_value is not None
                and _first_integer_increment(card.get("actions"), field)
                == added_value
            ):
                return False
        card_use_delta = re.fullmatch(
            r"cardUsesRemaining\+=([0-9]+)",
            _normalise_text(definition.get("actions")),
        )
        if card_use_delta and re.search(
            rf"(?:^|;)cardUsesRemaining\+={card_use_delta.group(1)}(?:;|$)",
            _normalise_text(card.get("actions")),
        ):
            return False
        return self._positive_signature_is_visible(
            card,
            customization_id,
            compact,
            level,
        )

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
        named_threshold = self._named_action_threshold_descriptor(card, customization_id)
        if named_threshold is not None:
            field, base_threshold, _ = named_threshold
            return self._named_action_threshold_is_visible(compact, field, base_threshold)
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
            if (
                action_field == "goodImpressionTurns"
                and "score+=goodImpressionTurns*" not in str(card.get("actions", ""))
            ):
                coefficient = self._good_impression_score_coefficient(card)
                return coefficient is not None and self._good_impression_score_is_visible(
                    compact, int(round(coefficient * 100)),
                )
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

    def _covered_target_growth_zero_signature_is_visible(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        *,
        resolved: Mapping[str, int],
        combined_detail_text: str,
        full_detail_text: str | None,
        effect_roi_text: str | None,
        effect_roi_title_bound: bool,
        effect_roi_observation_count: int,
    ) -> bool:
        """Prove one omitted target-growth row from provenance-bound OCR views.

        ``成長追加`` appends one visible ``target:hand/held/all`` score-growth
        row at every positive level.  Its level-zero UI has no replacement row,
        so one OCR pass cannot distinguish a real zero from an omitted positive
        row.  A zero is admitted here only when the complete full-frame detail
        and a uniquely title-bound effect ROI are provided separately.  The ROI
        must carry at least two non-empty independent observations (native plus
        an independently rerun enlarged or contrast-enhanced OCR view).  Every
        observation must cover all invariant actions of the resolved card and
        show neither a known positive value nor even the target-growth marker.
        The generic-cost runtime still confirms the same bounded pair on one
        fresh settled frame before applying its side-specific conservative
        policy.

        The DSL gate is intentionally exact.  Extra actions, another patch
        carrier, non-positive values, or mixed targets remain unsupported.
        """

        definition = self._customizations[customization_id]
        if any(
            _normalise_text(definition.get(field))
            for field in ("conditions", "cost", "effects")
        ):
            return False
        if definition.get("limit") not in (None, ""):
            return False
        if definition.get("forceInitialHand") not in (None, "", False):
            return False

        targets: set[str] = set()
        increments: set[int] = set()
        maximum = int(definition["max"])
        for level in range(1, maximum + 1):
            selected = _normalise_text(
                _selected_level_patch(definition.get("actions"), level)
            )
            match = re.fullmatch(
                r"target:(all|hand|held)\{g\.score\+=([1-9][0-9]*);?\};?",
                selected,
            )
            if match is None:
                return False
            targets.add(match.group(1))
            increments.add(int(match.group(2)))
        if len(targets) != 1 or len(increments) != maximum:
            return False

        if (
            not effect_roi_title_bound
            or not isinstance(full_detail_text, str)
            or not isinstance(effect_roi_text, str)
            or not full_detail_text.strip()
            or not effect_roi_text.strip()
            or _OCR_EFFECT_VIEW_BOUNDARY in full_detail_text
            or type(effect_roi_observation_count) is not int
            or effect_roi_observation_count < 2
        ):
            return False
        roi_views = tuple(
            dict.fromkeys(
                view.strip()
                for view in effect_roi_text.split(_OCR_EFFECT_VIEW_BOUNDARY)
                if view.strip()
            )
        )
        combined_views = tuple(
            dict.fromkeys(
                view.strip()
                for view in combined_detail_text.split(_OCR_EFFECT_VIEW_BOUNDARY)
                if view.strip()
            )
        )
        coverage_views = tuple(
            dict.fromkeys((full_detail_text.strip(), *roi_views))
        )
        # Provenance must describe the exact text being resolved.  This keeps a
        # stale full/ROI tuple from authenticating a later detail frame.  The
        # observation count is carried separately because two independent OCR
        # passes can legitimately produce byte-identical text and be deduped by
        # the compatibility merger.
        if combined_views != coverage_views:
            return False
        title = _normalise_skill_card_title_text(card.get("name"))
        if not title or title not in _normalise_skill_card_title_text(
            full_detail_text
        ):
            return False
        anchors = self._detail_coverage_anchors(card, resolved)
        if not anchors:
            return False
        matcher = self._effective_detail_matcher(card, customization_id)
        if matcher is None:
            return False
        for view in (full_detail_text, *roi_views):
            compact = _normalise_effect_text(view)
            # Target words (手札／保留／すべて) are more fragile than the
            # shared suffix.  Losing only that word must not turn a real
            # positive row into a hand/held zero, so any growth marker blocks
            # the covered-absence certificate before target-specific matching.
            if "コア値増加+" in compact:
                return False
            if not matcher(compact, 0):
                return False
            if any(matcher(compact, level) for level in range(1, maximum + 1)):
                return False
            if any(
                not self._detail_coverage_anchor_present(
                    card,
                    resolved,
                    anchor,
                    compact,
                )
                for anchor in anchors
            ):
                return False
        return True

    def _view_merge_zero_signature_is_visible(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        compact: str,
    ) -> bool:
        """Recognize explicit zero rows only for independent OCR-view checks."""

        if self._zero_signature_is_visible(card, customization_id, compact):
            return True
        definition = self._customizations[customization_id]
        direct_score_times = _direct_prestage_target_this_increment(
            definition.get("effects"),
            "scoreTimes",
            1,
        )
        if direct_score_times is not None:
            base_scores = _all_literal_integer_increments(
                card.get("actions"),
                "score",
            )
            if not base_scores:
                return False
            card_id = int(card["id"])
            available = self.available_customization_ids(card_id)
            maximum_total = sum(
                int(self._customizations[item]["max"]) for item in available
            )
            visible_zero_scores: set[int] = set()
            for expected_count in range(maximum_total + 1):
                for group in self.legal_customization_groups(
                    card_id,
                    expected_count=expected_count,
                ):
                    if int(group.get(str(customization_id), 0)) != 0:
                        continue
                    selected_score_delta = self._selected_direct_growth_increment(
                        group,
                        "score",
                    )
                    effective_scores = tuple(
                        score + selected_score_delta for score in base_scores
                    )
                    if (
                        effective_scores
                        and effective_scores.count(effective_scores[0]) == 1
                    ):
                        visible_zero_scores.add(effective_scores[0])
            annotation = self._literal_score_good_condition_annotation(
                card,
                base_scores[0],
            ) if "好調効果を" in compact else ""
            annotation_gap = rf"(?:{re.escape(annotation)})?" if annotation else ""
            return any(
                re.search(
                    rf"スコア\+{score}"
                    rf"(?![0-9]|[.,．，。][0-9]|{annotation_gap}[（(]?[0-9]+回)",
                    compact,
                )
                is not None
                for score in visible_zero_scores
            )

        condition_match = re.fullmatch(
            r"@usable if:(goodConditionTurns|goodImpressionTurns|genki|preservationTimes)>=([0-9]+)",
            str(definition.get("conditions", "")),
        )
        if condition_match:
            field = condition_match.group(1)
            base_match = re.search(
                rf"@usable if:{re.escape(field)}>=([0-9]+)",
                str(card.get("conditions", "")),
            )
            if base_match is None:
                return False
            threshold = int(base_match.group(1))
            labels = {
                "goodConditionTurns": f"好調が{threshold}ターン以上",
                "goodImpressionTurns": f"好印象が{threshold}以上",
                "preservationTimes": f"温存になった回数が{threshold}回以上",
                "genki": f"元気が{threshold}以上",
            }
            return labels[field] in compact

        trigger_condition_match = re.fullmatch(
            r"@trigger if:(goodImpressionTurns)>=([0-9]+)",
            str(definition.get("actions", "")),
        )
        if trigger_condition_match:
            field = trigger_condition_match.group(1)
            base_match = re.search(
                rf"@trigger if:{re.escape(field)}>=([0-9]+)",
                str(card.get("actions", "")),
            )
            return (
                base_match is not None
                and f"好印象が{int(base_match.group(1))}以上" in compact
            )

        coefficient_levels = tuple(
            re.search(
                r"score\+=genki\*([0-9]+(?:\.[0-9]+)?)",
                _selected_level_patch(definition.get("actions"), level),
            )
            for level in range(1, int(definition["max"]) + 1)
        )
        if any(match is not None for match in coefficient_levels):
            base_match = re.search(
                r"score\+=genki\*([0-9]+(?:\.[0-9]+)?)",
                str(card.get("actions", "")),
            )
            return (
                base_match is not None
                and f"元気の{int(round(float(base_match.group(1)) * 100))}%分スコア"
                in compact
            )

        if "g.stanceLevel+=1" in str(definition.get("effects", "")):
            base_stance = re.search(
                r"setStance\((strength|preservation)\)",
                str(card.get("actions", "")),
            )
            if base_stance is None:
                return False
            label = "強気" if base_stance.group(1) == "strength" else "温存"
            return f"{label}に変更" in compact
        return False

    def observe_customizations_from_clicked_text(
        self,
        card_id: int,
        detail_text: str,
    ) -> tuple[CustomizationObservation, ...]:
        """Extract only card-eligible customization rows from a clicked detail."""
        self._validate_added_numeric_effects(card_id, detail_text)

        import re

        compact = _normalise_effect_text(detail_text)
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
        self._validate_added_numeric_effects(card_id, detail_text)

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

        compact = _normalise_effect_text(detail_text)
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

    def merge_compatible_effect_detail_views(
        self,
        card_id: int,
        detail_texts: Sequence[str],
    ) -> str:
        """Merge independent OCR views without inventing or conflicting evidence.

        A native-resolution view and an enlarged view can have complementary
        omissions.  Missing positive evidence is therefore allowed, but two
        views that visibly identify different level sets for the same
        catalog customization must fail closed.  Non-empty level sets must be
        exactly equal: accepting only an overlap or a union would let a later
        badge count select a level that one OCR view did not support.
        """

        if isinstance(detail_texts, (str, bytes)):
            raise ArenaCatalogError("effect detail views must be a sequence of strings")
        views: list[str] = []
        for view_index, value in enumerate(detail_texts):
            if not isinstance(value, str):
                raise ArenaCatalogError(
                    f"effect detail OCR view {view_index} is not a string"
                )
            self._validate_added_numeric_effects(card_id, value)
            # A title-bound ROI may already contain independently certified
            # native and enlarged observations.  Flatten nested merges so the
            # outer full-detail/ROI gate still reasons over every source view.
            for atomic_view in value.split(_OCR_EFFECT_VIEW_BOUNDARY):
                atomic_view = atomic_view.strip()
                if atomic_view and atomic_view not in views:
                    views.append(atomic_view)
        if not views:
            raise ArenaCatalogError("effect detail OCR views are all empty")

        card = self._cards_by_id.get(card_id)
        if card is None:
            raise ArenaCatalogError(
                f"skill-card ID is absent from the bundled catalog: {card_id}"
            )
        available = tuple(
            int(value)
            for value in str(card.get("availableCustomizations", "")).split(",")
            if value
        )
        for customization_id in available:
            matcher = self._effective_detail_matcher(card, customization_id)
            if matcher is None:
                continue
            maximum = int(self._customizations[customization_id]["max"])
            evidence_by_view: list[tuple[int, frozenset[int]]] = []
            for view_index, view in enumerate(views):
                visible_levels = self._visible_effect_levels(
                    card,
                    customization_id,
                    matcher,
                    view,
                )
                if visible_levels:
                    evidence_by_view.append((view_index, visible_levels))
            if evidence_by_view and any(
                levels != evidence_by_view[0][1]
                for _, levels in evidence_by_view[1:]
            ):
                raise ArenaCatalogError(
                    "effect detail OCR views disagree for "
                    f"card {card_id} customization {customization_id}: "
                    f"{evidence_by_view!r}"
                )

        merged = f"\n{_OCR_EFFECT_VIEW_BOUNDARY}\n".join(views)
        for customization_id in available:
            matcher = self._effective_detail_matcher(card, customization_id)
            if matcher is None:
                continue
            maximum = int(self._customizations[customization_id]["max"])
            source_levels = tuple(
                frozenset(
                    level
                    for level in range(maximum + 1)
                    if matcher(_normalise_effect_text(view), level)
                )
                for view in views
            )
            merged_compact = _normalise_effect_text(merged)
            merged_levels = frozenset(
                level
                for level in range(maximum + 1)
                if matcher(merged_compact, level)
            )
            source_union = frozenset().union(*source_levels)
            manufactured_levels = merged_levels - source_union
            if manufactured_levels:
                raise ArenaCatalogError(
                    "effect detail OCR views manufacture cross-view evidence for "
                    f"card {card_id} customization {customization_id}: "
                    f"source_levels={source_levels!r}; merged_levels={merged_levels!r}"
                )
        return merged

    def _visible_effect_levels(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        matcher: Any,
        detail_text: str,
    ) -> frozenset[int]:
        """Return only independently visible zero-or-positive level evidence."""

        compact = _normalise_effect_text(detail_text)
        maximum = int(self._customizations[customization_id]["max"])
        # Zero evidence is recognized from its own rendered signature, not
        # from matcher(..., 0).  Several legacy matchers express zero as
        # ``base present AND positive absent``; using them here would erase an
        # explicit base row whenever the same OCR view also contained a
        # contradictory positive row.
        zero_visible = self._view_merge_zero_signature_is_visible(
            card,
            customization_id,
            compact,
        )
        positive_visible = frozenset(
            level
            for level in range(1, maximum + 1)
            if matcher(compact, level)
            and self._view_merge_positive_signature_is_visible(
                card,
                customization_id,
                compact,
                level,
            )
        )
        levels = frozenset(((0,) if zero_visible else ())) | positive_visible
        if len(levels) > 1:
            raise ArenaCatalogError(
                "one effect detail OCR view contains mutually exclusive levels for "
                f"card {card['id']} customization {customization_id}: {sorted(levels)!r}"
            )
        return levels

    def _resolve_failed_detail_unique_customizations(
        self,
        card_id: int,
        detail_text: str,
    ) -> dict[str, int]:
        """Strict error-only solve, evaluating each visible level exactly once.

        Unlike badge-assisted positive completion, every available effect must
        match the same text. There must be one positive group across all
        totals, including groups that share a total. Nothing is cached across
        observations and a failed solve never starts a badge-count search.
        """
        self._validate_added_numeric_effects(card_id, detail_text)
        available = self.available_customization_ids(card_id)
        if not available:
            raise ArenaCatalogError(f"card {card_id} has no positive detail group")
        card = self._cards_by_id[card_id]
        compact = _normalise_effect_text(detail_text)
        levels = []
        for customization_id in available:
            matcher = self._effective_detail_matcher(card, customization_id)
            if matcher is None:
                raise ArenaCatalogError(
                    f"card {card_id} has unsupported final-effect signature {customization_id}"
                )
            matched = tuple(
                count for count in range(int(self._customizations[customization_id]["max"]) + 1)
                if matcher(compact, count)
            )
            if not matched:
                raise ArenaCatalogError(f"card {card_id} has an unresolved visible effect")
            levels.append(matched)
        resolved = None
        for counts in itertools.product(*levels):
            if not any(counts):
                continue
            if resolved is not None:
                raise ArenaCatalogError(f"card {card_id} has multiple strict positive detail groups")
            resolved = {
                str(customization_id): count
                for customization_id, count in zip(available, counts, strict=True)
                if count
            }
        if resolved is None:
            raise ArenaCatalogError(f"card {card_id} has no strict positive detail group")
        return resolved

    def resolve_effective_customizations_without_badge_count(
        self,
        card_id: int,
        detail_text: str,
    ) -> dict[str, int]:
        """Infer one positive combination and its total from final-effect text."""
        self._validate_added_numeric_effects(card_id, detail_text)

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
        self._validate_added_numeric_effects(card_id, detail_text)

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
        self._validate_added_numeric_effects(card_id, detail_text)

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

    def admissible_clicked_customization_counts(
        self,
        card_id: int,
        detail_text: str,
        *,
        generic_cost_frame_values: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        """Return every positive badge count still compatible with detail truth.

        A count belongs to this domain only when resolving the same clicked
        detail under that observed count produces a state whose total still
        equals the observation.  A detail-unique state returned under a
        contrary observation is therefore excluded instead of silently
        widening the later glyph classifier to every integer below the static
        maximum.
        """
        self._validate_added_numeric_effects(card_id, detail_text)

        maximum_total = self.maximum_customization_count(card_id)
        admissible: list[int] = []
        for observed_count in range(1, maximum_total + 1):
            try:
                resolution = self.resolve_clicked_customizations(
                    card_id,
                    detail_text,
                    observed_badge_count=observed_count,
                    generic_cost_frame_values=generic_cost_frame_values,
                )
            except ArenaCatalogError:
                continue
            if (
                resolution.badge_count_match
                and resolution.resolved_count == observed_count
            ):
                admissible.append(observed_count)
        return tuple(admissible)

    def resolve_effective_customizations_unconstrained(
        self,
        card_id: int,
        detail_text: str,
    ) -> dict[str, int]:
        """Resolve zero or one unique positive combination without badge input."""
        self._validate_added_numeric_effects(card_id, detail_text)

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
        full_detail_text: str | None = None,
        effect_roi_text: str | None = None,
        effect_roi_title_bound: bool = False,
        effect_roi_observation_count: int = 0,
    ) -> GenericCostAmbiguityResolution:
        """Prove that one unreadable card-face cost is the only open bit.

        The result is deliberately a pair rather than an inferred truth.  It
        is valid only when the fixed catalog, settled detail text and every
        non-cost customization leave exactly two legal states.  Those states
        must differ solely by one single-level ``g.cost+=`` customization.
        """
        self._validate_added_numeric_effects(card_id, detail_text)
        self._validate_added_numeric_effects(card_id, full_detail_text or "")
        self._validate_added_numeric_effects(card_id, effect_roi_text or "")

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
        compact = _normalise_effect_text(detail_text)
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

        # Missing non-cost rows are not generally reclassified as zero.  Every
        # positive level must have a visible positive signature.  A zero level
        # needs either an explicitly rendered baseline or the narrow covered
        # optional-row proof above.  This keeps the bounded assumption strictly
        # on the generic-cost bit.
        representative = by_generic_level[0]
        for customization_id in available:
            if customization_id == generic_id:
                continue
            level = int(representative.get(str(customization_id), 0))
            if level > 0:
                visible = self._positive_signature_is_visible(
                    card,
                    customization_id,
                    compact,
                    level,
                    resolved=representative,
                )
            else:
                visible = self._zero_signature_is_visible(
                    card,
                    customization_id,
                    compact,
                ) or self._covered_target_growth_zero_signature_is_visible(
                    card,
                    customization_id,
                    resolved=representative,
                    combined_detail_text=detail_text,
                    full_detail_text=full_detail_text,
                    effect_roi_text=effect_roi_text,
                    effect_roi_title_bound=effect_roi_title_bound,
                    effect_roi_observation_count=(
                        effect_roi_observation_count
                    ),
                )
            if not visible:
                raise ArenaCatalogError(
                    f"card {card_id} non-cost customization {customization_id} "
                    f"level {level} lacks an explicit detail signature or a "
                    "covered optional-row zero proof"
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

    def _move_random_to_top_descriptor(
        self,
        card: Mapping[str, Any],
        customization_id: int,
    ) -> tuple[str, int] | None:
        definition = self._customizations.get(customization_id)
        if definition is None:
            raise ArenaCatalogError(
                f"customization {customization_id} is absent from the bundled catalog"
            )
        action = re.fullmatch(
            r"moveRandomToTopOfDeck\[(SSR|SR|T|N|R|L)&"
            r"\((deck|discarded)\|(deck|discarded)\)\]"
            r"\(([1-9][0-9]*)\);?",
            _normalise_text(definition.get("actions")),
        )
        if action is None:
            return None
        rarity, first_source, second_source, card_count_text = action.groups()
        extra_fields = tuple(
            field
            for field in (
                "conditions",
                "cost",
                "effects",
                "limit",
                "forceInitialHand",
            )
            if (value := definition.get(field)) not in (None, "", False)
        )
        maximum = definition.get("max")
        if (
            definition.get("type") != "effect"
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum != 1
            or {first_source, second_source} != {"deck", "discarded"}
            or extra_fields
            or "moveRandomToTopOfDeck[" in _normalise_text(card.get("actions"))
        ):
            return None
        return rarity, int(card_count_text)

    def _good_condition_applied_score_descriptor(
        self,
        card: Mapping[str, Any],
        customization_id: int,
    ) -> tuple[int, int, int] | None:
        """Describe one exact ``@base do`` good-condition score replacement.

        This customization family keeps the base score and proportional
        good-condition score, but applies a visible multiplier annotation to
        the base score row.  Bind every operand in both catalog actions so the
        annotation can be positive evidence without relying on a card ID or
        on the absence of a sibling row.
        """

        definition = self._customizations.get(customization_id)
        if definition is None:
            raise ArenaCatalogError(
                f"customization {customization_id} is absent from the bundled catalog"
            )
        base = re.fullmatch(
            r"@basescore\+=([1-9][0-9]*);"
            r"score\+=goodConditionTurns\*([1-9][0-9]*);?",
            _normalise_text(card.get("actions")),
        )
        replacement = re.fullmatch(
            r"@basedo\{goodConditionTurnsMultiplier=([2-9][0-9]*);"
            r"score\+=([1-9][0-9]*)\};"
            r"do\{score\+=goodConditionTurns\*([1-9][0-9]*)\};?",
            _normalise_text(definition.get("actions")),
        )
        extra_fields = tuple(
            field
            for field in (
                "conditions",
                "cost",
                "effects",
                "limit",
                "forceInitialHand",
            )
            if (value := definition.get(field)) not in (None, "", False)
        )
        maximum = definition.get("max")
        if (
            definition.get("type") != "effect"
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum != 1
            or base is None
            or replacement is None
            or extra_fields
        ):
            return None
        base_score, base_coefficient = (int(value) for value in base.groups())
        multiplier, replaced_score, replaced_coefficient = (
            int(value) for value in replacement.groups()
        )
        if (
            replaced_score != base_score
            or replaced_coefficient != base_coefficient
        ):
            return None
        return base_score, multiplier, base_coefficient

    @staticmethod
    def _good_condition_applied_score_level_is_visible(
        compact: str,
        level: int,
        *,
        base_score: int,
        multiplier: int,
        coefficient: int,
    ) -> bool:
        """Match one complete multiplier-annotated score view.

        Independent OCR views may omit the parenthetical annotation. One
        complete positive view therefore overrides a plain view, while the
        score row and its proportional companion must still occur together
        inside the same atomic view. Operands split across the protected view
        boundary never manufacture positive evidence.
        """

        if level not in (0, 1):
            return False
        positive_row = re.compile(
            rf"スコア\+{base_score}"
            rf"\(好調効果を{multiplier}倍適用\)"
        )
        annotation_prefix = f"スコア+{base_score}(好調効果を"
        proportional_row = f"好調の{coefficient * 100}%分スコア上昇"
        complete_base_view_seen = False
        positive_view_seen = False
        conflicting_annotation_seen = False
        for atomic_view in compact.split(_OCR_EFFECT_VIEW_BOUNDARY):
            base_visible = _contains_exact_integer_token(
                atomic_view,
                "スコア+",
                base_score,
            )
            complete_base_view_seen = complete_base_view_seen or (
                base_visible and proportional_row in atomic_view
            )
            positive_view_seen = positive_view_seen or (
                base_visible
                and proportional_row in atomic_view
                and positive_row.search(atomic_view) is not None
            )
            conflicting_annotation_seen = conflicting_annotation_seen or (
                annotation_prefix in atomic_view
                and positive_row.search(atomic_view) is None
            )
        if level == 1:
            return positive_view_seen
        return (
            complete_base_view_seen
            and not positive_view_seen
            and not conflicting_annotation_seen
        )

    def _dynamic_full_power_score_times_descriptor(
        self,
        card: Mapping[str, Any],
        customization_id: int,
    ) -> tuple[int, int, int] | None:
        """Describe one strictly rendered dynamic-score repetition patch.

        The contest UI renders ``score+=B+cumulativeFullPowerCharge*M`` as
        one score row whose operands remain visible, then appends ``(N回)``
        when a direct prestage ``scoreTimes`` patch repeats that row. Keep
        this family narrower than the general score grammar so an unrelated
        dynamic expression can never inherit the carrier.
        """

        definition = self._customizations.get(customization_id)
        if definition is None:
            raise ArenaCatalogError(
                f"customization {customization_id} is absent from the bundled catalog"
            )
        increment = _direct_prestage_target_this_increment(
            definition.get("effects"),
            "scoreTimes",
            1,
        )
        actions = _normalise_text(card.get("actions"))
        score_actions = tuple(
            re.finditer(r"(?<![.A-Za-z])score\+=", actions)
        )
        dynamic = re.search(
            r"(?:^|;)score\+=([1-9][0-9]*)"
            r"\+cumulativeFullPowerCharge\*([0-9]+(?:\.[0-9]+)?)"
            r"(?=;|$)",
            actions,
        )
        extra_fields = tuple(
            field
            for field in (
                "conditions",
                "cost",
                "actions",
                "limit",
                "forceInitialHand",
            )
            if (value := definition.get(field)) not in (None, "", False)
        )
        maximum = definition.get("max")
        if (
            definition.get("type") != "score"
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum != 1
            or increment is None
            or len(score_actions) != 1
            or dynamic is None
            or extra_fields
        ):
            return None

        multiplier_text = dynamic.group(2)
        whole, separator, fraction = multiplier_text.partition(".")
        scale = 10 ** len(fraction) if separator else 1
        numerator = int(whole) * scale + (int(fraction) if fraction else 0)
        scaled_percent = numerator * 100
        if numerator <= 0 or scaled_percent % scale:
            return None
        return int(dynamic.group(1)), scaled_percent // scale, increment

    @staticmethod
    def _visible_dynamic_full_power_score_repetitions(
        compact: str,
        *,
        base_score: int,
        full_power_percent: int,
    ) -> int | None:
        """Read the repetition count only from one complete atomic UI row."""

        score_anchor = (
            rf"スコア\+{base_score}(?![0-9]|[.,．，。][0-9])"
        )
        canonical = (
            score_anchor
            + rf"\(累積全力値の(?:[＊*×])?{full_power_percent}%分"
            + r"(?:、|,|，)スコア上昇量増加\)"
        )
        # Vertical OCR orders boxes by their top edge. In the live panel the
        # right-side ``全力値の`` fragment can be one pixel higher than the
        # left-side ``スコア+B（累積`` fragment, yielding this deterministic
        # row-local permutation. Keep the alternative contiguous and bind the
        # same complete operands; do not accept freely scattered anchors.
        vertical_ocr = (
            "全力値の"
            + score_anchor
            + rf"\(累積(?:[＊*×])?{full_power_percent}%分"
            + r"(?:、|,|，)スコア上昇量増加\)"
        )
        row = re.compile(rf"(?:{canonical}|{vertical_ocr})")
        repetitions: list[int] = []
        for atomic_view in compact.split(_OCR_EFFECT_VIEW_BOUNDARY):
            matches = tuple(row.finditer(atomic_view))
            if len(matches) > 1:
                # Multiple carriers inside one OCR observation are ambiguous.
                # The same carrier repeated once per independently certified
                # view is not: the boundary must remain semantic here instead
                # of turning compatible full/ROI observations into duplicates.
                return None
            if not matches:
                continue
            tail = atomic_view[matches[0].end() :]
            if not tail.startswith("("):
                repetitions.append(1)
                continue
            repetition = re.match(r"\(([1-9][0-9]*)回(まで)?\)", tail)
            if repetition is None:
                # One enlarged title-bound ROI can splice a clipped
                # background row into the repetition suffix (for example
                # ``(2-ジ3回)``).  That atomic OCR view proves no count; it
                # must not erase a complete, agreeing count from another
                # independently certified view.  With no complete view the
                # method still returns ``None`` below.
                continue
            if repetition.group(2) is not None:
                return None
            repetitions.append(int(repetition.group(1)))
        if not repetitions or len(set(repetitions)) != 1:
            return None
        return repetitions[0]

    def _dynamic_full_power_score_times_level_is_visible(
        self,
        card: Mapping[str, Any],
        customization_id: int,
        compact: str,
        level: int,
    ) -> bool:
        descriptor = self._dynamic_full_power_score_times_descriptor(
            card,
            customization_id,
        )
        if descriptor is None or level not in (0, 1):
            return False
        base_score, full_power_percent, increment = descriptor
        repetitions = self._visible_dynamic_full_power_score_repetitions(
            compact,
            base_score=base_score,
            full_power_percent=full_power_percent,
        )
        return repetitions == 1 + increment * level

    @staticmethod
    def _upgrade_hand_detail_signature(
        card: Mapping[str, Any],
        compact: str,
    ) -> bool | None:
        # An existing base action cannot prove whether an identical optional
        # action was added. Preserve that real ambiguity for such catalogs.
        if re.search(
            r"\bupgradeHand\b",
            f"{card.get('actions', '')};{card.get('effects', '')}",
        ):
            return None
        # Match one complete hand-wide action, not scattered background words.
        # Normalization already joins legal OCR line wraps and retains the
        # non-collapsible boundary between independently recognized views.
        return re.search(
            r"手札(?:(?:の|にある)スキルカード)?をすべて"
            r"(?:試験・ステージ中|レッスン中|ステージ中)?強化",
            compact,
        ) is not None

    @staticmethod
    def _good_impression_score_coefficient(card: Mapping[str, Any]) -> float | None:
        # A bare operand is multiplication by one.  Do not truncate a larger
        # expression or assign one visible percentage to two scoring actions.
        actions = str(card.get("actions", ""))
        bare = re.findall(
            r"(?:^|[;{])\s*score\+=goodImpressionTurns\s*(?=[;}]|$)", actions,
        )
        if not bare:
            # Preserve the established explicit-multiplier path, including
            # cards with repeated equal scores in different conditional blocks.
            explicit = re.search(r"score\+=goodImpressionTurns\*([0-9]+(?:\.[0-9]+)?)", actions)
            return float(explicit.group(1)) if explicit else None
        if len(bare) != 1 or actions.count("score+=goodImpressionTurns") != 1:
            return None
        return 1.0

    @staticmethod
    def _good_impression_score_is_visible(compact: str, percent: int) -> bool:
        views = tuple(view for view in compact.split(_OCR_EFFECT_VIEW_BOUNDARY) if "好印象の" in view)
        if any(len(set(re.findall(r"好印象の([0-9]+)%分スコア", view))) > 1 for view in views):
            raise ArenaCatalogError("one effect detail OCR view contains mutually exclusive good-impression percentages")
        return bool(views) and all(
            view.count("好印象の") == 1
            and re.findall(r"好印象の([0-9]+)%分スコア", view) == [str(percent)]
            for view in views
        )

    def _named_action_threshold_descriptor(
        self,
        card: Mapping[str, Any],
        customization_id: int,
    ) -> tuple[str, int, int] | None:
        """Bind a pure threshold replacement to one uniquely rendered field.

        Anchors are DSL identities, not displayed text.  Repeated conditions
        on the same field cannot be attributed from their text alone.
        """
        definition = self._customizations[customization_id]
        if int(definition["max"]) != 1 or any(
            definition.get(key) not in (None, "")
            for key in ("conditions", "effects", "cost", "limit", "forceInitialHand")
        ):
            return None
        patch = re.fullmatch(
            r"@([A-Za-z_][A-Za-z_0-9]*)\s+if:"
            r"(concentration|goodConditionTurns)>=([0-9]+)",
            str(definition.get("actions", "")).strip(),
        )
        if patch is None:
            return None
        anchor, field, custom_text = patch.groups()
        actions = str(card.get("actions", ""))
        base = re.findall(
            rf"@{re.escape(anchor)}\s+if:{field}>=([0-9]+)\s*\{{", actions,
        )
        if (
            len(base) != 1
            or len(re.findall(rf"@{re.escape(anchor)}\b", actions)) != 1
            or len(re.findall(rf"\bif:{field}\b", actions)) != 1
            or re.search(rf"\bif:{field}\b", str(card.get("conditions", "")))
        ):
            return None
        base_threshold, custom_threshold = int(base[0]), int(custom_text)
        if base_threshold == custom_threshold:
            return None
        for other_id in self.available_customization_ids(int(card["id"])):
            if other_id != customization_id and re.search(
                rf"@{re.escape(anchor)}\b",
                str(self._customizations[other_id].get("actions", "")),
            ):
                return None
        return field, base_threshold, custom_threshold

    @staticmethod
    def _named_action_threshold_is_visible(compact: str, field: str, threshold: int) -> bool:
        label, unit = (
            ("集中", "") if field == "concentration" else ("好調", "ターン")
        )
        prefix = rf"(?<!絶){label}が"
        views = tuple(view for view in compact.split(_OCR_EFFECT_VIEW_BOUNDARY) if re.search(prefix, view))
        return bool(views) and all(
            len(re.findall(prefix, view)) == 1
            and re.search(rf"{prefix}{threshold}{unit}以上", view) is not None
            for view in views
        )

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

        if "moveRandomToTopOfDeck[" in _normalise_text(
            definition.get("actions")
        ):
            move_random_to_top = self._move_random_to_top_descriptor(
                card,
                customization_id,
            )
            if move_random_to_top is None:
                return None
            rarity, card_count = move_random_to_top
            visible_pattern = re.compile(
                rf"ランダムな山札か捨(?:て)?札にある"
                rf"スキルカード\({re.escape(rarity)}\)"
                rf"{card_count}枚を山札の[一ー]番上に移動"
            )

            def match_move_random_to_top(
                compact: str,
                count: int,
                *,
                rarity: str = rarity,
                card_count: int = card_count,
                visible_pattern: Any = visible_pattern,
            ) -> bool:
                # This action has a stable, fully rendered contest-detail row.
                # Bind every semantic operand from the DSL instead of matching
                # loose words or fragments from independent OCR views.
                del rarity, card_count
                visible = visible_pattern.search(compact) is not None
                return visible is (count > 0)

            return match_move_random_to_top

        normalized_customization_actions = _normalise_text(
            definition.get("actions")
        )
        good_condition_applied_score = (
            self._good_condition_applied_score_descriptor(
                card,
                customization_id,
            )
        )
        if good_condition_applied_score is not None:
            base_score, multiplier, coefficient = good_condition_applied_score

            def match_good_condition_applied_score(
                compact: str,
                count: int,
                *,
                base_score: int = base_score,
                multiplier: int = multiplier,
                coefficient: int = coefficient,
            ) -> bool:
                return self._good_condition_applied_score_level_is_visible(
                    compact,
                    count,
                    base_score=base_score,
                    multiplier=multiplier,
                    coefficient=coefficient,
                )

            return match_good_condition_applied_score
        if re.fullmatch(
            r"@basedo\{goodConditionTurnsMultiplier=[2-9][0-9]*;"
            r"score\+=[1-9][0-9]*\};"
            r"do\{score\+=goodConditionTurns\*[1-9][0-9]*\};?",
            normalized_customization_actions,
        ):
            # A complex replacement row must never fall through to the simple
            # additive matcher merely because both DSLs contain ``score+=N``.
            # Unsupported or impure variants remain explicitly fail-closed.
            return None

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

        if re.search(
            r"g\.scoreTimes\+=([1-9][0-9]*)",
            str(definition.get("effects", "")),
        ):
            if self._dynamic_full_power_score_times_descriptor(
                card,
                customization_id,
            ) is not None:

                def match_dynamic_full_power_score_times(
                    compact: str,
                    count: int,
                ) -> bool:
                    return self._dynamic_full_power_score_times_level_is_visible(
                        card,
                        customization_id,
                        compact,
                        count,
                    )

                return match_dynamic_full_power_score_times

            customization_effects = str(definition.get("effects", ""))
            base_score_values = _all_literal_integer_increments(
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
            direct_score_times = _direct_prestage_target_this_increment(
                definition.get("effects"),
                "scoreTimes",
                1,
            )
            if "@grow" not in customization_effects and direct_score_times is None:
                return None
            if direct_score_times is not None and not base_score_values:
                # A dynamic score expression needs a frozen UI carrier; its
                # leading constant is not an independently rendered score.
                return None
            if direct_score_times is not None and int(definition["max"]) != 1:
                return None

            direct_score_times_groups: tuple[
                tuple[Mapping[str, int], tuple[str, ...]],
                ...,
            ] = ()
            if direct_score_times is not None:
                card_id = int(card["id"])
                available = self.available_customization_ids(card_id)
                maximum_total = sum(
                    int(self._customizations[item]["max"]) for item in available
                )
                signatures: list[tuple[Mapping[str, int], tuple[str, ...]]] = []
                for expected_count in range(1, maximum_total + 1):
                    for group in self.legal_customization_groups(
                        card_id,
                        expected_count=expected_count,
                    ):
                        if int(group.get(str(customization_id), 0)) < 1:
                            continue
                        selected_score_delta = self._selected_direct_growth_increment(
                            group,
                            "score",
                        )
                        anchors = (
                            f"スコア+{base_score_values[0] + selected_score_delta}",
                        )
                        signatures.append((group, anchors))
                direct_score_times_groups = tuple(signatures)

            def match_score_times(compact: str, count: int) -> bool:
                if base_growth_score is None:
                    if not base_score_values:
                        return False
                    repeated = any(
                        self._detail_counted_score_anchor_present(
                            card,
                            group,
                            anchor,
                            compact,
                        )
                        for group, anchors in direct_score_times_groups
                        for anchor in anchors
                    )
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

        named_threshold = self._named_action_threshold_descriptor(card, customization_id)
        if named_threshold is not None:
            field, base_threshold, custom_threshold = named_threshold
            named_thresholds = tuple(
                descriptor
                for other_id in self.available_customization_ids(int(card["id"]))
                if (descriptor := self._named_action_threshold_descriptor(card, other_id))
                is not None
            )

            def match_named_action_threshold(compact: str, count: int) -> bool:
                if count not in (0, 1):
                    return False
                # Preserve legacy completion for missing rows.  Only an
                # explicitly rendered invalid or conflicting peer value vetoes
                # positive evidence from another condition in the same card.
                if not all(
                    not re.search(rf"(?<!絶){'集中' if key == 'concentration' else '好調'}が", compact)
                    or any(self._named_action_threshold_is_visible(compact, key, value) for value in (base, custom))
                    for key, base, custom in named_thresholds
                ):
                    return False
                return self._named_action_threshold_is_visible(
                    compact, field, custom_threshold if count else base_threshold,
                )

            return match_named_action_threshold

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
            base_coefficient = self._good_impression_score_coefficient(card)
            if base_coefficient is None:
                return None
            bare_coefficient = "score+=goodImpressionTurns*" not in str(card.get("actions", ""))

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
                return (
                    self._good_impression_score_is_visible(compact, percent)
                    if bare_coefficient else f"好印象の{percent}%分スコア" in compact
                )

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
                visible = self._upgrade_hand_detail_signature(card, compact)
                return visible is None or visible is (count > 0)

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
