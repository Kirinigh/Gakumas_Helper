"""Deterministic, offline compilation of source-owned arena P-item text.

Literal native descriptions are the text authority. Catalog effects are only
binding evidence and are never parsed to synthesize a description.
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from collections import defaultdict
from collections.abc import Mapping, Sequence

GENERATOR_REVISION = "task410-p-item-text-compiler-v1"
MAX_REFERENCE_BYTES = 2 * 1024 * 1024
AUDITION = "ProduceDescriptionSwapType_Audition"
PLANS = {"ProducePlanType_Plan1": "sense", "ProducePlanType_Plan2": "logic",
         "ProducePlanType_Plan3": "anomaly", "ProducePlanType_Common": "free"}
RARITIES = {"ProduceItemRarity_Ssr": "SSR", "ProduceItemRarity_Sr": "SR",
            "ProduceItemRarity_R": "R", "ProduceItemRarity_N": "N"}
DYNAMIC = ("effectValue1", "effectValue2", "effectCount", "turn", "costValue", "targetLevel")
NUMBER = re.compile(r"[+＋\-－]?\d+(?:\.\d+)?[%％]?")
TEMPLATE = re.compile(r"[{}]|\$\(|%[sd]|<[^>]+>")
SPECIAL_FLAGS = ("isChallenge", "isHighScoreRush", "isResearch", "isEasy")
# Reviewed mappings also used by tools/p-item-embedding/build_p_item_classification.py.
# Require the expected catalog spelling as well as its ID; an ID alone cannot
# authorize applying an old alias to a newly changed catalog identity.
ALIASES = {
    174: ("アイドルパワー測定器", "アイドルパワー測定機"),
    242: ("Pっち", "「Pっち」"), 243: ("Pっち+", "「Pっち」+"),
    310: ("ビシッビシッ竹刀", "ビシッバシッ竹刀"),
    380: ("ぴったりコレクション", "ぴったしコレクション"),
    383: ("そっくりワンマン", "そっくりワンワン"),
    419: ("夜空に従える花", "夜空を従える花"),
    420: ("夜空に従える花+", "夜空を従える花+"),
}
PLAN_DIFFERENCES = {115: ("logic", "free"), 142: ("free", "logic"),
                    406: ("sense", "free"), 407: ("logic", "free"), 408: ("anomaly", "free")}
SPLIT_SOURCE = "pitem_00-3-325-0"
REFERENCE_REVISION = re.compile(r"[A-Za-z0-9._-]+-v([1-9][0-9]*)\Z")


def validate_reference_revision(value: object) -> str:
    """Validate an independent artifact revision with a monotonic ``-vN`` suffix."""
    if not isinstance(value, str) or len(value) > 96 or REFERENCE_REVISION.fullmatch(value) is None:
        raise ValueError("invalid reference revision: expected ASCII identity ending -vN (N>=1), at most 96 characters")
    return value


def reference_revision_number(value: object) -> int:
    """Return the validated revision's numeric suffix for upgrade comparisons."""
    valid = validate_reference_revision(value)
    return int(REFERENCE_REVISION.fullmatch(valid).group(1))


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _binding_matches(active: Mapping, reference: Mapping) -> bool:
    # Same contract as _p_item_detail_text._binding_matches. Keep this module
    # independently loadable by packaging gates without importing runtime UI.
    fields = ("id", "name", "mode", "sourceType", "pIdolId", "upgraded", "plan", "rarity")
    return bool(isinstance(active, Mapping) and isinstance(reference, Mapping)
                and all(active.get(key) == reference.get(key) for key in fields)
                and isinstance(active.get("effects"), str)
                and re.sub(r"\s+", "", active["effects"])
                == re.sub(r"\s+", "", str(reference.get("effects", ""))))


def unique_rows(rows: Sequence[Mapping], key: str) -> dict:
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or key not in row:
            raise ValueError(f"invalid row or missing {key}")
        value = row[key]
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"invalid {key}: expected a scalar identity") from exc
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def _valid_body(row: Mapping) -> bool:
    name, paragraphs = row.get("name"), row.get("paragraphs")
    return bool(isinstance(name, str) and normalize(name)
                and isinstance(paragraphs, list) and paragraphs
                and all(isinstance(text, str) and normalize(text) for text in paragraphs))


def validate_reference_asset(value: object) -> Mapping:
    """The full runtime asset contract, including rows outside this RIS catalog."""
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("unsupported reference schema")
    validate_reference_revision(value.get("revision"))
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("reference rows must be an array")
    indexed = unique_rows(rows, "id")
    if any(type(item_id) is not int or item_id <= 0 for item_id in indexed):
        raise ValueError("invalid reference business id")
    for row in rows:
        if not _valid_body(row) or not isinstance(row.get("active"), Mapping):
            raise ValueError(f"invalid reference row: {row['id']}")
    return value


def reference_coverage(reference: Mapping, catalog: Sequence[Mapping]) -> dict:
    """Cover only inherited arena items; report missing and stale separately."""
    if not isinstance(reference, Mapping) or reference.get("schema_version") != 1 or not isinstance(reference.get("rows"), list):
        raise ValueError("unsupported reference schema")
    validate_reference_revision(reference.get("revision"))
    refs = unique_rows(reference["rows"], "id")
    active = unique_rows(catalog, "id")
    if any(type(key) is not int or key <= 0 for key in refs.keys() | active.keys()):
        raise ValueError("invalid business id")
    eligible = {key: row for key, row in active.items()
                if row.get("mode") == "stage" and row.get("sourceType") in ("pIdol", "support")}
    return {
        "required_count": len(eligible),
        "missing_ids": sorted(key for key in eligible if key not in refs or not _valid_body(refs[key])),
        "binding_mismatch_ids": sorted(key for key in eligible.keys() & refs.keys()
                                       if not _binding_matches(eligible[key], refs[key].get("active", {}))),
    }


def render_paragraphs(source: Mapping, labels: Mapping, swaps: Mapping) -> list[str]:
    """Project explicit Audition swaps and literals, rejecting unknown templates.

    The swaps index is keyed by (id, swapType), because a native swap ID may
    legitimately have both Produce and Audition records.
    """
    segments = source.get("produceDescriptions")
    if not isinstance(segments, list):
        raise ValueError("missing source description segments")
    rendered = []
    for index, segment in enumerate(segments):
        try:
            original = segment["text"]
            if not isinstance(original, str):
                raise ValueError("invalid source literal")
            target = segment.get("targetId", "")
            label = labels.get(target, {})
            if isinstance(label, list):
                if len(label) != 1:
                    raise ValueError(f"duplicate source label: {target}")
                label = label[0]
            swap_id = segment.get("produceDescriptionSwapId") or label.get("produceDescriptionSwapId")
            text = original
            if swap_id:
                swap = swaps.get((swap_id, AUDITION))
                if isinstance(swap, list):
                    if len(swap) != 1:
                        raise ValueError(f"duplicate audition swap: {swap_id}")
                    swap = swap[0]
                if swap is None or swap.get("swapType") != AUDITION:
                    raise ValueError(f"unresolved audition swap: {swap_id}")
                text = swap["text"]
            # Decoded metadata can accompany an already materialized literal
            # (e.g. literal "30", effectValue1=300). Use that source verbatim;
            # only an empty slot would require unsupported dynamic evaluation.
            if not original and any(segment.get(key, 0) for key in DYNAMIC):
                raise ValueError("unresolved dynamic source value")
            if not original and label.get("iconAssetId") == "style-dot":
                text = "\n"
            elif not original and segment.get("produceDescriptionType") == "ProduceDescriptionType_Exam":
                pass  # Zero-valued decoded slots do not add visible prose.
            elif not original and target and not text:
                raise ValueError("unresolved empty source target")
            text = re.sub(r"</?nobr>", "", text)
            if TEMPLATE.search(text) or NUMBER.findall(original) != NUMBER.findall(text):
                raise ValueError("unsupported template/markup or changed source numbers")
            rendered.append(text)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"segment {index}: {exc}") from exc
    paragraphs = [text for text in "".join(rendered).split("\n") if text]
    if not paragraphs or not any(normalize(text) for text in paragraphs):
        raise ValueError("empty source body")
    return paragraphs


def _table(tables: Mapping, name: str) -> Sequence[Mapping]:
    rows = tables.get(name, tables.get(name + ".yaml", ()))
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"invalid native table: {name}")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"invalid native table row: {name}")
    return rows


def _group(rows: Sequence[Mapping], keys: tuple[str, ...]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return dict(grouped)


def _mapped_name(row: Mapping) -> str:
    name = normalize(str(row.get("name", "")))
    alias = ALIASES.get(row["id"])
    if alias and name == normalize(alias[0]):
        name = normalize(alias[1])
    return name.removesuffix("+")


def _validate_source(active: Mapping, source: Mapping, idols: Mapping, supports: Mapping) -> None:
    plan = PLANS.get(source.get("planType"))
    if not (source.get("isExamEffect") is True
            and type(active.get("upgraded")) is bool
            and source.get("isUpgraded") is active["upgraded"]
            and (plan == active.get("plan") or PLAN_DIFFERENCES.get(active["id"]) == (active.get("plan"), plan))
            and RARITIES.get(source.get("rarity")) == active.get("rarity")):
        raise ValueError("source identity disagrees with active item")
    origin, other = ("originIdolCardId", "originSupportCardId") if active["sourceType"] == "pIdol" else ("originSupportCardId", "originIdolCardId")
    if source.get(other):
        raise ValueError("source origin type disagrees with active item")
    origin_id = source.get(origin)
    if not origin_id:
        if (active["id"] in (406, 407, 408) and source.get("id") == SPLIT_SOURCE
                and normalize(str(source.get("name", ""))) == "あさり先生のおまもり"):
            return  # Existing reviewed business Plan split, not a generic bypass.
        raise ValueError("missing source origin")
    cards = idols if active["sourceType"] == "pIdol" else supports
    matches = cards.get((origin_id,), ())
    if len(matches) != 1:
        raise ValueError("missing or duplicate source origin card")
    if active["sourceType"] == "pIdol":
        card = matches[0]
        keys = ("afterProduceItemId", "afterLevelLimitProduceItemId") if active["upgraded"] else ("beforeProduceItemId", "beforeLevelLimitProduceItemId")
        if source["id"] not in {card.get(key) for key in keys}:
            raise ValueError("source origin card item relationship disagrees")


def compile_reference(
    reference: Mapping, catalog: Sequence[Mapping], native_tables: Mapping[str, Sequence], *,
    source_revision: str, catalog_revision: str, revision: str,
) -> dict:
    """Return ``{reference, report}`` without inventing unresolved item bodies.

    Unchanged old rows stay equal and in their original order. Changed/removed
    bindings are invalidated, even if a same-named native record is available:
    identity alone does not establish equivalence of changed catalog effects.
    Such an ID needs a separately reviewed fresh baseline before regeneration.
    Global schema/identity/revision mistakes raise ValueError; source failures
    are diagnosed per ID and remain missing in the returned reference.
    """
    for name, value in (("source_revision", source_revision), ("catalog_revision", catalog_revision), ("revision", revision)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required")
    validate_reference_revision(revision)
    reference_coverage(reference, catalog)
    active = unique_rows(catalog, "id")
    preserved, invalidated, diagnostics, blocked = [], [], [], set()
    for row in reference["rows"]:
        item_id = row["id"]
        if item_id in active and _binding_matches(active[item_id], row.get("active", {})):
            if _valid_body(row):
                preserved.append(deepcopy(row))
            else:
                invalidated.append(item_id)
                diagnostics.append({"id": item_id, "reason": "reference_body_invalid"})
        else:
            invalidated.append(item_id)
            blocked.add(item_id)
            diagnostics.append({"id": item_id, "reason": "catalog_binding_changed" if item_id in active else "catalog_item_removed"})
    items = _table(native_tables, "ProduceItem")
    by_name = defaultdict(list)
    for row in items:
        if not any(row.get(key, False) for key in SPECIAL_FLAGS):
            by_name[(normalize(str(row.get("name", ""))).removesuffix("+"), row.get("isUpgraded"))].append(row)
    native_ids = _group(items, ("id",))
    labels = {key[0]: rows for key, rows in _group(_table(native_tables, "ProduceDescriptionLabel"), ("id",)).items()}
    swaps = _group(_table(native_tables, "ProduceDescriptionSwap"), ("id", "swapType"))
    idols, supports = (_group(_table(native_tables, name), ("id",)) for name in ("IdolCard", "SupportCard"))
    result = deepcopy(dict(reference))
    result["rows"] = preserved
    missing = reference_coverage(result, catalog)["missing_ids"]
    generated = []
    for item_id in missing:
        if item_id in blocked:
            continue
        row = active[item_id]
        matches = by_name.get((_mapped_name(row), row.get("upgraded")), ())
        candidate_ids = sorted(str(source.get("id", "")) for source in matches)
        if len(matches) != 1:
            diagnostics.append({"id": item_id, "reason": "source_mapping_missing" if not matches else "source_mapping_ambiguous", "source_ids": candidate_ids})
            continue
        source = matches[0]
        try:
            if not source.get("id") or len(native_ids[(source.get("id"),)]) != 1:
                raise ValueError("missing or duplicate source ID")
            _validate_source(row, source, idols, supports)
            paragraphs = render_paragraphs(source, labels, swaps)
        except (KeyError, TypeError, ValueError) as exc:
            diagnostics.append({"id": item_id, "reason": "source_compile_failed", "source_ids": candidate_ids, "detail": str(exc)})
            continue
        result["rows"].append({"id": item_id, "source_id": source["id"], "name": source["name"],
                               "paragraphs": paragraphs, "active": deepcopy(dict(row))})
        generated.append(item_id)
    result["revision"] = revision
    result.setdefault("source", {})["compilation"] = {
        "source_revision": source_revision, "catalog_revision": catalog_revision,
        "generator_revision": GENERATOR_REVISION,
    }
    if result != reference and reference_revision_number(revision) <= reference_revision_number(reference.get("revision")):
        raise ValueError("reference revision number must increase when content or provenance changes")
    return {"reference": result, "report": {
        **reference_coverage(result, catalog), "generated_ids": generated,
        "preserved_ids": [row["id"] for row in preserved], "invalidated_ids": sorted(invalidated),
        "diagnostics": sorted(diagnostics, key=lambda row: row["id"]),
    }}
