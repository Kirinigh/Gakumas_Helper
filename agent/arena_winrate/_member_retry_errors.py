"""Fixed resource/configuration failures cannot be repaired by another read."""

from __future__ import annotations

import re

from ._reader_errors import ArenaReaderError

_P_ITEM_TEXT_RESOLUTION_FIELD = re.compile(r"(?:^|;\s*)text_resolution=")
_P_ITEM_TEXT_REFERENCE_GAP = re.compile(
    r"(?:^|;\s*)text_resolution=(?P<quote>['\"])"
    r"(?:ambiguous|unknown): reference_unavailable(?P=quote)(?=;|$)",
)


def p_item_text_reference_unavailable(code: str, detail: str) -> bool:
    """Accept only a P-detail read result's own exact field.

    Recovery failures and arbitrary OCR/error mentions remain separate. The
    transaction may retain this primary result while its mandatory restoration
    is still unproven; callers must preserve the separate return obligation.
    """
    if code != "p_item_detail_ambiguous":
        return False
    field = _P_ITEM_TEXT_RESOLUTION_FIELD.search(detail)
    # Inspect the transaction's first result field, never a later mention
    # inside retained raw OCR or another error's quoted detail.
    return field is not None and _P_ITEM_TEXT_REFERENCE_GAP.match(detail, field.start()) is not None


def fixed_member_read_failure(error: Exception) -> ArenaReaderError | None:
    """Find only an immutable catalog/configuration failure, never OCR sameness.

    Generic parser errors, filesystem failures and frame conflicts are not
    classified here. A recovery wrapper's typed primary error is independent
    of its return failure; string mentions cannot establish that ownership.
    """
    current = error
    seen = set()
    while isinstance(current, ArenaReaderError) and id(current) not in seen:
        seen.add(id(current))
        if current.code in {
            "skill_card_reference_catalog_invalid",
            "skill_card_reference_gallery_update_required",
            "p_item_reader_missing",
            "p_item_runtime_not_approved",
        } or p_item_text_reference_unavailable(current.code, current.detail):
            return current
        current = current.primary_error or current.__cause__
    return None
