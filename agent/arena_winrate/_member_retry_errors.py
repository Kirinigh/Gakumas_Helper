"""Static P-item text gaps cannot be repaired by another member observation."""

from __future__ import annotations

import re

_P_ITEM_TEXT_RESOLUTION_FIELD = re.compile(r"(?:^|;\s*)text_resolution=")
_P_ITEM_TEXT_REFERENCE_GAP = re.compile(
    r"(?:^|;\s*)text_resolution=(?P<quote>['\"])"
    r"(?:ambiguous|unknown): reference_unavailable(?P=quote)(?=;|$)",
)


def p_item_text_reference_unavailable(code: str, detail: str) -> bool:
    """Accept only the restored P-detail transaction's exact result field.

    Recovery failures and arbitrary OCR/error mentions remain separate. The
    transaction emits this ambiguity only after mandatory source restoration.
    """
    if code != "p_item_detail_ambiguous":
        return False
    field = _P_ITEM_TEXT_RESOLUTION_FIELD.search(detail)
    # Inspect the transaction's first result field, never a later mention
    # inside retained raw OCR or another error's quoted detail.
    return field is not None and _P_ITEM_TEXT_REFERENCE_GAP.match(detail, field.start()) is not None
