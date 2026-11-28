"""Shared terminology for comparing in-battle effect descriptions."""

import re


def normalize_score_effect_text(text: str) -> str:
    """Use スコア for known score effects in already compacted text."""
    return re.sub(r'パラメータ(?=\+\d|上昇|値増加\+\d)|(?<=%分)パラメータ', 'スコア', text)
