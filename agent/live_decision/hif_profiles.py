"""Resolve HIF identity from the observed idol and song in the shipped catalog."""

import json
import unicodedata
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

if __package__.startswith('agent.'):
    from agent.utils.runtime_paths import resolve_runtime_path
else:
    from utils.runtime_paths import resolve_runtime_path


@dataclass(frozen=True)
class HifIdentityProfile:
    identity: str
    idol: str
    song: str
    plan: str = 'sense'


HIF_IDENTITY_PROFILES = (
    HifIdentityProfile('篠澤広(ガラクタロード)', '篠澤広', 'ガラクタロード'),
    # Native notebook 1712/1713: exact version label, idol and focus in one page.
    HifIdentityProfile('十王星南(「ねえ、言っちゃうよ。」)', '十王星南', '「ねえ、言っちゃうよ。」'),
)

_PLANS = {'感性': 'sense', '理性': 'logic', '非凡': 'anomaly'}


def _display_key(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


@lru_cache(maxsize=1)
def _catalog_profiles():
    path = resolve_runtime_path(Path('assets/data/idols_cards.json'), Path('data/idols_cards.json'))
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    profiles = list(HIF_IDENTITY_PROFILES)
    authored = {(_display_key(row.idol), _display_key(row.song)) for row in profiles}
    for group in data.values():
        if not isinstance(group, list):
            continue
        for row in group:
            plan = _PLANS.get(row['推荐效果'].split('·', 1)[0])
            idol, song = row['偶像名称'], row['歌曲名称']
            if plan and (_display_key(idol), _display_key(song)) not in authored:
                profiles.append(HifIdentityProfile(f'{idol}({song})', idol, song, plan))
    return tuple(profiles)


def supported_hif_identity(identity, plan):
    for profile in _catalog_profiles():
        if (profile.identity, profile.plan) == (identity, plan):
            return profile
    raise ValueError('HIF Live identity is outside the observed profiles')


def notebook_hif_identity(song, idol):
    key = (_display_key(idol), _display_key(song))
    matches = {profile for profile in _catalog_profiles()
               if (_display_key(profile.idol), _display_key(profile.song)) == key}
    if len(matches) != 1:
        raise ValueError('HIF notebook idol and version do not identify one catalog entry')
    return matches.pop()
