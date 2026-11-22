"""Inspect retained one-time Produce settings without choosing player preferences."""
from __future__ import annotations

from .activity_production import canonical

DIALOGS = {
    'animation_skip_setting': '演出スキップ設定',
    'dialogue_speed_setting': 'コミュ早送り設定',
    'voice_setting': 'ボイス再生確認',
    'card_guide_setting': '獲得ガイド表示設定',
}
SWITCHES = {
    'animation_skip_setting': {
        'support_commu': ('サポートイベントの', 'コミュスキップ'),
        'start_commu': ('各プロデュース開始時の', '説明コミュスキップ'),
        'hajime_commu': ('気になる', '様子」コミュスキップ'),
        'affection_movie': ('親愛度コミュ内の', '映像スキップ'),
    },
    'dialogue_speed_setting': {'unread_fast_forward': ('未読コミュでも早送り',)},
}
OPTIONS = {
    'dialogue_speed_setting': ('コミュを早送りする', 'コミュを早送りしない'),
    'voice_setting': ('ボイスありで再生', 'ボイスなしで再生'),
    'card_guide_setting': ('獲得ガイドを表示する', '獲得ガイドを表示しない'),
}
HIRO_PREFERENCES = {
    'animation_skip_setting': {'switches': dict.fromkeys(SWITCHES['animation_skip_setting'], True)},
    'dialogue_speed_setting': {'option': 'コミュを早送りする', 'switches': {'unread_fast_forward': True}},
    'voice_setting': {'option': 'ボイスありで再生'},
    'card_guide_setting': {'option': '獲得ガイドを表示しない'},
}


class ProduceSettingsObserved(ValueError):
    """A currently visible Produce settings modal interrupts the business page."""


def read_produce_settings(rows):
    texts = [(canonical(row['text']), row['box']) for row in rows if row['score'] >= .95]
    kinds = [kind for kind, title in DIALOGS.items() if any(title in text for text, _ in texts)]
    if len(kinds) != 1:
        return None
    kind = kinds[0]
    controls = {}
    for key, labels in SWITCHES.get(kind, {}).items():
        heights = []
        for title in labels:
            heights = [y + h / 2 for text, (_, y, _, h) in texts if text == title or text.startswith(title)]
            if heights:
                break
        candidates = [(value, box) for value, box in texts if value in ('ON', 'OFF')
            and box[0] > 480 and any(abs(box[1] + box[3] / 2 - y) <= 60 for y in heights)]
        controls[key] = candidates[0] if len(candidates) == 1 else None
    options = {}
    for name in OPTIONS.get(kind, ()):
        matches = [box for text, box in texts if name in text]
        options[name] = matches[0] if len(matches) == 1 else None
    confirm = [box for text, box in texts if text.lstrip('く<〈✓✔√レ') == '決定']
    return dict(dialog=kind, switches={key: (dict(value=match[0] == 'ON', box=match[1])
        if match is not None and list(controls.values()).count(match) == 1 else None)
        for key, match in controls.items()},
        options=options, confirm=confirm[0] if len(confirm) == 1 else None)


def settings_preview(rows, preferences):
    """Show where explicit preferences could be applied to the saved dialog."""
    state = read_produce_settings(rows)
    if state is None:
        return None
    changes, unknown = [], []
    for key, desired in (preferences.get('switches') or {}).items():
        if key not in state['switches'] or type(desired) is not bool:
            raise ValueError('preference must name a switch on the observed dialog and its desired state')
        before = state['switches'][key]
        if before is None:
            unknown.append(key)
        elif before['value'] != desired:
            changes.append(dict(setting=key, value=desired, box=before['box']))
    option = preferences.get('option')
    if option is not None:
        if option not in state['options']:
            raise ValueError('preference must name an option on the observed dialog')
        box = state['options'][option]
        if box is None:
            unknown.append('option')
        else:
            changes.append(dict(setting='option', value=option, box=box))
    return dict(**state, requested_changes=changes, unread_preferences=unknown)


def next_setting(state, completed):
    """Move through the selected Hiro profile one currently visible control at a time."""
    preferences = HIRO_PREFERENCES[state['dialog']]
    option = preferences.get('option')
    if option and 'option' not in completed and state['options'][option] is not None:
        return 'option', state['options'][option]
    for key, value in preferences.get('switches', {}).items():
        switch = state['switches'][key]
        if switch is not None and switch['value'] != value and completed.count(key) < 3:
            return key, switch['box']
    if state['confirm'] is not None and 'confirm' not in completed:
        return 'confirm', state['confirm']
    return None


def detect_produce_settings(context, rows):
    host = getattr(context, 'gkh_hif_route_host', None)
    if (getattr(context, 'gkh_hif_cultivation_runtime', None) is None
            or host is None or not getattr(host, '_active', False)
            or (host.ownership or {}).get('owner') != 'task600'):
        return
    # The full reader is needed only on a matching dialog.
    if not any(title in row['text'] for row in rows for title in DIALOGS.values()):
        return
    state = read_produce_settings(rows)
    if state is not None and state['confirm'] is not None:
        raise ProduceSettingsObserved(state['dialog'])
