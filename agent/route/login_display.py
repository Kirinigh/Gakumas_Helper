"""Settled claimed-reward display in the original login navigation only.

No weekday, campaign, item name or purchase policy is selected here. The
caller must separately prove its successful title/login navigation context.
"""
from __future__ import annotations

import re

import cv2
import numpy as np


def hiro_login_notice(image, rows):
    """Find the affection unlock or item expiry notification on the login path."""
    from .date_recovery import _row

    captions = [row['text'] for row in rows if 160 <= row['box'][1] < 1100]
    if any('使用期限確認' in text or '使用期限の近い持ち物' in text for text in captions):
        cancel = _row(rows, r'キャンセル', (0, 800, 420, 480), minimum=0)
        return dict(scene='login_bonus', target=tuple(cancel['box']), page_kind='item-expiry')
    if any('親愛度' in text for text in captions) and any('解放' in text for text in captions):
        # The HUD can also show a round X. Require the actual notification's
        # caption and the same light ring around the dark X that Hiro uses.
        offset = np.arange(-40, 40)
        radius_squared = offset[:, None] ** 2 + offset[None, :] ** 2
        hsv = cv2.cvtColor(image[1160:1240, 320:400], cv2.COLOR_BGR2HSV)
        light = (hsv[..., 1] < 30) & (hsv[..., 2] > 230)
        dark = hsv[..., 2] < 110
        if light[(radius_squared > 22**2) & (radius_squared < 34**2)].mean() >= .8 and dark[radius_squared < 16**2].mean() >= .1:
            return dict(scene='login_bonus', target=(360, 1200, 1, 1),
                page_kind='affection-unlock:' + '|'.join(sorted(set(captions))))
    return None


def reject_login_controls(rows):
    conflicts = (r'(?:購入(?:する)?|有償.*|消費.*|使用する|交換する|確認(?:する)?|キャンセル|はい|いいえ|'
        r'選択.*|選ぶ|決定|\d+回引く|ガシャを引く|閉じる|OK|次へ|受け取る|戻る|やめる)')
    if any(row['score'] >= .85 and re.fullmatch(conflicts, row['text']) for row in rows):
        raise ValueError('login reward display conflicts with a purchase, choice or confirmation')


def claimed_login_display(image, rows):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('login reward display is uncalibrated')
    reject_login_controls(rows)
    periods = [row for row in rows if row['score'] >= .95
        and re.fullmatch(r'開催期間:\d{4}/\d{1,2}/\d{1,2}\d{2}:\d{2}まで', row['text'])
        and 120 <= row['box'][0] <= 220 and 190 <= row['box'][1] <= 250
        and row['box'][0]+row['box'][2] <= 620 and row['box'][1]+row['box'][3] <= 270]
    if len(periods) != 1:
        raise ValueError('login reward display lacks its settled campaign period')
    # This calibrated one-column calendar has a hollow red claimed stamp over
    # the lower reward tile. A red illustration or a text-only ad is not it.
    tile = image[938:1066, 296:424]
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 160)).astype('uint8')
    _, _, components, _ = cv2.connectedComponentsWithStats(red)
    outline = [row for row in components[1:] if 90 <= row[2] <= 116 and 90 <= row[3] <= 116
        and 1000 <= row[4] <= 2300 and row[0] >= 5 and row[1] >= 5]
    glyphs = [row for row in components[1:] if 300 <= row[4] <= 950 and 25 <= row[2] <= 65
        and 25 <= row[3] <= 70]
    white = ((tile.min(2) > 200) & (tile.max(2)-tile.min(2) < 40)).mean()
    if len(outline) != 1 or len(glyphs) != 2 or white < .30:
        raise ValueError('login reward display lacks its completed claimed stamp')
    def mean(x, y, w, h):
        return float(image[y:y+h, x:x+w].mean())
    overlay = (mean(176, 570, 8, 210) < .75*mean(150, 570, 8, 210)
        and mean(534, 570, 8, 210) < .75*mean(562, 570, 8, 210))
    if overlay:
        labels = [row for row in rows if row['score'] >= .95 and len(row['text']) >= 2
            and 185 <= row['box'][0] <= 310 and 740 <= row['box'][1] <= 800
            and row['box'][0]+row['box'][2] <= 560 and row['box'][1]+row['box'][3] <= 815]
        if len(labels) != 1:
            raise ValueError('login reward display item overlay is still animating')
    # Original upstream ClickCenter is this blank corner, despite its name.
    # It is admitted only for the proven display, never an unknown page.
    if any(row['score'] >= .85 and row['box'][0] < 50 and row['box'][1] < 50 for row in rows):
        raise ValueError('login reward blank continuation area is obstructed')
    phase = 'item-overlay' if overlay else 'claimed-calendar'
    return dict(scene='login_bonus', target=(0, 0, 50, 50),
        page_kind=f'claimed-login:{phase}:{periods[0]["text"]}')
