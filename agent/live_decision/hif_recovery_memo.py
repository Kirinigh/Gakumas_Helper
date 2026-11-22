"""One-call reuse of unchanged, already verified recovery OCR regions.

This stores computation only. The caller must supply the current frame and a
trusted model binding, and must run its original fresh guard on every access.
No native object, input permission or timestamp is restored from the cache.
"""
from __future__ import annotations

import re
import json
import unicodedata
from types import SimpleNamespace

import numpy as np

from .hif_session import notebook_identity_fields

MENU_FIELDS = {
    'Notebook': ((243, 1004, 74, 35), 'P手帳'),
    'Help': ((407, 1004, 75, 35), 'ヘルプ'),
}
IDENTITY_FIELDS = frozenset(('song', 'idol', 'focus'))
RECOVERY_FIELD_NAMES = (*MENU_FIELDS, *sorted(IDENTITY_FIELDS))
_FIELD_SPECS = {(name, roi): pattern for name, (roi, pattern) in MENU_FIELDS.items()}
for _layout in ('legacy', 'campaign_banner'):
    for _name, (_roi, _pattern) in notebook_identity_fields(2, layout=_layout).items():
        if _name in IDENTITY_FIELDS:
            _FIELD_SPECS[(_name, _roi)] = _pattern


def recovery_field_spec(field, roi=None):
    if roi is None and field in MENU_FIELDS:
        roi = MENU_FIELDS[field][0]
    if not isinstance(roi, (tuple, list)) or (field, tuple(roi)) not in _FIELD_SPECS:
        raise ValueError('recovery crop memo field or original region is unsupported')
    return tuple(roi), _FIELD_SPECS[(field, tuple(roi))]


class RecoveryCropMemo:
    def __init__(self, *, call_id, epoch):
        if not isinstance(call_id, str) or not call_id or not isinstance(epoch, str) or not epoch:
            raise ValueError('recovery crop memo needs its original call and epoch')
        self.call_id, self.epoch = call_id, epoch
        self._entries = {}
        self._active = True

    def close(self):
        self._entries.clear()
        self._active = False

    def recognise(self, field, image, *, call_id, epoch, frame_id, model_key,
                  parameters, guard, recognise, roi=None):
        try:
            return self._recognise(field, image, call_id=call_id, epoch=epoch, frame_id=frame_id,
                model_key=model_key, parameters=parameters, guard=guard, recognise=recognise, roi=roi)
        except Exception:
            self.close()
            raise

    def _recognise(self, field, image, *, call_id, epoch, frame_id, model_key,
                   parameters, guard, recognise, roi):
        if not self._active or call_id != self.call_id or epoch != self.epoch:
            raise ValueError('recovery crop memo call or epoch changed')
        if not callable(guard) or guard() is not True:
            raise ValueError('recovery crop memo fresh pixel permission rejected')
        if (not isinstance(image, np.ndarray)
                or image.shape != (1280, 720, 3) or image.dtype != np.uint8
                or not isinstance(frame_id, str) or not frame_id):
            raise ValueError('recovery crop memo lacks its complete current field')
        roi, expected = recovery_field_spec(field, roi)
        x, y, width, height = roi
        crop = image[y:y + height, x:x + width].copy()
        pixels = crop.tobytes()
        if (not isinstance(parameters, dict) or parameters.get('recognition') != 'OCR'
                or parameters.get('only_rec') is not True or parameters.get('expected') != '.+'):
            raise ValueError('recovery crop memo recognition contract changed')
        encoded = json.dumps(parameters, sort_keys=True, allow_nan=False, separators=(',', ':'))
        # An unknown model binding never turns an old result into new evidence.
        model = (json.dumps(model_key, sort_keys=True, allow_nan=False, separators=(',', ':'))
                 if model_key else None)
        key = (model, encoded, roi, crop.shape, crop.dtype.str, pixels)
        slot = field, roi
        previous = self._entries.get(slot)
        if model is not None and previous is not None and previous['key'] == key and previous['frame_id'] != frame_id:
            row = previous['row']
            result = SimpleNamespace(hit=True, reco_id=previous['reco_id'], name='HifRecovery' + field,
                algorithm='OCR', filtered_results=[SimpleNamespace(text=row[0], score=row[1], box=list(row[2]))])
            return result, self._evidence(field, frame_id, previous, reused=True)

        # A changed, weak or failed observation must not leave an older entry
        # available for a later return to the same crop.
        self._entries.pop(slot, None)
        result = recognise(crop)
        rows = list(result.filtered_results) if result is not None and result.hit else []
        accepted = rejected = False
        if len(rows) == 1:
            row = rows[0]
            text = ''.join(unicodedata.normalize('NFKC', row.text).split()) if isinstance(row.text, str) else ''
            box = tuple(row.box)
            complete = (bool(text) and type(row.score) in (int, float) and 0 <= row.score <= 1
                and box == (0, 0, width, height)
                and type(getattr(result, 'reco_id', None)) is int and result.reco_id > 0)
            accepted = complete and row.score >= .95 and re.fullmatch(expected, text) is not None
            # An unchanged rejected identity crop stays rejected. This only
            # avoids repeating OCR of the same unsupported alternate layout;
            # the caller still applies its original .95/identity/unique gates.
            rejected = complete and field in IDENTITY_FIELDS and row.score < .95
        if (accepted or rejected) and model is not None:
            entry = dict(key=key, row=(row.text, row.score, box), frame_id=frame_id,
                         reco_id=result.reco_id, accepted=accepted)
            self._entries[slot] = entry
            return result, self._evidence(field, frame_id, entry, reused=False)
        return result, dict(source='original_crop_ocr', field=field, current_frame_id=frame_id,
                            source_frame_id=frame_id, reusable=False)

    def _evidence(self, field, frame_id, entry, *, reused):
        roi = entry['key'][2]
        return dict(source='exact_crop_compute_reuse' if reused else 'original_crop_ocr',
            field=field, source_reco_id=entry['reco_id'], source_frame_id=entry['frame_id'],
            current_frame_id=frame_id, call_id=self.call_id, clock_epoch=self.epoch,
            source_model_binding=json.loads(entry['key'][0]),
            recognition_parameters=json.loads(entry['key'][1]),
            roi=list(roi), shape=[roi[3], roi[2], 3], dtype='uint8', byte_length=roi[2] * roi[3] * 3,
            full_crop_bytes_equal=reused, new_ocr=not reused, reusable=True, source_accepted=entry['accepted'])
