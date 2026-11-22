"""Create HIF run facts from start-page identity and the new opening HUD."""

import re
import math
import unicodedata
from dataclasses import dataclass

from .hif_visual import HifLiveRunContext
from .hif_profiles import notebook_hif_identity

NOTEBOOK_IDENTITY_FIELDS = {
    # Header controls move vertically with the client layout. The campaign
    # badge also shifts the event title left without moving the idol panel.
    'stage': ((40,25,145,60), r'H[.]I[.]F[.]?本戦'),
    'round': ((32,77,160,60), r'ラウンド1'),
    'title': ((20,185,440,50), r'HatsuboshiIDOLFESTIVAL'),
    'song': ((287,246,166,36), r'.+'),
    'idol': ((285,280,126,35), r'.+'),
    'stamina': ((324,45,110,58), r'\d{1,2}/\d{1,2}'),
}

# 1712/1713 has a campaign banner between the event heading and idol block.
# The header/resources remain fixed; only these complete fields move down.
NOTEBOOK_BANNER_FIELDS = {
    'song': (287,324,250,36),
    'idol': (285,358,150,36),
}


@dataclass(frozen=True)
class HifRoundContext:
    """Round metadata bound to its original Live-entry or notebook observations."""

    run_id: str
    revision: int
    round_number: int
    base_turn_count: int
    frame_ids: tuple[str, ...]

    def __post_init__(self):
        if (not isinstance(self.run_id, str) or not self.run_id.strip()
                or type(self.revision) is not int or self.revision < 1
                or type(self.round_number) is not int or self.round_number not in (1, 2)
                or type(self.base_turn_count) is not int
                or self.base_turn_count != {1: 9, 2: 12}[self.round_number]
                or not isinstance(self.frame_ids, (tuple, list)) or not self.frame_ids
                or any(not isinstance(key, str) or not key for key in self.frame_ids)
                or len(set(self.frame_ids)) != len(self.frame_ids)):
            raise ValueError('HIF round context lacks distinct source frames or valid round metadata')
        object.__setattr__(self, 'frame_ids', tuple(self.frame_ids))


def resolve_hif_round_context(round_context=None, *, run_id=None, revision=None):
    """Absent metadata preserves the historically calibrated round-one slice."""
    if round_context is None:
        return 1, 9
    if (not isinstance(round_context, HifRoundContext)
            or (run_id is not None and round_context.run_id != run_id)
            or (revision is not None and round_context.revision != revision)):
        raise ValueError('HIF round context belongs to another observation epoch')
    return round_context.round_number, round_context.base_turn_count


def notebook_identity_fields(round_number=1, *, layout='legacy'):
    if type(round_number) is not int or round_number not in (1, 2):
        raise ValueError('HIF notebook round is unsupported')
    if layout not in ('legacy', 'campaign_banner'):
        raise ValueError('HIF notebook layout is unsupported')
    fields = {**NOTEBOOK_IDENTITY_FIELDS,
              'round': (NOTEBOOK_IDENTITY_FIELDS['round'][0], f'ラウンド{round_number}')}
    if layout == 'campaign_banner':
        fields.update({name: (roi, fields[name][1]) for name, roi in NOTEBOOK_BANNER_FIELDS.items()})
    return fields


def observe_hif_notebook_identity(rows, *, round_number=1):
    """Read current round/identity and the explicit stamina denominator."""
    rows = tuple(rows)
    matches = []
    for layout in ('legacy', 'campaign_banner'):
        try:
            values = {name: _text(rows, pattern, roi)
                      for name, (roi, pattern) in notebook_identity_fields(round_number, layout=layout).items()}
            profile = notebook_hif_identity(values['song'], values['idol'])
        except ValueError:
            continue
        matches.append((values, profile))
    if len(matches) != 1:
        raise ValueError('HIF notebook identity layout is not uniquely observed')
    values, profile = matches[0]
    current, maximum = map(int, values['stamina'].split('/'))
    if not 0 <= current <= maximum < 100 or maximum == 0:
        raise ValueError('HIF notebook stamina is invalid')
    return profile.identity, profile.plan, current, maximum


def bind_recovered_hif_run(frames, *, run_id, revision=1, round_number=1):
    """Fresh notebook observations replace the retired process context.

    This produces metadata only. The host must close the notebook, revalidate
    the returned Live, and recollect every opening component before play.
    """
    frames = tuple(frames)
    if not frames or len({f['frame_id'] for f in frames}) != len(frames):
        raise ValueError('HIF recovery needs a current notebook observation')
    if any(not isinstance(f['frame_id'],str) or not f['frame_id'] for f in frames):
        raise ValueError('HIF recovery frame identity is missing')
    identities = [observe_hif_notebook_identity(f['rows'], round_number=round_number) for f in frames]
    if len(set(identities)) != 1:
        raise ValueError('HIF recovery observations disagree')
    idol, plan, _, maximum = identities[0]
    return HifLiveRunContext(run_id, revision, idol, plan, maximum)


def bind_hif_round_context(frames, *, run, round_number=1):
    """Bind the observed round text and identity from the same notebook page."""
    frames = tuple(frames)
    observed = bind_recovered_hif_run(frames, run_id=run.run_id, revision=run.revision,
                                      round_number=round_number)
    if observed != run:
        raise ValueError('HIF round notebook identity contradicts its run')
    return HifRoundContext(run.run_id, run.revision, round_number,
                           {1: 9, 2: 12}[round_number], tuple(f['frame_id'] for f in frames))


def _text(rows, pattern, region):
    x, y, width, height = region
    matches = []
    for row in rows:
        box = row.get('box')
        score = row.get('score')
        value = row.get('text')
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(number) is not int for number in box)
                or min(box[:2]) < 0 or min(box[2:]) <= 0
                or box[0] + box[2] > 720 or box[1] + box[3] > 1280
                or type(score) not in (int, float) or not math.isfinite(score)
                or not 0 <= score <= 1 or not isinstance(value, str)):
            raise ValueError('malformed HIF session observation')
        value = re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))
        if (score >= .95 and re.fullmatch(pattern, value)
                and x <= box[0] and y <= box[1]
                and box[0] + box[2] <= x + width
                and box[1] + box[3] <= y + height):
            matches.append(value)
    if len(matches) != 1:
        raise ValueError('HIF session field is not uniquely observed: ' + pattern)
    return matches[0]


def observe_hif_start_identity(rows):
    """The supported character and song must both appear in their title area."""
    _text(rows, r'2[.．]開始確認', (30, 15, 210, 65))
    _text(rows, r'プロデュース開始', (210, 1035, 310, 115))
    _text(rows, r'選択中の選抜試験メモリー', (10, 540, 340, 80))
    title = _text(rows, r'\[.+\]', (355, 104, 365, 55))
    idol = _text(rows, r'.+', (500, 145, 215, 90))
    profile = notebook_hif_identity(title[1:-1], idol)
    return profile.identity, profile.plan


def observe_hif_opening_stamina(rows):
    _text(rows, r'H[.]I[.]F[.]?本戦まで', (25, 30, 180, 50))
    _text(rows, r'6日', (40, 78, 145, 90))
    _text(rows, r'行動を選択してください', (210, 830, 290, 75))
    current, maximum = map(int, _text(rows, r'\d{1,3}/\d{1,3}', (290, 35, 175, 75)).split('/'))
    if not 0 <= current <= maximum <= 100 or maximum == 0:
        raise ValueError('new HIF stamina is invalid')
    return current, maximum


def bind_new_hif_run(start_frames, opening_frames, *, run_id, revision=1):
    """Bind two start identity frames and three opening frames, all independent.

    The host must confirm the start input succeeded, preserve game identity
    across that transition, and retire this context on exit/restart. This
    function produces identity/max stamina, never a complete strategy state.
    """
    start_frames, opening_frames = tuple(start_frames), tuple(opening_frames)
    if len(start_frames) != 2 or len(opening_frames) != 3:
        raise ValueError('new HIF requires two start and three opening frames')
    keys = [frame['frame_id'] for frame in start_frames + opening_frames]
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != 5:
        raise ValueError('new HIF evidence frames must be independent')
    identities = [observe_hif_start_identity(frame['rows']) for frame in start_frames]
    stamina = [observe_hif_opening_stamina(frame['rows']) for frame in opening_frames]
    if len(set(identities)) != 1 or len(set(stamina)) != 1:
        raise ValueError('new HIF opening observations disagree')
    idol, plan = identities[0]
    return HifLiveRunContext(run_id, revision, idol, plan, stamina[0][1])
