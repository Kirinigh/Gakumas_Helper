"""Identify a HIF opening from its current page or its automatic route context."""
import re
from dataclasses import asdict

from .hif_visual import HifLiveRunContext
from .hif_session import HifRoundContext

SOURCE = 'route_handoff_and_opening_turns'
UNREAD_INVENTORY_SOURCES = ('unbound_notebook', 'unbound_route_entry')
# fb8fd00 battle_recognition.py, _detect_battle_number / first _read_turn_count.
ENTRY_TURNS = {9: 1, 12: 2, 11: 2, 10: 2}


def read_entry_turn(context, image):
    """Use the pinned first-read ROI and primary/fallback OCR in horizontal order."""
    node = 'ProduceRecognitionScore'
    for only_rec, threshold in ((True, .9), (False, .3)):
        config = dict(recognition='OCR', roi=[34, 68, 56, 44], expected=r'^[1-9]\d?$',
                      only_rec=only_rec, threshold=threshold)
        try:
            result = context.run_recognition(node, image, pipeline_override={node: config})
            if result and result.hit and result.filtered_results:
                rows = sorted(result.filtered_results, key=lambda row: row.box[0] if row.box else 0)
                candidates = [int(s) for row in rows for s in re.findall(r'\d{1,2}', row.text or '')]
                digits = [s for row in rows for s in re.findall(r'\d', row.text or '')]
                if len(digits) == 2:
                    candidates.append(int(''.join(digits)))
                return max(candidates, default=None)
        except Exception:
            # The alternate source is optional even when the SDK reader fails.
            return None
    return None


def current_opening_round(context, image):
    """Recognize the playable opening, on the same screenshot as the turn count."""
    from .hif_upstream_ui import HIF_SKIP_RECOGNITION

    turns = read_entry_turn(context, image)
    if turns not in (9, 12):
        return None
    skip_node = 'ProduceRecognitionSkipRound'
    skip = context.run_recognition(skip_node, image, pipeline_override={skip_node: HIF_SKIP_RECOGNITION})
    if not (skip and skip.hit):
        cards = context.run_recognition('ProduceRecognitionCards', image)
        if not (cards and cards.hit):
            return None
    return ENTRY_TURNS[turns]


def entry_metadata(event, identity=None):
    """Rebuild a zero-navigation receipt from its route provenance and actual Live frames."""
    proof = event['entry_identity']
    number, pid = proof['round_number'], event['game_pid']
    groups = event['observations']
    observed, = groups
    run = HifLiveRunContext(**event['context'])
    if (event['round_source'] != SOURCE or event['accepted'] is not True
            or event['returned_to_live'] is not False or event['input_click_count'] != 0 or event['inputs'] != []
            or type(pid) is not int or pid != proof['game_pid'] or type(number) is not int or number not in (1, 2)
            or type(proof['ledger_revision']) is not int or type(proof['ownership_revision']) is not int
            or proof['handoff_id'] != f"{proof['run_id']}:r{number}:ledger-{proof['ledger_revision']}"
            or proof['identity_source'] != {1: 'cultivation_start', 2: 'recovery_identity'}[number]
            or (run.idol, run.plan, run.max_stamina) != (proof['idol'], proof['plan'], proof['max_stamina'])
            or not observed or any(row['phase'] != 'live' or type(row['entry_turns']) is not int
                or ENTRY_TURNS.get(row['entry_turns']) != number or row['turns'] != row['entry_turns']
                or not 0 <= row['stamina'] <= run.max_stamina for row in observed)):
        raise ValueError('HIF opening turn observations differ from the route entry')
    for field in ('source_frame_ids', 'identity_frame_ids', 'max_stamina_frame_ids'):
        frames = proof[field]
        if not frames or len(set(frames)) != len(frames) or any(
                not re.fullmatch(rf'route-frame-{pid}-\d+', frame) for frame in frames):
            raise ValueError('HIF opening round lacks route source frames')
    if identity is not None and (identity['route_run_id'], identity['handoff_id'], identity['ledger_revision'],
            identity['game_pid'], identity['ownership_revision']) != (
                proof['run_id'], proof['handoff_id'], proof['ledger_revision'], pid, proof['ownership_revision']):
        raise ValueError('HIF round evidence belongs to another route handoff')
    return HifRoundContext(run.run_id, run.revision, number, {1: 9, 2: 12}[number],
                           tuple(row['frame_id'] for row in observed))


def bind_entry(context, proof, batch, entry_turns, *, run_id, revision, game_pid):
    if not proof or len(entry_turns) != len(batch):
        return None
    try:
        run = HifLiveRunContext(run_id, revision, proof['idol'], proof['plan'], proof['max_stamina'])
        event = dict(phase='hif_context_recovery', round_source=SOURCE, entry_identity=proof,
                     accepted=True, returned_to_live=False, input_click_count=0, inputs=[], game_pid=game_pid,
                     context=asdict(run), observations=[[dict(frame, entry_turns=turn)
                        for frame, turn in zip(batch, entry_turns, strict=True)]])
        metadata = entry_metadata(event)
    except (KeyError, TypeError, ValueError):
        return None
    event['round_context'] = asdict(metadata)
    context.gkh_hif_live_context = run
    context.gkh_hif_round_context = metadata
    context.gkh_hif_context_game_pid = game_pid
    return run, event


def verified_entry_receipt(row, event):
    if event.get('round_source') != SOURCE:
        return False
    try:
        metadata = entry_metadata(event, row['automatic_battle']['identity'])
        saved = HifRoundContext(**event['round_context'])
        return (metadata == saved and event['game_pid'] == row['endpoints']['game']['pid']
                and event['entry_identity']['handoff_id'] == row['handoff_id'])
    except (KeyError, TypeError, ValueError):
        return False
