"""Resolve an unknown new hand by the existing free select/read/cancel path."""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, replace

TITLE_VISIT_VERSION = 5
HAND_VISITS_VERSION = 6


class InspectionTitles:
    """Resolve one complete title and its ordinary/upgraded family for this plan."""

    def __init__(self, catalog, plan, family=()):
        self.ids_by_title = {}
        for card_id, entry in catalog.entries.items():
            if entry.plan in ('free', plan) and (not family or card_id in family):
                self.ids_by_title.setdefault(self.key(entry.name), []).append(card_id)

    @staticmethod
    def key(text):
        from .upgrade import normalize_upgrade_title
        from .hif_single_plus import normalize_hif_title

        return normalize_upgrade_title(normalize_hif_title(text))

    def lookup(self, text):
        title = self.key(text)
        exact = self.ids_by_title.get(title, ())
        base = title.rstrip('+')
        ordinary,upgraded = self.ids_by_title.get(base, ()),self.ids_by_title.get(base+'+', ())
        if len(exact) != 1 or len(ordinary) > 1 or len(upgraded) > 1:
            return None,[]
        return exact[0],sorted((*ordinary,*upgraded),key=int)


def unresolved_hand(error):
    """A recognized hand with physical slots whose artwork needs a complete title."""
    events = getattr(error, 'events', ())
    if len(events) == 1 and events[0].get('phase') == 'hif_settlement_hand_pending':
        events = events[0].get('original_failure_events', ())
    if len(events) != 1 or events[0].get('phase') != 'inventory_live_identity':
        return None
    rows = events[0].get('predictions', ())
    unknown = [p for p in rows if p.get('accepted') is not True]
    if (not rows or not unknown or any(p.get('accepted') is not False or p.get('reason')
            not in ('below_acceptance_threshold', 'below_margin_threshold') for p in unknown)
            or [p.get('slot') for p in rows] != list(range(len(rows)))
            or any(not p.get('top_k_card_ids') or not p['top_k_card_ids'][0] for p in rows)):
        return None
    return [p['slot'] for p in unknown], [list(p['top_k_card_ids'][0]) for p in rows]


def unresolved_slot(error):
    """The original single-slot reader supports four- and five-card artwork."""
    hand = unresolved_hand(error)
    if hand and len(hand[0]) == 1 and len(hand[1]) in (4, 5):
        return hand[0][0], len(hand[1])
    return None


def inspect_unresolved_hand(context, params, error, event, module):
    """Read only unresolved physical slots; the submitted intent stays pending."""
    if getattr(context, 'gkh_hif_upstream_card_ui', False):
        hand = unresolved_hand(error)
        if hand is None:
            return False
        slots, families = hand
        if (event.get('event') != 'live_opening_state' or len(slots) > 1
                or 'hand_identity_inspection' in event or len(families) not in (4, 5)):
            return _inspect_upstream_hand(context, module, event, slots, families)
    proposal = unresolved_slot(error)
    if proposal is None or 'hand_identity_inspection' in event:
        return False
    slot, count = proposal
    retained = event.get('retained_hand_identity_inspection', {})
    if retained.get('accepted') is True and retained.get('slot') == slot:
        return False
    audit = dict(slot=slot, hand_count=count, business_submit_count=0,
                 input_click_count=0, accepted=False,
                 observed_live_context=asdict(context.gkh_hif_live_context))
    event['hand_identity_inspection'] = audit
    if event.get('event') == 'live_opening_state':
        audit['opening'] = True
    if getattr(context, 'gkh_hif_upstream_card_ui', False):
        audit['candidate_families'] = _candidate_families(error.events)
        if audit.get('opening'):
            _inspect_opening_title(context, audit, module)
            context.gkh_hif_current_hand_inspection = audit
            return True
    frames = module._capture_three_frames(context)
    selection, rows, _, stable = module._observe_live_selected_slot_consensus(
        context, frames, expected_selected_slot=None)
    if (not stable or selection is None or not selection.accepted
            or selection.selected_slot is not None or len(rows[-1]) != count):
        raise module.LiveSelectionError('hand inspection needs a fresh unselected row')
    # Historical reader only. The upstream path uses selected titles directly.
    if slot + 1 < count and not audit.get('opening'):
        from .hif_inspection_neighbor import build_reference

        prepared = context.gkh_hif_card_art
        predictions = tuple(module.classify_live_card_views(prepared.recognizer, frame, cards,
            frame_id=module.frame_identifier(frame), plan=context.gkh_hif_live_context.plan,
            acceptance_threshold=params['debug_identity_threshold'], min_margin=params['debug_identity_min_margin'],
            **module._live_card_art_profile(context, cards, None))
            for frame, cards in zip(frames, rows, strict=True))
        try:
            context.gkh_hif_inspection_neighbor_reference = build_reference(
                run_id=context.gkh_hif_live_context.run_id, revision=context.gkh_hif_live_context.revision,
                selected_slot=slot, frames=frames, candidates=rows, predictions=predictions,
                frame_ids=[module.frame_identifier(frame) for frame in frames])
        except ValueError:
            # Missing continuity never grants identity. The normal selected
            # reader must independently identify that neighbor instead.
            context.gkh_hif_inspection_neighbor_reference = None
    cards, deadline, preflight = module._selection_observation_preflight(
        context, rows[-1], None, slot, 1000.)
    audit['preflight'] = preflight
    if preflight.get('accepted') is not True:
        raise module.LiveSelectionError('hand inspection preflight failed')
    if not callable(getattr(context, 'gkh_hif_post_live_click', None)):
        from utils.maa_input import verify_maa_input_endpoints

        audit['endpoints'] = verify_maa_input_endpoints(context.tasker.controller.info,
            expected_client_size=context.gkh_hif_client_size)
    # Formal hosts verify every real endpoint at dispatch. No saved frame or
    # prior source timestamp is used to authorize this information-only click.
    expired, _ = module._validate_live_input_clock_window(deadline, preflight['authorization_floor'])
    if expired:
        raise module.LiveSelectionError('hand inspection frame expired')
    audit['input_click_count'] = 1
    try:
        clicked = module._click_live_candidate(context, module._live_selection_click_candidate(cards, slot),
            input_deadline=deadline, authorization_floor=preflight['authorization_floor'])
    except module._LiveInputNotDispatched:
        audit['input_click_count'] = 0
        raise
    if not clicked:
        raise module.LiveSelectionError('hand inspection selection failed')
    time.sleep(module.SELECTION_SETTLE_SECONDS)
    confirmation = module._capture_selected_live_confirmation(context, expected_hand_size=count, target_slot=slot)
    audit['confirmation'] = dict(accepted=confirmation['accepted'], events=confirmation['events'])
    if confirmation['accepted'] is not True:
        raise module.LiveSelectionError('hand inspection selection unconfirmed')
    try:
        audit['pending_inspection_reset'] = module._cancel_hif_live_selection(context, params, inspect_slot=slot)
    except module.LiveSelectionError as failure:
        audit['pending_inspection_reset'] = failure.events
        raise
    finally:
        audit['input_click_count'] += sum(p.get('input_click_count', 0)
            for p in audit.get('pending_inspection_reset', ()))
    audit['inspected_art_reference'] = context.gkh_hif_inspected_art
    audit['accepted'] = True
    context.gkh_hif_current_hand_inspection = audit
    return True


def _visits(audit):
    if not audit:
        return []
    if audit.get('version') == HAND_VISITS_VERSION:
        return [*audit.get('retained_visits', ()), *audit['visits']]
    return [audit] if audit.get('version') == TITLE_VISIT_VERSION else []


def _inspect_upstream_hand(context, module, event, slots, families):
    from .hif_upstream_ui import UpstreamHifCardUI

    current = event.get('hand_identity_inspection')
    retained = event.get('retained_hand_identity_inspection')
    previous = _visits(current) if current else _visits(retained)
    visited = {visit['slot'] for visit in previous}
    needed = [slot for slot in slots if slot not in visited]
    if not needed or current and current.get('accepted') is not True:
        return False
    if current is None or current.get('version') != HAND_VISITS_VERSION:
        current = dict(version=HAND_VISITS_VERSION, slot=needed[0], opening=event.get('event') == 'live_opening_state',
            hand_count=len(families), candidate_families=deepcopy(families), accepted=False,
            observed_live_context=asdict(context.gkh_hif_live_context), business_submit_count=0,
            input_click_count=0, visits=_visits(current), retained_visits=_visits(retained) if not current else [])
        event['hand_identity_inspection'] = current
    for visit in previous:
        current['candidate_families'][visit['slot']] = visit['candidate_families'][visit['slot']]
    current['accepted'] = False
    catalog = module.SkillCardFeatureCatalog.load(module.CATALOG_PATH)
    started = time.monotonic()
    for index, slot in enumerate(needed):
        if index and time.monotonic() - started >= UpstreamHifCardUI.TIME_OUT:
            break  # fb8 stops probing after this visit budget and returns to its loop.
        visit = dict(slot=slot, opening=current['opening'], hand_count=len(families),
            candidate_families=deepcopy(current['candidate_families']), weak_candidate_family=families[slot],
            input_click_count=0, business_submit_count=0, observed_live_context=current['observed_live_context'],
            accepted=False)
        current['visits'].append(visit)
        visit['candidate_families'][slot] = []  # The selected title decides the weak slot's family.
        try:
            read_upstream_opening_slot(context, visit, module, catalog)
            visit['accepted'] = True
            verify_upstream_inspection(visit, context.gkh_hif_live_context, catalog)
            current['candidate_families'][slot] = visit['candidate_families'][slot]
        finally:
            current['input_click_count'] = sum(item['input_click_count'] for item in current['visits'])
    current['accepted'] = True
    context.gkh_hif_current_hand_inspection = current
    return True


def restore_hand_inspection(context, event, module, *, source=None, pending=False):
    audit = event.get('hand_identity_inspection', event.get('retained_hand_identity_inspection'))
    if audit is None and pending and source is not None:
        # An inline upstream card publishes its observer event, whereas a cold
        # observer also writes the top-level settlement alias. Reuse either.
        observed = [item for item in source.get('live_events', ())
            if item.get('event') == 'live_committed_state' and item.get('hand_identity_inspection')]
        if len(observed) > 1:
            raise ValueError('pending hand inspection has conflicting observer events')
        if observed:
            audit = observed[0]['hand_identity_inspection']
    if audit is None:
        return
    version = audit.get('version')
    if version == HAND_VISITS_VERSION:
        _restore_upstream_hand(context, module, audit, source, pending)
        return
    if version != TITLE_VISIT_VERSION:
        raise ValueError('unsupported HIF hand inspection receipt')
    if audit.get('opening') is not True or audit.get('accepted') is not True:
        raise ValueError('upstream opening inspection is incomplete')
    catalog = module.SkillCardFeatureCatalog.load(module.CATALOG_PATH)
    run = context.gkh_hif_live_context
    verify_upstream_inspection(audit, run, catalog)
    card_id = audit['detail']['card_id']
    matched, family = InspectionTitles(catalog, run.plan).lookup(catalog.entries[card_id].name)
    if matched != card_id or family != audit['candidate_families'][audit['slot']]:
        raise ValueError('restored upstream opening title family differs from the catalog')
    context.gkh_hif_current_hand_inspection = audit


def restore_pending_drink_opening_inspection(context, checkpoint, module):
    """A submitted drink can preserve the original physical hand through settlement."""
    from .high_score.contracts import ActionKind

    if (checkpoint.get('pending') is not True or checkpoint['action'].kind is not ActionKind.USE_DRINK
            or getattr(context, 'gkh_hif_current_hand_inspection', None) is not None
            or getattr(context, 'gkh_hif_pending_hand_inspection', None) is not None):
        return
    opening = checkpoint['event']
    audit = opening.get('hand_identity_inspection', opening.get('retained_hand_identity_inspection'))
    if (opening.get('event') != 'live_opening_state' or not isinstance(audit, dict) or not audit.get('opening')
            or audit.get('version') not in (TITLE_VISIT_VERSION, HAND_VISITS_VERSION) or opening['core']['families'] != audit['candidate_families']):
        return
    before = checkpoint['bundle'].state
    expected = checkpoint['expected']
    results = expected if isinstance(expected, tuple) else (expected,)
    if (not results or any(after.turn != before.turn or after.deck.hand != before.deck.hand for after in results)
            or [asdict(ref) for ref in before.deck.hand] != opening['state']['deck']['hand']):
        return
    restore_hand_inspection(context, opening, module)
    context.gkh_hif_current_hand_inspection = dict(context.gkh_hif_current_hand_inspection,
        continuity_scope='verified_same_hand_business_state')


def _candidate_families(events):
    events = events[0].get('original_failure_events', events) if events else ()
    rows = events[0].get('predictions', ()) if events else ()
    if len(rows) not in (4, 5) or [p.get('slot') for p in rows] != list(range(len(rows))):
        raise ValueError('inspection lacks its original hand candidates')
    return [list(p['top_k_card_ids'][0]) for p in rows]


def clear_upstream_selection(context, module, *, events=None):
    from .hif_upstream_ui import UpstreamHifCardUI

    step = dict(phase='selection_reset', navigation_scope='pending_preview_only',
        accepted=False, input_click_count=0, after=dict(selected_slot=None))
    if events is not None:
        events.append(step)
    def saved(receipt):
        step['upstream_ui'] = receipt
        step['input_click_count'] = sum(p.get('dispatched') is True for p in receipt['inputs'])
    ui = UpstreamHifCardUI(context, frame_identifier=module.frame_identifier, sink=saved)
    step['accepted'] = ui.clear_selection()
    if not step['accepted']:
        raise module.LiveSelectionError('upstream selection cancellation remains unresolved', events=[step])
    return step


def _inspect_opening_title(context, audit, module):
    """Run fb8's select/read/neutral-clear visit on the unknown physical slot."""
    slot = audit['slot']
    audit['weak_candidate_family'] = audit['candidate_families'][slot]
    audit['candidate_families'][slot] = []
    catalog = module.SkillCardFeatureCatalog.load(module.CATALOG_PATH)
    read_upstream_opening_slot(context, audit, module, catalog)
    audit['accepted'] = True
    try:
        verify_upstream_inspection(audit, context.gkh_hif_live_context, catalog)
    except ValueError:
        audit['accepted'] = False
        raise


def read_upstream_opening_slot(context, audit, module, catalog, *, returned_frames=None):
    """Run the upstream preview, complete-title read, neutral clear and fresh hand recheck."""
    from .hif_upstream_ui import UpstreamHifCardUI
    from .hif_upstream_policy import _stable_hand_geometry

    slot = audit['slot']
    audit['version'] = TITLE_VISIT_VERSION
    def save_selection(receipt):
        audit['upstream_selection'] = receipt
        audit['input_click_count'] = sum(p.get('dispatched') is True for p in receipt['inputs'])

    ui = UpstreamHifCardUI(context, frame_identifier=module.frame_identifier, sink=save_selection)
    run = context.gkh_hif_live_context
    ui.inspection_titles = InspectionTitles(catalog, run.plan, audit['candidate_families'][slot])
    image, boxes, slots = _stable_hand_geometry(module, ui, hand_size=audit['hand_count'], target_slot=slot)
    ui._hif_bind_hand(boxes)
    index = slots.index(slot)
    ui.receipt['action'] = dict(kind='inspect_hand', slot=slot)
    audit['physical_slots'] = list(slots)
    audit['bound_centers'] = list(ui._hif_hand_geometry)
    audit['bound_boxes'] = [list(box) for box, _ in boxes]
    try:
        if ui._hif_select_card(ui.context, image, index) is None:
            raise ValueError('upstream hand inspection selection unconfirmed')
        _read_bound_title(ui, audit, index)
    finally:
        ui.publish()
        audit['pending_inspection_reset'] = []
        try:
            clear_upstream_selection(context, module, events=audit['pending_inspection_reset'])
        finally:
            audit['input_click_count'] += sum(p['input_click_count'] for p in audit['pending_inspection_reset'])
    fresh = _returned_hand_frame(ui, audit)
    if returned_frames is not None:
        returned_frames.append(fresh)


def _returned_hand_frame(ui, visit):
    image = ui.context.tasker.controller.post_screencap().wait().get()
    returned = ui._hif_frame_boxes(ui.context, image)
    if not ui._hif_same_geometry(returned):
        raise ValueError('HIF opening hand changed after title inspection')
    visit['returned_centers'] = list(ui._hif_geometry(returned))
    return image


def _read_bound_title(ui, visit, index):
    card_id, title = ui._hif_read_detail(ui.context, index)
    if card_id is None or not title:
        raise ValueError('upstream hand inspection full title is unresolved')
    rows = ui.receipt['detail_observations'][-2:]
    visit['candidate_families'][visit['slot']] = ui.inspected_family
    visit['detail'] = dict(accepted=True, card_id=card_id,
        observations=[dict(row, card_id=card_id) for row in rows])


def _restore_upstream_hand(context, module, audit, source, pending):
    run = context.gkh_hif_live_context
    catalog = module.SkillCardFeatureCatalog.load(module.CATALOG_PATH)
    audit = deepcopy(audit)
    visits = _visits(audit)
    slots = [visit['slot'] for visit in visits]
    if (audit.get('observed_live_context') != asdict(run) or audit.get('business_submit_count') != 0
            or audit.get('hand_count') != len(audit['candidate_families'])
            or len(slots) != len(set(slots)) or any(type(i) is not int or not 0 <= i < audit['hand_count'] for i in slots)
            or audit.get('input_click_count') != sum(v['input_click_count'] for v in audit['visits'])):
        raise ValueError('upstream hand inspection differs from its recorded hand or inputs')
    for visit in visits:
        if visit.get('version') != TITLE_VISIT_VERSION:
            raise ValueError('unsupported HIF hand inspection receipt')
        if visit.get('accepted') is True:
            verify_upstream_inspection(visit, run, catalog)
            if audit['candidate_families'][visit['slot']] != visit['candidate_families'][visit['slot']]:
                raise ValueError('upstream hand title differs from the recorded slot')
            continue
        if (not pending or not source or audit.get('accepted') is not False or not audit['visits']
                or visit is not audit['visits'][-1]):
            raise ValueError('upstream hand inspection is incomplete')
        selection = visit.get('upstream_selection', {}).get('inputs', ())
        resets = visit.get('pending_inspection_reset', ())
        dispatches = source.get('native_input_dispatches', ())
        for proof in [*selection, *(p for r in resets for p in r.get('upstream_ui', {}).get('inputs', ()))]:
            index = proof.get('dispatch_index')
            if type(index) is not int or index < 0:
                raise ValueError('upstream inspection lacks its original dispatch index')
            if index >= len(dispatches) and proof.get('dispatched') is False:
                continue
            if (index >= len(dispatches) or dispatches[index].get('kind', 'click') != 'click'
                    or any(proof.get(k) != dispatches[index].get(k) for k in ('x', 'y'))):
                raise ValueError('upstream inspection differs from its original dispatch')
        if resets and resets[-1].get('accepted') is True:
            verify_upstream_clear(resets[-1])
        elif any(p['dispatch_index'] < len(dispatches) for p in selection):
            context.gkh_hif_pending_hand_inspection = audit
            return
        # An unread, closed (or never opened) visit supplies no card identity.
        audit['visits'].pop()
        audit['input_click_count'] = sum(v['input_click_count'] for v in audit['visits'])
    audit['accepted'] = True
    if _visits(audit):
        context.gkh_hif_current_hand_inspection = audit


def _finish_bound_visit(context, audit, module):
    """Continue a recorded physical preview, without ever issuing a selection."""
    from .hif_upstream_ui import UpstreamHifCardUI

    run = context.gkh_hif_live_context
    catalog = module.SkillCardFeatureCatalog.load(module.CATALOG_PATH)
    visit = audit['visits'][-1]
    ui = UpstreamHifCardUI(context, frame_identifier=module.frame_identifier, sink=lambda _: None)
    ui._hif_bind_hand([(box, None) for box in visit['bound_boxes']])
    index = visit['physical_slots'].index(visit['slot'])
    ui.inspection_titles = InspectionTitles(catalog, run.plan)
    image = ui.context.tasker.controller.post_screencap().wait().get()
    boxes = ui._hif_frame_boxes(ui.context, image)
    try:
        if not visit.get('detail') and ui._hif_selected_index(ui.context, image, boxes) == index:
            _read_bound_title(ui, visit, index)
    finally:
        audit['pending_inspection_reset'] = []
        clear_upstream_selection(context, module, events=audit['pending_inspection_reset'])
    if visit.get('detail'):
        _returned_hand_frame(ui, visit)
        visit['pending_inspection_reset'] = audit['pending_inspection_reset']
        visit['accepted'] = True
        verify_upstream_inspection(visit, run, catalog)
        audit['candidate_families'][visit['slot']] = visit['candidate_families'][visit['slot']]
    else:
        audit['visits'].pop()
    # Old selection inputs and the current reset stay on their own host rows.
    audit['retained_visits'] = _visits(audit)
    audit.update(visits=[], accepted=True, input_click_count=0)


def finish_upstream_inspection(context, audit, module):
    """Continue an already-dispatched current preview without selecting it again."""
    if audit.get('version') != HAND_VISITS_VERSION:
        raise ValueError('unsupported HIF hand inspection receipt')
    _finish_bound_visit(context, audit, module)


def verify_upstream_inspection(audit, run, catalog):
    from .hif_upstream_ui import UPSTREAM_REVISION

    detail = audit.get('detail', {})
    rows = detail.get('observations', ())
    family = audit['candidate_families'][audit['slot']]
    resets = audit.get('pending_inspection_reset', ())
    if (audit.get('version') != TITLE_VISIT_VERSION or audit.get('accepted') is not True
            or audit.get('business_submit_count') != 0 or audit.get('observed_live_context') != asdict(run)
            or len(audit['candidate_families']) != audit['hand_count']
            or detail.get('accepted') is not True or detail.get('card_id') not in family
            or len(rows) != 2 or any(not r.get('frame_id') for r in rows)
            or len(resets) != 1 or resets[0].get('accepted') is not True
            or resets[0].get('upstream_ui', {}).get('revision') != UPSTREAM_REVISION
            or resets[0]['upstream_ui'].get('cleared') is not True):
        raise ValueError('upstream inspected title is incomplete')
    verify_upstream_clear(resets[0])
    expected = InspectionTitles.key(catalog.entries[detail['card_id']].name)
    from .hif_single_plus import normalize_hif_title

    names = set()
    for row in rows:
        texts = row.get('ocr', ())
        full_title = ''.join(text.get('text', '') for text in texts)
        if (row.get('card_id') != detail['card_id'] or not texts
                or InspectionTitles.key(full_title) != expected):
            raise ValueError('upstream inspected complete title changed')
        names.add(normalize_hif_title(full_title))
    if len(names) != 1:
        raise ValueError('upstream inspected complete title changed')
    return names.pop()


def verify_upstream_clear(step):
    from .hif_upstream_ui import UPSTREAM_REVISION

    receipt = step.get('upstream_ui', {})
    observations = receipt.get('clear_observations', ())[-2:]
    inputs = receipt.get('inputs', ())
    if (receipt.get('revision') != UPSTREAM_REVISION or receipt.get('cleared') is not True
            or step.get('accepted') is not True or len(observations) != 2
            or any(not r.get('frame_id') for r in observations)
            or any(r.get('selected') is not False or r.get('has_cards') is not True for r in observations)
            or step.get('input_click_count') != len(inputs)
            or any(p.get('phase') != 'cancel_selection' or p.get('dispatched') is not True
                or (p.get('x'), p.get('y')) != (5, 5)
                or p.get('input_deadline', 0)-p.get('captured_at', 0) != 1. for p in inputs)):
        raise ValueError('upstream neutral cancellation was not confirmed')
    return inputs


def corroborate_upstream_inspection(context, predictions, evidence):
    """Carry a complete title through neutral navigation in the same hand only."""
    audit = getattr(context, 'gkh_hif_current_hand_inspection', None)
    if not isinstance(audit, dict) or audit.get('version') not in (TITLE_VISIT_VERSION, HAND_VISITS_VERSION):
        return predictions
    families = audit['candidate_families']
    visits = {visit['slot']: visit for visit in _visits(audit)}
    if (audit.get('accepted') is not True or audit.get('observed_live_context') != asdict(context.gkh_hif_live_context)
            or len(predictions) != len(families) or any(p.slot != i or not p.top_k_card_ids
                or (audit.get('version') != HAND_VISITS_VERSION and i not in visits and list(p.top_k_card_ids[0]) != families[i])
                for i, p in enumerate(predictions))):
        return predictions
    result = list(predictions)
    for slot, visit in visits.items():
        current = predictions[slot]
        different = tuple(current.top_k_card_ids[0]) != tuple(families[slot])
        if audit.get('version') == TITLE_VISIT_VERSION and current.accepted and different:
            raise ValueError('HIF opening hand changed after title inspection')
        observed_title_over_art = (audit.get('version') == HAND_VISITS_VERSION and visit.get('version') == TITLE_VISIT_VERSION
            and different and current.accepted and current.reason == 'accepted_visual_family')
        if not observed_title_over_art and (current.accepted or current.reason not in (
                'below_acceptance_threshold', 'below_margin_threshold')):
            continue
        evidence.append(dict(scope='upstream_inspected_complete_title', slot=slot,
            family=families[slot], frame_id=current.frame_id, detail=visit['detail'], upgrade_verified=False,
            original_prediction=current.to_dict(),
            continuity_scope=audit.get('continuity_scope', 'no_business_input_since_inspection')))
        result[slot] = replace(current, accepted=True, reason='upstream_inspected_complete_title')
    return tuple(result)
