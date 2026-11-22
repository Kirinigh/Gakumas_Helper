"""Locate the strategy's retrieval target and confirm its observed slot through Maa."""

import time
from dataclasses import asdict, dataclass

from .hif_choice_panel import (
    CHOICE_BOXES,
    read_hif_choice_panel,
    confirm_hif_choice_title,
    observe_hif_choice_panel,
    bind_hif_choice_candidates,
    resolve_hif_choice_detail_evidence,
)
from .hif_choice_execution import (
    _dispatch,
    choice_target_binding,
    select_hif_choice_card,
    confirm_hif_choice_card,
)
from .inventory_customization import measure_customization_badge, stable_customization_badge_presence


@dataclass(frozen=True)
class HifChoiceCapture:
    panel: dict
    frames: tuple
    frame_ids: tuple


def _frame_identifier(image, index):
    try:
        from agent.card_selection.model import frame_identifier
    except ModuleNotFoundError:
        from card_selection.model import frame_identifier
    return frame_identifier(image, index)


def _one(capture_frame):
    raw = capture_frame()
    if isinstance(raw, tuple) and len(raw) == 2 and type(raw[1]) in (int, float):
        return raw
    return raw, time.perf_counter()


def _transient(error):
    message = str(error)
    return message.startswith(('HIF choice grid is incomplete',
        'HIF choice list extent is not completely visible',
        'HIF choice selected marker', 'HIF choice confirmation button',
        'HIF choice selected detail title', 'HIF choice pages have not settled',
        'HIF choice detail visual family is unresolved',
        'HIF choice detail has not reached the requested slot',
        'HIF choice detail title changes across frames',
        'inventory page anchor is missing or ambiguous: 移動',
        'inventory page anchor is missing or ambiguous: スキルカードが選択されていません'))


def capture_hif_choice_panel(context, projected, reader, *, capture_frame,
                            detail_catalog, epoch, detail_evidence=(),
                            frame_identifier=_frame_identifier, max_attempts=3,
                            expected_slot=None):
    """Read the current choice page. Retry if animation obscures its controls."""
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError('HIF choice settling budget must be one to three frame sets')
    for attempt in range(max_attempts):
        frames = (_one(capture_frame)[0],)
        frame_ids = tuple(frame_identifier(image, index) for index, image in enumerate(frames))
        try:
            panel = observe_hif_choice_panel(context, frames, reader, projected,
                frame_ids=frame_ids, detail_evidence=detail_evidence,
                detail_catalog=detail_catalog, detail_epoch=epoch)
            if expected_slot is not None and panel.get('selected_slot') != expected_slot:
                raise ValueError('HIF choice detail has not reached the requested slot')
            return HifChoiceCapture(panel, frames, frame_ids)
        except ValueError as error:
            family_settling = (expected_slot is not None
                and str(error) == 'HIF choice detail no longer matches the visible family')
            if attempt == max_attempts-1 or not (_transient(error) or family_settling):
                raise
            time.sleep(.08)
    raise ValueError('HIF choice settling budget exhausted')


def _family(panel, slot):
    groups = {row['visual_group'] for row in panel['identities'][slot]['observations']}
    if len(groups) != 1 or None in groups:
        raise ValueError('HIF choice target has no stable visual family')
    return next(iter(groups))


def _fresh_target(context, image, panel, reader, slot, detail_catalog):
    """Current mandatory page and physical target, without repeating 18 reads.

    The complete candidate ledger is retained only within the caller's unchanged
    pending epoch. Page extent, selection, target family, upgrade when visible,
    customization presence and current selected title are independently reread.
    """
    previous_boxes = panel['pages'][0]['boxes']
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    options = {'target_slot': slot} if upstream else {}
    if panel['pages'][0].get('scroll_offset') is not None:
        options['scroll_offset'] = panel['pages'][0]['scroll_offset']
    page = read_hif_choice_panel(context, image, expected_count=len(previous_boxes), **options)
    boxes = page['boxes']
    if not upstream and (previous_boxes != boxes or page['selected_slot'] != panel.get('selected_slot')
                         or page['selected_count'] != panel['selected_count']):
        raise ValueError('HIF choice page changed before target input')
    observed = reader.identify(image, boxes[slot], plan='sense')
    previous = panel['identities'][slot]
    if ((observed.visual_group != _family(panel, slot)
            and not (upstream and page['selected_slot'] == slot and observed.card_id is None))
            or (observed.card_id is not None and previous['card_id'] is not None
                and observed.card_id != previous['card_id'])):
        raise ValueError('HIF choice target identity changed before input')
    badge = stable_customization_badge_presence(
        (measure_customization_badge(image, boxes[slot]),), required_frames=1).state.value
    if badge != panel['badge_presence'][slot]:
        raise ValueError('HIF choice target customization changed before input')
    selected = page['selected_slot']
    if selected is not None:
        canonical = panel['identities'][selected]['card_id']
        if canonical is not None:
            confirm_hif_choice_title(detail_catalog,
                '\n'.join(row['text'] for row in page['detail_title_rows']),
                candidate_card_ids=(int(canonical),))
    result = dict(panel, pages=(page,), single_frame=True,
        frame_ids=(_frame_identifier(image, 0),),
        selected_slot=selected, selected_count=page['selected_count'], unselected=not page['selected_count'],
        confirm_enabled=page['confirm_enabled'], confirm_target=page['confirm_target'])
    if upstream:
        result['observed_target_slot'] = slot
    if panel.get('accepted') is True:
        if upstream:
            result['candidates'] = [dict(row, box=boxes[slot]) if row['slot'] == slot else row
                                    for row in panel['candidates']]
        result['selected_instance_id'] = next((row['instance_id'] for row in panel['candidates']
                                               if row['slot'] == selected), None)
    return result


def _selected_target(context, panel, reader, slot, catalog, capture_frame, frame_identifier):
    """miyu _read_move_title_stable: poll the chosen slot and its visible title."""
    def read_current(image):
        try:
            return _fresh_target(context, image, panel, reader, slot, catalog)
        except ValueError:
            return None  # The previous title or thumbnail may still be visible.

    return _poll_move_title(context, slot, capture_frame, frame_identifier, read_current)


def _poll_move_title(context, slot, capture_frame, frame_identifier, read_current):
    deadline = time.monotonic() + 2.5
    pages, ids, times = [], [], []
    last, failure = None, None
    while time.monotonic() < deadline and getattr(getattr(context, 'tasker', None), 'stopping', False) is not True:
        time.sleep(.2)
        image, captured_at = _one(capture_frame)
        try:
            current = read_current(image)
        except ValueError as error:
            if not _transient(error):
                raise
            failure, current = error, None
        if current is None or current['selected_slot'] != slot:
            pages.clear()
            ids.clear()
            times.clear()
            last = None
            continue
        title = tuple(row['text'] for row in current['pages'][0]['detail_title_rows'])
        if pages and tuple(row['text'] for row in pages[-1]['detail_title_rows']) != title:
            pages.clear()
            ids.clear()
            times.clear()
        last = current
        pages.append(current['pages'][0])
        ids.append(frame_identifier(image, len(ids)))
        times.append(captured_at)
        if len(pages) == 2:
            break
    if last is None:
        if failure is not None:
            raise failure
        raise ValueError('HIF retrieval no longer shows the saved selected card')
    return dict(last, pages=tuple(pages), frame_ids=tuple(ids), title_capture_monotonic=tuple(times),
                single_frame=len(pages) == 1)


def _capture_detail(context, projected, panel, reader, slot, *, detail_catalog,
                    epoch, capture_frame, frame_identifier):
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    if upstream:
        selected = _selected_target(context, panel, reader, slot, detail_catalog, capture_frame, frame_identifier)
    else:
        def read_current(image):
            boxes = panel['pages'][0]['boxes']
            page = read_hif_choice_panel(context, image, expected_count=len(boxes))
            if (page['boxes'] != boxes or page['selected_slot'] != slot or page['selected_count'] != 1
                    or page['confirm_enabled'] is not True):
                raise ValueError('HIF choice detail has not reached the requested slot')
            identity = reader.identify(image, boxes[slot], plan='sense')
            if identity.visual_group is None:
                raise ValueError('HIF choice detail visual family is unresolved')
            if identity.visual_group != _family(panel, slot):
                raise ValueError('HIF choice detail thumbnail changed visual family')
            return dict(selected_slot=slot, pages=(page,))

        selected = _poll_move_title(context, slot, capture_frame, frame_identifier, read_current)

    pages = selected['pages']
    evidence = resolve_hif_choice_detail_evidence(detail_catalog, reader, projected,
        slot=slot, box=pages[0]['boxes'][slot], visual_group=_family(panel, slot),
        frame_ids=selected['frame_ids'], title_rows=tuple(page['detail_title_rows'] for page in pages),
        observation_epoch=epoch, capture_monotonic=selected['title_capture_monotonic'], upstream=upstream)
    return evidence, pages


def collect_hif_choice_details(context, projected, reader, *, detail_catalog, epoch,
                               capture_frame, journal,
                               detail_evidence=(), frame_identifier=_frame_identifier):
    """Reusable pending recovery: gather only missing titles, without confirming.

    Known details can span Maa hosts when the host has independently preserved
    the same game/pending epoch and no selection confirmation occurred.
    """
    if not isinstance(epoch, str) or not epoch:
        raise ValueError('HIF choice detail collection needs a bound pending epoch')
    details = list(detail_evidence)
    captured = capture_hif_choice_panel(context, projected, reader, capture_frame=capture_frame,
        detail_catalog=detail_catalog, epoch=epoch, detail_evidence=details,
        frame_identifier=frame_identifier)
    panel = captured.panel
    if panel['accepted']:
        return dict(panel=panel, detail_evidence=tuple(details))
    slots = tuple(panel['unresolved_slots'])
    if len(set(slots)) != len(slots) or len(slots) > len(CHOICE_BOXES):
        raise ValueError('HIF choice unresolved slot set is invalid')
    for slot in slots:
        if panel.get('selected_slot') != slot:
            image, captured_at = _one(capture_frame)
            fresh = _fresh_target(context, image, panel, reader, slot, detail_catalog)
            target = fresh['pages'][0]['boxes'][slot]
            navigation = dict(slot=slot, choice_submit_count=0, selection_click_count=0)

            def navigation_journal(stage, payload):
                journal('pending_ui_captured', dict(detail_stage=stage,
                    detail_navigation=payload['choice'], detail_evidence=tuple(details)))

            if fresh.get('selected_slot') != slot:
                _dispatch(context, target, captured_at, navigation, navigation_journal, 'choice_detail_select')
                navigation['selection_click_count'] = 1
                navigation_journal('choice_detail_selected', {'choice': navigation})
        evidence, pages = _capture_detail(context, projected, panel, reader, slot,
            detail_catalog=detail_catalog, epoch=epoch, capture_frame=capture_frame,
            frame_identifier=frame_identifier)
        details.append(evidence)
        identities = [dict(row) for row in panel['identities']]
        identities[slot].update(card_id=evidence['card_id'], detail_evidence=evidence)
        panel = dict(panel, pages=pages, identities=tuple(identities),
            selected_slot=slot, selected_count=1, confirm_enabled=True, unselected=False)
        journal('pending_ui_captured', dict(detail_stage='detail_observed',
            detail_evidence=tuple(details), selected_slot=slot))
    if getattr(context, 'gkh_hif_upstream_card_ui', False) is True and slots:
        # Selecting a detail changes the selection, not the catalogue. Combine
        # the original thumbnail reads with the resolved titles from this page.
        ids = tuple(row['card_id'] for row in panel['identities'])
        binding = bind_hif_choice_candidates(projected, ids, pages[-1]['boxes'], badge_presence=panel['badge_presence'])
        panel.update(binding, accepted=True, unresolved_slots=(), preview_targets=(), single_frame=len(evidence['frame_ids']) == 1,
            selected_instance_id=binding['candidates'][slot]['instance_id'], frame_ids=evidence['frame_ids'],
            selected_count_source=pages[-1]['selected_count_source'], confirm_target=pages[-1]['confirm_target'])
    else:
        panel = capture_hif_choice_panel(context, projected, reader, capture_frame=capture_frame,
            detail_catalog=detail_catalog, epoch=epoch, detail_evidence=details,
            frame_identifier=frame_identifier).panel
    if not panel['accepted']:
        raise ValueError('HIF choice identities remain unresolved after bounded detail collection')
    journal('pending_ui_captured', dict(detail_stage='complete_candidates',
        detail_evidence=tuple(details), choice_panel=panel))
    return dict(panel=panel, detail_evidence=tuple(details))


def _planned_target_panel(context, projected, instance, reader, capture_frame, frame_identifier, catalog,
                          *, journal=None, previous_scrolls=()):
    """Stop at the first visible match when the retained deck contains one such copy."""
    state = projected.pending_state
    ids = state.pending_choice.candidate_instance_ids
    if instance not in ids:
        return None
    permanent = {key: (card_id, mods) for key, card_id, mods in projected.source_bundle.inventory_keys}
    if not all(key in permanent for key in ids):
        return None
    card_id, mods = permanent[instance]
    candidate_ids = [permanent[key][0] for key in ids]
    if candidate_ids.count(card_id) != 1:
        return None  # Distinct copies can differ by zone, position or customization.
    image, _ = _one(capture_frame)
    from .hif_choice_scroll import low_move_layout, scan_move_viewports

    scrollable = bool(previous_scrolls) or low_move_layout(image)
    if scrollable:
        count = len(ids)
        views = scan_move_viewports(context, image, count, capture=lambda: _one(capture_frame)[0],
            identify_frame=frame_identifier, journal=journal, previous=previous_scrolls)
    else:
        boxes = CHOICE_BOXES[:len(ids)]
        count = len(boxes)
        views = ((image, None, enumerate(boxes)),)
    for image, offset, cells in views:
        geometry = {} if offset is None else dict(scroll_offset=offset)
        for slot, box in cells:
            identity = reader.identify(image, box, plan='sense')
            if identity.visual_group is None or identity.card_id not in (None, card_id):
                continue
            family = None
            if identity.card_id is None:
                gallery = reader.recognizer.gallery
                family = {str(gallery.card_ids[i]) for i in reader.groups.get(identity.visual_group, ())}
                if card_id not in family:
                    continue
            try:
                page = read_hif_choice_panel(context, image, expected_count=count, target_slot=slot, **geometry)
            except ValueError:
                continue  # Borders or controls can still be animating.
            observed_frame = frame_identifier(image, 0)
            if (page['selected_slot'] != slot and family is not None
                    and family.intersection(candidate_ids) != {card_id}):
                # Only this family could contain the chosen target. Read its title
                # at the actual visible box; an already-selected match needs no tap.
                current, captured_at = _one(capture_frame)
                page = read_hif_choice_panel(context, current, expected_count=count, target_slot=slot, **geometry)
                check = reader.identify(current, page['boxes'][slot], plan='sense')
                if check.visual_group != identity.visual_group or check.card_id not in (None, card_id):
                    continue
                badge = stable_customization_badge_presence(
                    (measure_customization_badge(current, page['boxes'][slot]),), required_frames=1).state.value
                if page['selected_slot'] != slot:
                    def detail_journal(stage, payload):
                        journal('pending_ui_captured', dict(detail_stage=stage, detail_navigation=payload['choice']))

                    navigation = dict(slot=slot, choice_submit_count=0)
                    _dispatch(context, page['boxes'][slot], captured_at, navigation, detail_journal, 'choice_detail_select')
                    navigation['selection_click_count'] = 1
                    detail_journal('choice_detail_selected', {'choice': navigation})
                identities, badges = [None] * count, [None] * count
                identities[slot] = dict(card_id=None, observations=(asdict(identity),))
                badges[slot] = badge
                partial = dict(accepted=False, pages=(page,), identities=identities, badge_presence=badges)
                selected = _selected_target(context, partial, reader, slot, catalog, capture_frame, frame_identifier)
                page = selected['pages'][-1]
                observed_frame = selected['frame_ids'][-1]
            if page['selected_slot'] == slot:
                try:
                    confirm_hif_choice_title(catalog, '\n'.join(row['text'] for row in page['detail_title_rows']),
                                             candidate_card_ids=(int(card_id),))
                except ValueError:
                    continue
            badge = stable_customization_badge_presence(
                (measure_customization_badge(image, box),), required_frames=1).state.value
            if badge != ('DETAIL_CANDIDATE' if mods else 'CONFIDENT_ZERO'):
                continue
            ref, zone = next((ref, zone) for zone in ('draw_pile', 'discard')
                             for ref in getattr(state.deck, zone) if ref.instance_id == instance)
            candidate = dict(slot=slot, box=box, card_id=card_id, instance_id=instance, model_card_id=ref.card_id,
                             equivalent_instance_ids=(instance,), customizations={str(k): v for k, v in mods},
                             source_zone=zone)
            identities, badges = [None] * count, [None] * count
            identities[slot] = dict(card_id=card_id, observations=(asdict(identity),))
            badges[slot] = badge
            return dict(page, pages=(page,), frame_ids=(observed_frame,), accepted=True,
                        selection_limit=1, target_instance_only=True, observed_target_slot=slot,
                        selected_instance_id=instance if page['selected_slot'] == slot else None,
                        identities=identities, badge_presence=badges, candidates=[candidate], card_slots=((instance, slot),))
    if scrollable:
        raise ValueError('HIF retrieval target could not be identified on the observed move-list pages')
    return None


def execute_hif_choice_transaction(context, projected, *, reader, detail_catalog, epoch,
                                    capture_frame, journal, choose_action,
                                    observe_settlement, baseline, detail_evidence=(),
                                    frame_identifier=_frame_identifier, selected_choice=None, planned_instance=None,
                                    previous_scrolls=()):
    """Complete a retrieval using the current page and selected title."""
    from .high_score.contracts import ActionKind, GameAction

    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True

    def selected_panel(before, slot):
        if upstream:
            return _selected_target(context, before, reader, slot, detail_catalog, capture_frame, frame_identifier)
        return capture_hif_choice_panel(context, projected, reader, capture_frame=capture_frame,
            detail_catalog=detail_catalog, epoch=epoch, detail_evidence=details,
            frame_identifier=frame_identifier, expected_slot=slot).panel

    def fresh_observer(previous):
        def observe(frames, *, single_frame=False):
            if not single_frame or len(frames) != 1:
                raise ValueError('HIF choice input observer requires one explicit fresh frame')
            return _fresh_target(context, frames[0], previous, reader, action.source_slot, detail_catalog)
        return observe

    if selected_choice is not None:
        from .hif_native.native_choice_resume import _same, require_current_choice_evidence

        require_current_choice_evidence(dict(choice=selected_choice, detail_evidence=detail_evidence))

        raw = selected_choice.get('action') or {}
        if (type(raw.get('source_slot')) is not int or selected_choice.get('choice_confirm_started')
                or selected_choice.get('confirmation_submitted') or selected_choice.get('choice_submit_count') != 0):
            raise ValueError('HIF retrieval lacks an unconfirmed selected action')
        details = detail_evidence
        before = selected_choice['before_panel']
        selected = selected_panel(before, raw['source_slot'])
        if (selected.get('accepted') is not True or selected.get('selected_count') != 1
                or selected.get('confirm_enabled') is not True or selected.get('selected_instance_id') != raw['card_instance_id']):
            raise ValueError('HIF retrieval no longer shows the saved selected card and candidates')
        saved_action = GameAction(**dict(raw, kind=ActionKind(raw['kind'])))
        if not _same(choice_target_binding(before, saved_action), choice_target_binding(selected, saved_action)):
            raise ValueError('HIF retrieval no longer shows the saved selected card')
        action = choose_action(projected, selected)
        if asdict(action) != raw:
            raise ValueError('HIF retrieval policy changed its saved selected card')
        # The parent retains the original selection click. This attempt starts
        # on the already-selected card and can only confirm it.
        panel = selected
        record = dict(selected_choice)
    else:
        panel = (_planned_target_panel(context, projected, planned_instance, reader, capture_frame, frame_identifier,
                                       detail_catalog, journal=journal, previous_scrolls=previous_scrolls)
                 if upstream and planned_instance is not None else None)
        details = detail_evidence
        if panel is None:
            collected = collect_hif_choice_details(context, projected, reader,
                detail_catalog=detail_catalog, epoch=epoch,
                capture_frame=capture_frame, journal=journal, detail_evidence=details,
                frame_identifier=frame_identifier)
            panel, details = collected['panel'], collected['detail_evidence']
        action = choose_action(projected, panel)
        record = select_hif_choice_card(context, projected, panel, action,
            observe_panel=fresh_observer(panel), capture_frame=capture_frame, journal=journal)
        selected = selected_panel(panel, action.source_slot)
    if (selected.get('accepted') is False and panel.get('accepted') is True
            and tuple(selected.get('unresolved_slots', ())) == (action.source_slot,)
            and selected.get('selected_slot') == action.source_slot
            and selected.get('selected_count') == 1 and selected.get('confirm_enabled') is True):
        candidates = [row for row in panel['candidates']
                      if row['slot'] == action.source_slot
                      and row['instance_id'] == action.card_instance_id]
        if (len(candidates) != 1 or not candidates[0].get('card_id')
                or (getattr(context, 'gkh_hif_upstream_card_ui', False) is not True
                    and selected['pages'][0]['boxes'] != panel['pages'][0]['boxes'])
                or _family(selected, action.source_slot) != _family(panel, action.source_slot)
                or any(row['slot'] == action.source_slot for row in details)):
            raise ValueError('HIF choice selected target cannot reuse its complete candidate binding')
        # A previously resolved thumbnail can become weak after selection.
        # Read that already-visible target's complete detail, without another
        # click, then rebuild the current complete page through the same gate.
        evidence, _ = _capture_detail(context, projected, selected, reader, action.source_slot,
            detail_catalog=detail_catalog, epoch=epoch, capture_frame=capture_frame,
            frame_identifier=frame_identifier)
        if evidence['card_id'] != candidates[0]['card_id']:
            raise ValueError('HIF choice selected detail changed the policy target identity')
        details = (*details, evidence)
        journal('pending_ui_captured', dict(detail_stage='selected_target_detail_observed',
            detail_evidence=details, selected_slot=action.source_slot))
        selected = capture_hif_choice_panel(context, projected, reader, capture_frame=capture_frame,
            detail_catalog=detail_catalog, epoch=epoch, detail_evidence=details,
            frame_identifier=frame_identifier, expected_slot=action.source_slot).panel
    if (selected.get('accepted') is not True or selected.get('selected_count') != 1
            or selected.get('selected_instance_id') != action.card_instance_id):
        raise ValueError('HIF choice did not reach the policy-selected card')
    return confirm_hif_choice_card(context, projected, selected, action,
        observe_panel=fresh_observer(selected), capture_frame=capture_frame, journal=journal,
        choice_record=record, observe_settlement=observe_settlement, baseline=baseline)
