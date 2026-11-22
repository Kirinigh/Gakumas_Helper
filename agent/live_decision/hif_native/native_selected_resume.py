"""Restore a recorded selection stage without reselecting or authorizing PLAY."""
import copy
import json

from . import diagnostic_depth_reached


def _same(left, right):
    from ..high_score.hif_mcts_cache import same_json, json_value

    return same_json(json_value(left), json_value(right))


def _frames(values):
    return (isinstance(values, (list, tuple)) and len(values) == 3
            and all(isinstance(value, str) and value for value in values)
            and len(set(values)) == 3)


def selected_reobservation_has_no_input(row):
    """A stopped re-read cannot consume or replace its parent's selected action.

    Diagnostic reasons and weak readings do not authorize anything. The
    complete parent journal/cache and new A/B evidence are still required by
    the existing loader and input path. Prepared or uncertain inputs keep
    their separate outcome-reconciliation contract.
    """
    if (not isinstance(row.get('native_selected_source'), str) or not row['native_selected_source']
            or type(row.get('input_count')) is not int or row['input_count'] != 0
            or row.get('native_input_dispatches') != []
            or row.get('native_input_preparation') or row.get('native_navigation_preparation')):
        return False
    reports = row.get('live_events')
    if not isinstance(reports, list) or not reports:
        return False
    counts = ('execution_submit_count', 'end_turn_submit_count', 'drinks_consumed',
        'selection_click_count', 'execution_click_count',
        'execution_selection_click_count', 'identity_probe_click_count', 'total_input_click_count')
    for report in reports:
        if (not isinstance(report, dict) or report.get('event') != 'live_selection_stop' or report.get('executed') is not False
                or report.get('input_may_have_been_dispatched') is not False
                or any(type(report.get(key, 0)) is not int or report.get(key, 0) != 0 for key in counts)):
            return False
        for key in ('foreground_preparation_events', 'execution_selection_events', 'authoritative_refresh_events',
                    'identity_probe_events', 'shadow_transaction_events', 'failure_events'):
            events = report.get(key, ())
            if not isinstance(events, (list, tuple)):
                return False
            for event in events:
                if (not isinstance(event, dict)
                        or event.get('click_dispatched', False) is not False
                        or event.get('input_may_have_been_dispatched', False) is not False
                        or event.get('phase') == 'execution_submit'
                        or event.get('dispatch_state') is not None
                        or type(event.get('input_click_count', 0)) is not int
                        or event.get('input_click_count', 0) != 0):
                    return False
                if key == 'foreground_preparation_events' and type(event.get('input_click_count')) is not int:
                    return False
    return True


def restore_strong_selection_art(context, source, runtime, module):
    """Rebuild a visual reference after journal validation, never an input permit."""
    import cv2
    from card_selection import CandidateBox, CardPrediction
    from card_selection.model import frame_identifier

    from ..hif_selected_neighbor_art import build_strong_hand_reference

    journal = source['gray_selection_journal']
    a = journal['refresh_a']
    if journal.get('ordinary_safe_preview') is not True or journal.get('business_submit_count') != 0:
        return
    record = source
    if journal.get('source_record') and journal['source_record'] != source['id']:
        record = json.loads((runtime / ('result-'+journal['source_record']+'.json')).read_bytes())
    images = {}
    for item in record.get('frames', ()):
        image = cv2.imread(item['path'])
        if image is not None:
            fid = frame_identifier(image)
            if fid in a.get('frame_ids', ()):
                images[fid] = image
    if set(images) != set(a.get('frame_ids', ())):
        return  # Missing reference images cannot be recreated from recognition scores.
    def prediction(row):
        values = dict(row['fresh_prediction'])
        for key in ('box','source_frames','top_k_scores'):
            values[key] = tuple(values[key])
        values['top_k_card_ids'] = tuple(tuple(v) for v in values['top_k_card_ids'])
        return CardPrediction(**values)
    action = a['proposal']['candidate_traces']['high_score']['action']
    context.gkh_hif_selected_neighbor_art = None
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    target = action['source_slot']
    if (a.get('accepted') is not True or bundle is None
            or not isinstance(run, module.HifLiveRunContext)
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or type(target) is not int or not 0 <= target < len(bundle.state.deck.hand)):
        return
    # The fixed root entry has no UI callbacks. Use the same pure builder as
    # live A retention; do not recreate old title/inspection runtime receipts.
    try:
        strong = build_strong_hand_reference(run_id=run.run_id, revision=run.revision,
            selected_slot=target, frames=tuple(images[fid] for fid in a['frame_ids']),
            candidates=tuple(tuple(CandidateBox(slot=i, box=tuple(box)) for i, box in enumerate(row))
                             for row in a['candidate_boxes']),
            predictions=tuple(prediction(row) for row in a['identity_revalidation']),
            generation=a['generation'], hand_instance_ids=tuple(c.instance_id for c in bundle.state.deck.hand))
    except ValueError:
        return
    context.gkh_hif_selected_neighbor_art = {'strong_hand': strong}


def _restore_selected_support(row, bundle, baseline, journal):
    """Retain independently checked support proof without granting a new submit."""
    from types import SimpleNamespace
    from dataclasses import asdict

    from live_decision.hif_skill_support import (
        verify_selected_skill_support_observations,
        revalidate_original_support_body_observations,
    )
    from live_decision.high_score.contracts import ActionKind, GameAction
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy

    from .native_selected_support import bind_recorded_selected_support
    replay_reasons = {
        '87': 'selected extra enhancement is outside the verified slice',
        '35': 'selected support complete body or printed cost is unresolved',
    }
    def replayable(receipt):
        return (receipt.get('reason') == 'selected extra enhancement is outside the verified slice'
                or (receipt.get('card_id') in replay_reasons
                    and receipt.get('reason') == replay_reasons[receipt['card_id']]))

    reports = [event for event in row.get('live_events', ())
               if event.get('selected_skill_support') is not None or any(
                   step.get('phase') == 'hif_selected_skill_support'
                   and ((step.get('receipt') or {}).get('effects_verified') is True
                        or replayable(step.get('receipt') or {}))
                   for step in event.get('execution_selection_events', ()))]
    if not reports:
        return journal
    if len(reports) != 1:
        raise ValueError('native selected support has multiple original reports')
    raw = journal['refresh_a']['proposal']['candidate_traces']['high_score']['action']
    action = GameAction(**{**raw, 'kind': ActionKind(raw['kind'])})
    action.validate()
    policy = BoundHifOpeningPolicy(bundle, action_order=baseline['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    report = reports[0]
    if not bind_recorded_selected_support(policy, report):
        # A stopped, unsubmitted selection can supply its original raw detail
        # evidence after adding a fixed catalog tier. Keep the old rejected
        # receipt untouched, and do not turn this replay into submission proof.
        steps = [step for step in report.get('execution_selection_events', ())
                 if step.get('phase') == 'hif_selected_skill_support']
        if len(steps) != 1:
            raise ValueError('native selected support replay has ambiguous detail evidence')
        original = steps[0].get('receipt') or {}
        ref = bundle.state.deck.hand[action.source_slot]
        if (report.get('event') != 'live_selection_stop'
                or report.get('executed') is not False or report.get('execution_submit_count') != 0
                or journal.get('selection_count') != 1 or journal.get('business_submit_count') != 0
                or original.get('accepted') is not False or original.get('effects_verified') is not False
                or not replayable(original)
                or original.get('submission_authorized') is not False
                or original.get('card_id') != ref.card_id
                or original.get('run_id') != bundle.run_id or original.get('revision') != bundle.revision
                or original.get('selected_slot') != action.source_slot
                or original.get('card_instance_id') != ref.instance_id
                or not _same(original.get('bound_state'), asdict(bundle.state))):
            raise ValueError('native selected support replay differs from its unsubmitted source')
        observations = original.get('observations', ())
        if ref.card_id == '35':
            try:
                observations = revalidate_original_support_body_observations(observations)
            except ValueError:
                from live_decision.hif_gray_selection import recovered_selected_support_visible_id

                # Keep 1695's refused effect rows as refused. The existing
                # selected recovery must freshly read a complete receipt first.
                pending = [step.get('receipt') for step in journal.get('selection_events', ())
                           if step.get('phase') == 'hif_selected_skill_support']
                if len(pending) != 1 or not _same(pending[0], original):
                    raise ValueError('native selected unresolved support differs from its journal')
                if recovered_selected_support_visible_id(bundle,
                        complete_event=baseline, journal=journal) != '35':
                    raise
                return journal
        verified = verify_selected_skill_support_observations(observations,
            bundle=bundle, selected_slot=action.source_slot,
            selection_evidence=original.get('selection_evidence', {}))
        if (verified.get('accepted') is not True
                or verified.get('frame_ids') != original.get('frame_ids')
                or verified.get('raw_title') != original.get('raw_title')):
            raise ValueError('native selected support raw observations do not prove the fixed tier')
        policy.bind_selected_skill_support(verified)
        journal = dict(journal, selected_skill_support_replay=dict(
            source_record=row['id'], source_phase='hif_selected_skill_support',
            original_receipt=copy.deepcopy(original), usage='selected_resource_baseline_only',
            submission_authorized=False, fresh_support_required_before_submit=True))
    return dict(journal, selected_skill_support=copy.deepcopy(policy.selected_skill_support))


def _restore_selected_single_plus(row, bundle, baseline, journal, *, reobserved=False):
    """Carry verified zero-submit title evidence only in the returned journal."""
    from types import SimpleNamespace

    from live_decision.hif_single_plus import SOURCE, verify_selected_single_plus_estimate
    from live_decision.high_score.contracts import ActionKind, GameAction
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy

    def rejected_base_title(receipt):
        return (isinstance(receipt, dict) and receipt.get('source') == SOURCE
                and receipt.get('accepted') is False and receipt.get('single_plus_estimate') is False
                and receipt.get('reason') == 'selected title differs from ordinary family or enhancement')

    inherited = journal.get('single_plus_estimate')
    reports = [event for event in row.get('live_events', ())
               if event.get('single_plus_estimate') is not None or any(
                   step.get('phase') == 'hif_selected_single_plus'
                   and (inherited is not None or (step.get('receipt') or {}).get('single_plus_estimate') is True
                        or rejected_base_title(step.get('receipt')))
                   for step in event.get('execution_selection_events', ()))]
    if not reports and inherited is None:
        return journal
    if (len(reports) > 1 or type(journal.get('selection_count')) is not int
            or journal['selection_count'] != 1
            or type(journal.get('business_submit_count')) is not int
            or journal['business_submit_count'] != 0):
        raise ValueError('native selected single-plus lacks its original one-selection zero-submit journal')
    raw = journal['refresh_a']['proposal']['candidate_traces']['high_score']['action']
    action = GameAction(**{**raw, 'kind': ActionKind(raw['kind'])})
    action.validate()
    policy = BoundHifOpeningPolicy(bundle, action_order=baseline['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    if reports:
        report = reports[0]
        steps = report.get('execution_selection_events', ())
        title_steps = [step for step in steps if step.get('phase') == 'hif_selected_single_plus']
        receipt = report.get('single_plus_estimate')
        if (receipt is None and len(title_steps) == 1
                and rejected_base_title(title_steps[0].get('receipt'))):
            # Historical failures retain their original observations. Only the
            # newly supported, fully rechecked catalog single-plus case can be
            # bound in this returned journal; the saved failure is unchanged.
            receipt = title_steps[0]['receipt']
            checked = verify_selected_single_plus_estimate(receipt, bundle=bundle,
                action=action, require_submitted=False)
            if checked.get('ordinary_plus_projection') is not True:
                raise ValueError('native selected rejected title is not a current ordinary-plus projection')
        if (report.get('event') not in ('live_selection_stop', 'live_decision_observe')
                or report.get('executed') is not False
                or type(report.get('execution_submit_count')) is not int
                or report['execution_submit_count'] != 0
                or type(report.get('execution_selection_click_count')) is not int
                or report['execution_selection_click_count'] != (0 if reobserved else 1)
                or len(title_steps) != 1 or title_steps[0].get('click_dispatched') is not False
                or any(step.get('phase') == 'hif_single_plus_submitted'
                       or step.get('phase') == 'execution_submit' and step.get('click_dispatched') is True
                       for step in steps)):
            raise ValueError('native selected single-plus differs from its original unsubmitted report')
        if receipt is None and report.get('selected_skill_support') is not None:
            # A complete current support reading can supersede the earlier
            # title-only estimate. Recheck both original proofs and their order;
            # an accepted flag or two simultaneously active receipts cannot do so.
            from .native_selected_support import bind_recorded_selected_support

            exact_steps = [step for step in steps if step.get('phase') == 'hif_selected_skill_support']
            original = title_steps[0].get('receipt')
            checked = verify_selected_single_plus_estimate(original, bundle=bundle,
                action=action, require_submitted=False)
            if (inherited is not None or not _same(original, checked)
                    or len(exact_steps) != 1 or exact_steps[0].get('click_dispatched') is not False
                    or steps.index(title_steps[0]) >= steps.index(exact_steps[0])
                    or not _same(journal.get('selected_skill_support'), report['selected_skill_support'])
                    or not bind_recorded_selected_support(policy, report)
                    or policy.selected_skill_support.get('frame_ids') != checked.get('frame_ids')
                    or policy.selected_skill_support.get('raw_title') != checked.get('raw_title')):
                raise ValueError('native selected exact support did not replace its original title estimate')
            return dict(journal)
        if receipt is None and journal.get('selected_skill_support_replay'):
            exact_steps = [step for step in steps if step.get('phase') == 'hif_selected_skill_support']
            original = title_steps[0].get('receipt')
            checked = verify_selected_single_plus_estimate(original, bundle=bundle,
                action=action, require_submitted=False)
            replay = journal['selected_skill_support_replay']
            exact = journal.get('selected_skill_support') or {}
            if (inherited is not None or not _same(original, checked) or len(exact_steps) != 1
                    or exact_steps[0].get('click_dispatched') is not False
                    or steps.index(title_steps[0]) >= steps.index(exact_steps[0])
                    or not _same(replay['original_receipt'], exact_steps[0].get('receipt'))
                    or exact.get('effects_verified') is not True
                    or exact.get('frame_ids') != checked.get('frame_ids')
                    or exact.get('raw_title') != checked.get('raw_title')):
                raise ValueError('native replayed support did not replace its original title estimate')
            return journal
        if receipt is None and bundle.state.deck.hand[action.source_slot].card_id in ('35','709'):
            from live_decision.hif_gray_selection import recovered_selected_support_visible_id

            # 1695 raised after recording the title and refused body steps, so
            # neither local receipt reached the report's top-level fields.
            # Preserve both as observations; never bind the title's old cost.
            exact_steps = [step for step in steps if step.get('phase') == 'hif_selected_skill_support']
            pending = [step.get('receipt') for step in journal.get('selection_events', ())
                       if step.get('phase') == 'hif_selected_skill_support']
            original = title_steps[0].get('receipt')
            checked = verify_selected_single_plus_estimate(original, bundle=bundle,
                action=action, require_submitted=False)
            if (inherited is not None or reobserved or report.get('event') != 'live_selection_stop'
                    or journal.get('selected_skill_support') is not None
                    or report.get('selected_skill_support') is not None
                    or not _same(original, checked) or len(exact_steps) != 1 or len(pending) != 1
                    or exact_steps[0].get('click_dispatched') is not False
                    or steps.index(title_steps[0]) >= steps.index(exact_steps[0])
                    or not _same(exact_steps[0].get('receipt'), pending[0])
                    or pending[0].get('frame_ids') != checked.get('frame_ids')
                    or pending[0].get('raw_title') != checked.get('raw_title')
                    or recovered_selected_support_visible_id(bundle,
                        complete_event=baseline, journal=journal) != bundle.state.deck.hand[action.source_slot].card_id):
                raise ValueError('native selected unresolved support did not follow its original title estimate')
            return journal
        if receipt is None or not _same(title_steps[0].get('receipt'), receipt):
            raise ValueError('native selected single-plus differs from its original unsubmitted report')
        if journal.get('selected_skill_support') is not None or report.get('selected_skill_support') is not None:
            raise ValueError('native selected support and single-plus are both active')
    else:
        receipt = inherited
    # The caller has already verified the complete selection/epoch boundary.
    # A newer receipt is checked in full; otherwise the same-epoch inherited
    # receipt is checked again, never promoted to submitted evidence.
    if journal.get('selected_skill_support') is not None:
        raise ValueError('native selected support and single-plus are both active')
    policy.bind_selected_single_plus_estimate(receipt, require_submitted=False)
    return dict(journal, single_plus_estimate=copy.deepcopy(policy.selected_single_plus_estimate))


def _verify_completed_selection_prefix(row, report):
    """Account for navigation after the loader verified both history bindings."""
    from .native_history_navigation import history_navigation_proofs
    from .native_pending_components import _navigation_dispatch_matches

    events = row.get('live_events', ())
    if (not row.get('native_resume_source') or row.get('native_inprocess_source')
            or row.get('native_selected_source')
            or [event.get('event') for event in events] != ['live_preinput_state', 'live_decision_observe']
            or events[-1] is not report):
        raise ValueError('native resumed selection lacks its verified preinput prefix')
    preinput = events[0]
    recovery = row.get('context_recovery', {})
    context_inputs = recovery.get('inputs', ())
    if (preinput.get('accepted') is not True or preinput.get('executed') is not False
            or preinput.get('strategy_state_bound') is not True
            or any(type(preinput.get(key, 0)) is not int or preinput.get(key, 0) != 0
                   for key in ('execution_submit_count', 'end_turn_submit_count', 'drinks_consumed',
                               'execution_selection_click_count', 'identity_probe_click_count'))
            or recovery.get('accepted') is not True or recovery.get('returned_to_live') is not True
            or not recovery.get('round_context')
            or recovery.get('context') != row.get('observed_live_context')
            or recovery.get('game_pid') != row.get('endpoints', {}).get('game', {}).get('pid')
            or [entry.get('phase') for entry in context_inputs] != ['live', 'menu', 'notebook', 'identity']
            or type(recovery.get('input_click_count')) is not int
            or recovery['input_click_count'] != len(context_inputs)):
        raise ValueError('native resumed selection context navigation is incomplete')
    proofs = [dict(entry, phase='open', box=entry.get('target')) for entry in context_inputs]
    histories = (row.get('resume_history', {}), preinput.get('preinput_history_observation', {}))
    for history in histories:
        inputs = history.get('inputs', ())
        has_tail = any(key.startswith('turn_tail_') for key in history)
        if (history.get('event') != 'live_history_observe' or history.get('accepted') is not True
                or history.get('returned_unchanged') is not True
                or type(history.get('input_click_count')) is not int
                or history['input_click_count'] != len(inputs)
                or not has_tail and (len(inputs) != 2 or inputs[0].get('closing') is not False
                                     or inputs[1].get('closing') is not True)):
            raise ValueError('native resumed selection history navigation is incomplete')
        proofs.extend(history_navigation_proofs(history))
    dispatches = row.get('native_input_dispatches', ())
    if (type(preinput.get('total_input_click_count')) is not int
            or preinput['total_input_click_count'] != histories[1]['input_click_count']
            or any(type(report.get(key)) is not int or report[key] != 1
                   for key in ('execution_click_count', 'execution_selection_click_count', 'total_input_click_count'))
            or type(row.get('input_count')) is not int or row['input_count'] != len(dispatches)
            or len(dispatches) != len(proofs) + 1
            or any(entry.get('kind') not in (None, 'swipe') for entry in dispatches)
            or dispatches[-1].get('kind') is not None
            or any(not isinstance(proof.get('box'), (list, tuple)) for proof in proofs)
            or any(not _navigation_dispatch_matches(actual, proof, history=True)
                   for actual, proof in zip(dispatches[:-1], proofs, strict=True))):
        raise ValueError('native resumed selection contains unaccounted or changed navigation')


def _held_support_complementary_title(report, bundle, slot, fresh_title, *, permanent_id, visible_id):
    """Replay the current full blue title beside its verified held base card."""
    from live_decision.opening_support import _verify_frame

    ref = bundle.state.deck.hand[slot]
    events = report.get('identity_probe_events', ())
    origins = [event for event in events if event.get('phase') == 'initial_observation']
    proofs = [event for event in events
        if event.get('phase') == 'hif_temporary_support_identity' and event.get('slot') == slot]
    if not 1 <= len(bundle.state.deck.hand) <= 3 or len(origins) != 1 or len(proofs) != 1:
        raise ValueError('native complementary support lacks its unique complete current proof')
    origin, proof = origins[0], proofs[0]
    frames, boxes = origin.get('frame_ids'), origin.get('candidate_boxes', ())
    observations = proof.get('observations', ())
    if (visible_id is None or ref.upgraded or permanent_id != ref.card_id
            or proof.get('source') != 'bound_rightmost_blue_support_identity'
            or (proof.get('run_id'), proof.get('revision'), proof.get('observed_turn'))
                != (bundle.run_id, bundle.revision, bundle.state.turn)
            or (proof.get('card_instance_id'), proof.get('card_id'), proof.get('visible_card_id'))
                != (ref.instance_id, ref.card_id, visible_id)
            or proof.get('effects_verified') is not False or proof.get('submission_authorized') is not False
            or origin.get('geometry_stable') is not True or origin.get('identity_source_accepted') is not True
            or not _frames(frames) or frames != fresh_title.get('frame_ids')
            or len(observations) != 3 or [item.get('frame_id') for item in observations] != frames
            or len(boxes) != 3 or any(len(frame) != len(bundle.state.deck.hand) for frame in boxes)):
        raise ValueError('native complementary support changed its instance, epoch or original frames')
    for index, observation in enumerate(observations):
        box = boxes[index][slot]
        if (len(box) != 4 or any(type(value) is not int for value in box)
                or box[2:] != [190, 250]):
            raise ValueError('native complementary support lacks its complete original card')
        try:
            _verify_frame(observation, slot, anchor=box[0],
                title_text=bundle.kernel.cards[visible_id].name, suffix_right_margin=6)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError('native complementary support title proof is incomplete') from error


def _selection_identity_titles(report, *, verified_short_hand=None):
    """Separate a retained title observation from its later epoch-bound result."""
    events = report.get('identity_probe_events', ())
    titles = [event for event in events
        if event.get('phase') in ('hif_complete_title_identity', 'retained_complete_hand_identity')]
    observations = [event for event in titles if 'retained_title_evidence' in event]
    if not observations:
        return titles
    bound = [event for event in titles if 'retained_title_evidence' not in event]
    retained = [event for event in bound if event.get('phase') == 'retained_complete_hand_identity']
    origins = [event for event in events if event.get('phase') == 'initial_observation']
    if (len(observations) == 1 and not bound and verified_short_hand is not None
            and len(verified_short_hand.get('card_ids', ())) in (1, 2)
            and observations[0].get('card_ids') == verified_short_hand['card_ids']):
        # Short-hand producers omit the separate epoch-binding event.
        # Reconstruct only its business binding from the verified rule source;
        # the original title/geometry and fresh A evidence remain mandatory.
        identity = dict(phase='retained_complete_hand_identity', accepted=True,
            **verified_short_hand, binding_source='verified_rule_short_hand')
        retained = bound = [identity]
    if len(observations) != 1 or len(retained) != 1 or len(origins) != 1:
        raise ValueError('native retained identity observation has no unique binding')
    observation, identity, origin = observations[0], retained[0], origins[0]
    evidence = observation.get('retained_title_evidence', {})
    ids = identity.get('card_ids', ())
    frames = observation.get('frame_ids')
    rows = evidence.get('observations', ())
    if (observation.get('phase') != 'retained_complete_hand_identity'
            or observation.get('accepted') is not True or identity.get('accepted') is not True
            or 'run_id' in observation or 'revision' in observation
            or not ids or observation.get('card_ids') != ids or evidence.get('card_ids') != ids
            or not _frames(frames) or frames != origin.get('frame_ids')
            or origin.get('geometry_stable') is not True or origin.get('identity_source_accepted') is not True
            or len(rows) != 3 or any([row.get('slot') for row in frame] != list(range(len(ids))) for frame in rows)):
        raise ValueError('native retained identity detail differs from its complete observation')
    # Text observations may be weak: only the existing bound epoch, complete
    # inventory and fresh A identities below can authorize the selected intent.
    return bound


def _completed_selection_report(row, bundle, *, verified_resume_prefix=False):
    """Describe an actual completed selection; never invent a pre-input journal."""
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy

    reports = [event for event in row.get('live_events', ())
               if event.get('event') in ('live_selection_stop', 'live_decision_observe')
               and event.get('execution_selection_click_count') == 1]
    if len(reports) != 1:
        raise ValueError('native selection recovery lacks one completed selection report')
    report = reports[0]
    observed_return = report.get('event') == 'live_decision_observe'
    if observed_return and verified_resume_prefix is True:
        _verify_completed_selection_prefix(row, report)
    elif observed_return and (not row.get('native_inprocess_source')
            or len(row.get('live_events', ())) != 1 or row.get('input_count') != 1
            or len(row.get('native_input_dispatches', ())) != 1):
        raise ValueError('native observation return lacks its single bound selection')
    if observed_return and (report.get('execution_click_count') != 1
            or report.get('total_input_click_count') != 1
            or row.get('observed_live_context', {}).get('run_id') != bundle.run_id
            or row.get('observed_live_context', {}).get('revision') != bundle.revision):
        raise ValueError('native observation return lacks its single bound selection')
    if (report.get('executed') is not False or report.get('execution_submit_count') != 0
            or report.get('identity_probe_click_count') != 0
            or sum(event.get('execution_selection_click_count', 0)
                   for event in row.get('live_events', ())) != 1):
        raise ValueError('native selection report contains another business or probe input')
    readings = [event for event in report.get('authoritative_refresh_events', ())
                if event.get('phase') == 'authoritative_refresh_a']
    verified_short_hand = None
    if row.get('information_boundary_execution') is True and row.get('native_rule_resume_source'):
        verified_short_hand = dict(run_id=bundle.run_id, revision=bundle.revision,
            card_ids=[bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id
                      for ref in bundle.state.deck.hand])
    titles = _selection_identity_titles(report, verified_short_hand=verified_short_hand)
    phases = [event['phase'] for event in titles]
    complementary = (len(titles) == 2 and len(set(phases)) == 2)
    if (len(readings) != 1 or readings[0].get('accepted') is not True
            or (len(titles) != 0 if observed_return else len(titles) != 1 and not complementary)
            or any(event.get('accepted') is not True for event in titles)):
        raise ValueError('native selection report lacks a complete accepted A identity')
    fresh_a = readings[0]
    retained_title = next((event for event in titles
        if event['phase'] == 'retained_complete_hand_identity'), None)
    retained = retained_title is not None
    if retained and (retained_title.get('run_id') != bundle.run_id
                     or retained_title.get('revision') != bundle.revision):
        raise ValueError('native retained hand identity belongs to another complete binding')
    if complementary:
        fresh_title = next(event for event in titles if event['phase'] == 'hif_complete_title_identity')
        if (len(fresh_title.get('card_ids', ())) != len(bundle.state.deck.hand)
                or len(retained_title.get('card_ids', ())) != len(bundle.state.deck.hand)
                or not _frames(fresh_title.get('frame_ids')) or not _frames(fresh_a.get('frame_ids'))):
            raise ValueError('native complementary hand identities disagree or lack complete fresh frames')
    identities = fresh_a.get('identity_revalidation', ())
    card_ids = ([bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id
                 for ref in bundle.state.deck.hand] if observed_return else titles[0].get('card_ids', ()))
    if len(identities) != len(bundle.state.deck.hand) or len(card_ids) != len(identities):
        raise ValueError('native selection report has an incomplete bound hand')
    certificates = []
    identity_policy = BoundHifOpeningPolicy(bundle, action_order=())
    for slot, (ref, identity, card_id) in enumerate(zip(bundle.state.deck.hand, identities, card_ids, strict=True)):
        definition = bundle.kernel.cards[ref.card_id]
        permanent_id = definition.canonical_card_id or definition.card_id
        support_id = identity_policy._opening_support_visible_id(ref)
        held_support_id = identity_policy._hand_support_visible_id(ref)
        support_id = support_id or held_support_id
        if complementary and fresh_title['card_ids'][slot] != retained_title['card_ids'][slot]:
            if (fresh_title['card_ids'][slot] != held_support_id
                    or retained_title['card_ids'][slot] != permanent_id):
                raise ValueError('native complementary hand identities differ from the verified held mapping')
            _held_support_complementary_title(report, bundle, slot, fresh_title,
                permanent_id=permanent_id, visible_id=held_support_id)
        title_ids = {permanent_id if observed_return else support_id or permanent_id}
        if held_support_id is not None:
            # The original title proof has already regenerated this held fact;
            # retained observers intentionally serialize the permanent ID.
            title_ids.add(permanent_id)
        prediction = identity.get('fresh_prediction', {})
        if (observed_return or complementary) and (not _frames(fresh_a.get('frame_ids'))
                or prediction.get('accepted') is not True or prediction.get('slot') != slot
                or prediction.get('frame_id') != fresh_a['frame_ids'][-1]
                or prediction.get('source_frames') != [fresh_a['frame_ids'][-1]]):
            raise ValueError('native observation return lacks its fresh per-slot identity')
        if (identity.get('slot') != slot or identity.get('accepted') is not True
                or identity.get('standard_accepted') is not True or identity.get('carried') is not False
                or ((retained or observed_return) and identity.get('acceptance_mode') != 'fresh_exact_family')
                or card_id not in title_ids
                or card_id not in identity.get('fresh_family', ())
                or (support_id is not None and support_id not in identity.get('fresh_family', ()))
                or permanent_id not in identity.get('fresh_family', ())
                or identity.get('fresh_family') != identity.get('expected_family')):
            raise ValueError('native selection report identity differs from its complete binding')
        certificate = dict(slot=slot, card_id=permanent_id,
            family=copy.deepcopy(identity['fresh_family']), upgraded=ref.upgraded,
            provenance='completed_native_selection_report', upgrade_source='verified_resume_state')
        title_proofs = [proof for proof in fresh_a.get('bound_complete_title_families', ())
                        if proof.get('slot') == slot]
        if len(title_proofs) == 1 and prediction:
            # Preserve actual A evidence for later independent title rereads.
            # The consumer still validates its complete binding and raw model
            # fields; an accepted family alone does not manufacture this proof.
            certificate.update(source_prediction=copy.deepcopy(prediction),
                               bound_title_evidence=copy.deepcopy(title_proofs[0]))
        if support_id:
            # The existing policy replays the original blue-mark receipt against
            # its verified epoch and physical instance. Never promote the deck.
            certificate.update(visible_card_id=support_id, visible_upgraded=True,
                               upgrade_source='verified_opening_temporary_support')
        certificates.append(certificate)
    result = dict(provenance='completed_native_selection_report',
        source_event=report['event'], source_record=row['id'],
        identity_source_phase='authoritative_refresh_a' if observed_return else titles[0]['phase'],
        stage='selection_dispatched', stages=['selection_dispatched'],
        ordinary_safe_preview=True, refresh_a=copy.deepcopy(fresh_a),
        identity_certificate=certificates, selection_ready=True,
        selection_count=report['execution_selection_click_count'],
        business_submit_count=report['execution_submit_count'],
        selection_events=copy.deepcopy(report.get('execution_selection_events', ())))
    if complementary:
        result.update(identity_source_phases=phases, complete_identity_evidence=copy.deepcopy(titles))
    if titles and titles[0].get('binding_source') == 'verified_rule_short_hand':
        result.update(reconstructed_identity_binding=copy.deepcopy(titles[0]),
            complete_identity_evidence=copy.deepcopy(report['identity_probe_events']))
    return result


def verify_selected_continuation(row, source):
    """A new Maa host may recover this selection only in the same game/epoch."""
    _verify_selected_epoch(row, source)
    if row.get('native_resume_source') or row.get('native_inprocess_source'):
        raise ValueError('native selected continuation has conflicting sources')
    journal = source['gray_selection_journal']
    selected = next(event['attempted_slot'] for event in journal['selection_events']
                    if event.get('phase') == 'execution_selection')
    original_action = (dict(card_instance_id=journal['permit']['card_instance_id'], source_slot=selected)
                       if journal.get('permit') else
                       journal['refresh_a']['proposal']['candidate_traces']['high_score']['action'])
    events = row.get('live_events', ())
    if any(event.get('execution_selection_click_count', 0)
           or any(step.get('phase') == 'execution_selection' and step.get('click_dispatched') is True
                  for step in event.get('execution_selection_events', ()))
           for event in events):
        raise ValueError('native selected continuation selected a card again')
    submitted = [event for event in events
                 if event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)]
    if len(submitted) != 1:
        raise ValueError('native selected continuation must contain one card submission')
    report = submitted[0]
    if report.get('event') == 'live_decision_exception_stop':
        from .native_settlement_resume import decoded_dispatched_action

        _, report = decoded_dispatched_action(row)
    elif report.get('event') != 'live_decision_observe':
        raise ValueError('native selected continuation must contain one card submission')
    action = report.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
    if (action.get('kind') != 'PLAY_CARD' or action.get('source_slot') != selected
            or action.get('card_instance_id') != original_action.get('card_instance_id')):
        raise ValueError('native selected continuation changed the reserved card')


def _verify_selected_epoch(row, source):
    if (row.get('observed_live_context') != source.get('observed_live_context')
            or any(row.get('endpoints', {}).get('game', {}).get(key)
                   != source.get('endpoints', {}).get('game', {}).get(key)
                   for key in ('pid', 'client_size'))):
        raise ValueError('native selected continuation changed game, dimensions, or epoch')


def load_selected_source(runtime, name, module, trail=()):
    """Return the pre-submit bundle/state/event plus the untouched journal row.

    Old selected evidence is only a recovery record. A caller must take fresh
    selected A/B observations before considering any business input.
    """
    from .native_opening_root import has_native_opening_root, load_native_opening_root, verify_native_round_context
    from .native_support_resume import restore_native_history
    from .native_inprocess_resume import _same_input_host, restore_inprocess_source
    from .native_settlement_resume import load_result, _verified_endpoints, load_verified_source

    if name in trail or diagnostic_depth_reached(trail):
        raise ValueError('native selected evidence chain is cyclic or unbounded')
    row = load_result(runtime, name)
    from .native_handoff_resume import verify_cleared_selection, is_cleared_selection_timing_stop

    if is_cleared_selection_timing_stop(row):
        bundle, state, baseline, parent = load_selected_source(
            runtime, row['native_rule_resume_source'], module, (*trail, name))
        verify_native_round_context(row, module, parent)
        _verify_selected_epoch(row, parent)
        verify_cleared_selection(bundle, baseline, parent, row['native_selection_rebind'])
        return bundle, state, baseline, dict(row, gray_selection_journal=copy.deepcopy(parent['gray_selection_journal']),
            selection_evidence_source=parent.get('selection_evidence_source', parent['id']))
    source_keys = [key for key in ('native_resume_source', 'native_selected_source', 'native_inprocess_source',
                                  'native_rule_resume_source')
                   if row.get(key)]
    # A handoff may republish its verified untouched opening. Its original
    # parent and receipt still own that observation; it is not a new root.
    reused_opening = (row.get('handoff_resume', {}).get('boundary_kind') == 'unsubmitted_opening'
                      and row.get('native_rule_resume_source'))
    root = has_native_opening_root(row) and not reused_opening
    if len(source_keys) != (0 if root else 1):
        raise ValueError('native selected preview lacks an unambiguous verified resume source')
    if row.get('native_selected_source'):
        bundle, state, baseline, parent = load_selected_source(
            runtime, row['native_selected_source'], module, (*trail, name))
        verify_native_round_context(row, module, parent)
        _verify_selected_epoch(row, parent)
        if (row.get('native_input_dispatches') or row.get('input_count', 0)
                or any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
                       or event.get('execution_selection_click_count', 0) or event.get('identity_probe_click_count', 0)
                       or event.get('executed') is True for event in row.get('live_events', ()))):
            raise ValueError('native selected reobservation contains additional input')
        journal = _restore_selected_support(row, bundle, baseline,
            copy.deepcopy(parent['gray_selection_journal']))
        journal = _restore_selected_single_plus(row, bundle, baseline, journal, reobserved=True)
        return bundle, state, baseline, dict(row,
            gray_selection_journal=journal,
            selection_evidence_source=parent.get('selection_evidence_source', parent['id']))
    run = module.HifLiveRunContext(**row['observed_live_context'])
    inventory_source = None
    if root:
        bundle, baseline = load_native_opening_root(row, module)
        state = bundle.state
    elif row.get('native_inprocess_source'):
        bundle, baseline = restore_inprocess_source(runtime, row, module, trail)
        state = bundle.state
    elif row.get('native_rule_resume_source'):
        from .native_rule_progress_resume import restore_rule_source
        bundle, baseline = restore_rule_source(runtime, row, module, trail)
        state = bundle.state
    else:
        parent_bundle, parent_state, saved, parent = load_verified_source(
            runtime, row['native_resume_source'], module, (*trail, name))
        inventory_source = (parent_bundle, parent_state, saved, parent['id'])
        verify_native_round_context(row, module, parent)
        bundle, state, baseline = restore_native_history(runtime, module, parent_bundle, parent_state,
            settlement=saved, reference=row['resume_reference'], current=row['resume_history'],
            old_run=module.HifLiveRunContext(**parent['observed_live_context']), new_run=run,
            old_pid=parent['endpoints']['game']['pid'], new_pid=row['endpoints']['game']['pid'])
        if not _same(baseline, row.get('resume_settlement')):
            raise ValueError('native selected preview rebound differs from its recorded observations')
    from .native_hand_support import bind_planning_hand_support
    from .native_preinput_history import verify_preinput_history
    bundle = bind_planning_hand_support(bundle, row, runtime=runtime, module=module, trail=trail,
        inventory_source=inventory_source)
    if row.get('information_boundary_execution') is True:
        from .native_rule_progress_resume import verify_rule_preinput

        baseline = verify_rule_preinput(row, bundle, baseline)
    verified_preinput_history = verify_preinput_history(row, bundle, baseline)
    state = bundle.state
    from_report = 'gray_selection_journal' not in row
    if from_report:
        # Only this dependency-closed normal resume may contain navigation
        # before an observed one-selection return. Retain every original row.
        if inventory_source is not None and verified_preinput_history is True:
            journal = _completed_selection_report(row, bundle, verified_resume_prefix=True)
        else:
            journal = _completed_selection_report(row, bundle)
        row = dict(row, gray_selection_journal=journal)
    journal = row.get('gray_selection_journal', {})
    stages = journal.get('stages')
    allowed_stages = ((['selection_dispatched'],) if from_report else
                      (['selection_reserved', 'selection_dispatched'],
                       ['selection_reserved', 'selection_dispatched', 'selected_observation']))
    if (stages not in allowed_stages
            or journal.get('stage') != stages[-1]
            or journal.get('selection_ready') is not True
            or type(journal.get('selection_count')) is not int or journal['selection_count'] != 1
            or journal.get('business_submit_count', 0) != 0):
        raise ValueError('native selected preview does not prove exactly one completed selection')
    if any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
           or event.get('drinks_consumed', 0) or event.get('executed') is True
           for event in row.get('live_events', ())):
        raise ValueError('native selected preview contains a business submission')
    selection_events = [event for event in journal.get('selection_events', ())
                        if event.get('phase') == 'execution_selection']
    if (len(selection_events) != 1 or selection_events[0].get('accepted') is not True
            or selection_events[0].get('click_dispatched') is not True
            or selection_events[0].get('reason') != 'target_selection_dispatched'):
        raise ValueError('native selected preview lacks a confirmed selection dispatch')
    slot = selection_events[0].get('attempted_slot')
    if type(slot) is not int or slot < 0:
        raise ValueError('native selected slot is invalid')

    if slot >= len(state.deck.hand):
        raise ValueError('native selected target lies outside the verified hand')
    ref = state.deck.hand[slot]
    card = bundle.kernel.cards[ref.card_id]
    card_id = card.canonical_card_id or card.card_id
    certificates = journal.get('identity_certificate', ())
    if len(certificates) != len(state.deck.hand):
        raise ValueError('native selected preview identity certificates are incomplete')
    certificate = certificates[slot]
    if (certificate.get('slot') != slot or certificate.get('card_id') != card_id
            or certificate.get('upgraded') is not ref.upgraded
            or card_id not in certificate.get('family', ())):
        raise ValueError('native selected preview target identity differs')

    fresh_a = journal.get('refresh_a', {})
    generation = fresh_a.get('generation', {})
    expected = dict(turn=state.turn, turns_remaining=state.turns_remaining,
        stamina=state.stamina, shield=state.genki, score=state.score,
        score_multiplier=state.score_multiplier, hand_count=len(state.deck.hand),
        plan=run.plan, action_order=baseline['core']['fields'][5])
    if (fresh_a.get('phase') != 'authoritative_refresh_a'
            or fresh_a.get('expected_selected_slot') is not None
            or fresh_a.get('unselected_geometry_authorized') is not True
            or fresh_a.get('click_dispatched') is not False
            or fresh_a.get('input_may_have_been_dispatched') is not False
            or not _frames(fresh_a.get('frame_ids'))
            or not _same(generation, expected)):
        raise ValueError('native selected preview lacks bound unselected A evidence')
    from .native_card_evidence import input_evidence_is_fresh

    if not input_evidence_is_fresh(fresh_a):
        raise ValueError('native selected preview A evidence was stale')
    consensus = fresh_a.get('select_consensus', {})
    if (consensus.get('accepted') is not True or consensus.get('selected_slot') is not None
            or consensus.get('observations') != [[], [], []]):
        raise ValueError('native selected preview A was not unanimously unselected')
    boxes = fresh_a.get('candidate_boxes', ())
    if (len(boxes) != 3 or any(len(frame) != len(state.deck.hand) for frame in boxes)
            or any(len(frame[slot]) != 4 or any(type(value) is not int for value in frame[slot])
                   for frame in boxes)):
        raise ValueError('native selected preview selection geometry is incomplete')
    box = boxes[-1][slot]
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError('native selected preview selection geometry is invalid')
    ordinary = journal.get('ordinary_safe_preview') is True
    if ordinary:
        safe = fresh_a.get('safe_decision', {})
        original_action = fresh_a.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
        if ('permit' in journal or fresh_a.get('accepted') is not True
                or safe.get('status') != 'ALLOW' or safe.get('has_action_intent') is not True
                or safe.get('action') != dict(kind='PLAY_CARD', card_slot=slot)
                or original_action.get('kind') != 'PLAY_CARD'
                or original_action.get('source_slot') != slot
                or original_action.get('card_instance_id') != ref.instance_id):
            raise ValueError('native ordinary preview lacks its original safe A decision')
    else:
        from live_decision.hif_gray_selection import verify_recorded_gray_selection_reservation
        verify_recorded_gray_selection_reservation(bundle, journal,
            slot=slot, box=box, generation=generation)
    dispatches = row.get('native_input_dispatches', ())
    if not dispatches:
        raise ValueError('native selected preview lacks audited input')
    for entry in dispatches:
        _verified_endpoints(entry.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], entry['endpoints'])
    x, y, width, height = box
    audited_targets = {(x + width // 2, y + height // 2)}
    if len(bundle.state.deck.hand) >= 4:
        from live_decision.hif_visual import hif_selection_click_box

        # First-selection clicks may use the independently visible strip,
        # while the journal keeps the complete card box for identity checks.
        original_boxes = journal['refresh_a'].get('candidate_boxes', ())
        if (len(original_boxes) == 3
                and all(len(row) == len(bundle.state.deck.hand) for row in original_boxes)
                and list(original_boxes[-1][slot]) == list(box)):
            tx, ty, tw, th = hif_selection_click_box(original_boxes[-1], slot)
            audited_targets.add((tx + tw // 2, ty + th // 2))
    matches = [entry for entry in dispatches if entry.get('kind') != 'swipe'
               and (entry.get('x'), entry.get('y')) in audited_targets]
    if len(matches) != 1 or matches[0] is not dispatches[-1]:
        raise ValueError('native selected preview has no unique final audited selection')
    if journal['stage'] == 'selected_observation':
        fresh_b = journal.get('refresh_b', {})
        selected = fresh_b.get('select_consensus', {})
        if (journal.get('selected_slot') != slot or journal.get('business_submit_count') != 0
                or fresh_b.get('phase') != 'authoritative_refresh_b'
                or fresh_b.get('expected_selected_slot') != slot
                or not _frames(fresh_b.get('frame_ids'))
                or set(fresh_b['frame_ids']) & set(fresh_a['frame_ids'])
                or selected.get('accepted') is not True or selected.get('selected_slot') != slot):
            raise ValueError('native selected preview B stage is inconsistent')
    if not ordinary:
        # Only this returned copy records the completed native loader checks.
        # The saved A remains rejected; the permit still grants zero submits.
        journal = dict(journal, provenance='verified_native_gray_selection_reservation',
            source_record=row['id'], business_submit_count=journal.get('business_submit_count', 0))
    journal = _restore_selected_support(row, bundle, baseline, journal)
    journal = _restore_selected_single_plus(row, bundle, baseline, journal)
    # The original report, A generation and single selection have now all
    # been verified. Attach missing historical hand metadata to the existing
    # complete high-margin title proof; never rewrite the saved source record.
    from types import SimpleNamespace

    from live_decision.hif_bound_title import bind_complete_base_title

    bound_context = SimpleNamespace(gkh_hif_opening_state=bundle,
        gkh_hif_live_context=run,gkh_hif_committed_event=baseline)
    certificates = copy.deepcopy(journal['identity_certificate'])
    for certificate in certificates:
        raw = certificate.get('source_prediction')
        proof = certificate.get('bound_title_evidence')
        if raw and proof and proof.get('scope') is None:
            prediction = SimpleNamespace(**raw,to_dict=lambda source=raw:source)
            bound_proof = bind_complete_base_title(bound_context,prediction,proof)
            if bound_proof is not None:
                certificate['bound_title_evidence'] = bound_proof
    journal = dict(journal,identity_certificate=certificates)
    return bundle, state, baseline, dict(row, gray_selection_journal=journal)
