"""Non-card HIF submission through the existing Maa transaction checks."""
from .drink_panel import observe_drink_detail_transaction
from .status_panel import StatusInspectionError, _normalize_status_text
from .high_score.contracts import ActionKind, DecisionKind
from .high_score.maintenance import complete_action_baseline


def _rule_drink_core_binding(bundle, baseline, observed):
    """Bind visual families to the derived exact hand without rewriting either."""
    from .high_score.hif_mcts_cache import json_value
    from .high_score.hif_rule_progress import VERIFICATION_SCOPE, resolve_rule_preinput

    inner = baseline.get('rule_progress', {})
    if (baseline.get('verification_scope') != VERIFICATION_SCOPE
            or inner.get('core') != baseline.get('core')):
        raise ValueError('rule drink baseline wrapper differs from its source')
    same, receipt = resolve_rule_preinput(bundle,inner,observed)
    fields_match = json_value(observed.get('fields')) == json_value(baseline['core']['fields'])
    return dict(accepted=receipt['accepted'] is True and same == bundle.state
            and fields_match,
        reason=receipt.get('reason') if fields_match else 'complete_core_fields_changed',
        checks=dict(receipt.get('checks',{}),complete_core_fields=fields_match),
        source_core=baseline['core'],observed_core={key:observed.get(key)
            for key in ('families','fields','frame_ids')},
        submission_authorized=False,complete_observation=False)


def revalidate_slow_hif_decision(context, baseline, policy, *, observe_complete):
    """Retain the bound state for upstream UI; preserve the legacy refresh path.

    fb8 does not require a full state reread because planning took over 500 ms.
    Fresh physical targets and actual settlement remain their respective owners.
    """
    response = policy.response
    if response is None or not response.has_action_intent:
        return baseline
    if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
        bundle = policy.bundle
        run = context.gkh_hif_live_context
        policy.bound_snapshot = None
        if (context.gkh_hif_opening_state != bundle
                or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)):
            policy.response = None
            raise ValueError('HIF state changed during search')
        return baseline
    if getattr(response.trace, 'elapsed_ms', 0) <= 500:
        return baseline
    bundle = policy.bundle
    run = context.gkh_hif_live_context
    previous_state = getattr(context, 'gkh_hif_committed_state', None)
    previous_event = getattr(context, 'gkh_hif_committed_event', None)
    policy.bound_snapshot = None
    try:
        if (context.gkh_hif_opening_state != bundle
                or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)):
            raise ValueError('HIF state changed during search')
        refreshed = observe_complete(bundle.state, baseline)
        current_run = context.gkh_hif_live_context
        observed = getattr(context, 'gkh_hif_committed_state', None)
        if (refreshed.get('accepted') is not True or not refreshed.get('checks')
                or not all(value is True for value in refreshed['checks'].values())
                or (refreshed.get('run_id'), refreshed.get('revision')) != (bundle.run_id, bundle.revision)
                or (current_run.run_id, current_run.revision) != (bundle.run_id, bundle.revision)
                or observed != bundle.state
                or context.gkh_hif_opening_state != bundle
                or len(set(refreshed.get('core', {}).get('frame_ids', ()))) != 3):
            raise ValueError('HIF complete state changed after slow search')
        return dict(refreshed, event='live_preinput_state', execution_submit_count=0,
                    search_elapsed_ms=response.trace.elapsed_ms, decision_reused=True)
    except Exception:
        policy.response = None
        raise
    finally:
        context.gkh_hif_opening_state = bundle
        context.gkh_hif_committed_state = previous_state
        context.gkh_hif_committed_event = previous_event


def execute_hif_joint_action(context, baseline, policy, *, capture_frames, checkpoint,
                             frame_identifier, observe_settlement):
    event = dict(event='live_joint_action', accepted=False, execution_submit_count=0)
    try:
        bundle = context.gkh_hif_opening_state
        run = context.gkh_hif_live_context
        response = policy.response
        if (policy.bundle != bundle or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
                or (baseline.get('run_id'), baseline.get('revision')) != (run.run_id, run.revision)
                or not complete_action_baseline(bundle, baseline)
                or response is None or not response.has_action_intent):
            raise ValueError('HIF joint action lacks a complete bound decision')
        action = response.action
        legal = bundle.kernel.legal_actions(bundle.state, plan_type=bundle.kernel.plan_type,
                                             decision_kind=DecisionKind.LIVE)
        if not any(a.kind == action.kind and a.drink_id == action.drink_id for a in legal):
            raise ValueError('HIF joint action is no longer legal')
        event['proposal'] = response.to_dict()
        if action.kind is ActionKind.END_TURN:
            from .hif_turn_execution import execute_empty_hand_turn
            return execute_empty_hand_turn(context, baseline, bound_policy=policy,
                capture_frames=capture_frames, checkpoint=checkpoint,
                observe_settlement=lambda expected, previous: observe_settlement(expected, previous, action))
        if action.kind is not ActionKind.USE_DRINK:
            raise ValueError('HIF joint transaction does not handle this action kind')
        if (action.source_slot, action.drink_id) not in (bundle.drink_slots or ()):
            raise ValueError('HIF joint drink action lacks a physical bottle')
        choice = None
        try:
            expected = policy.committed_transition()
        except ValueError:
            handler = getattr(context, 'gkh_hif_choice_transaction', None)
            journal = getattr(context, 'gkh_hif_choice_journal', None)
            if not callable(handler) or not callable(journal):
                raise
            from dataclasses import asdict

            from .high_score.choice import project_hif_drink_choice
            from .high_score.hif_mcts_native import HifMctsNativePolicy
            candidate = getattr(policy, 'decision_policy', None)
            choice = project_hif_drink_choice(bundle, action, canonical_input=(
                isinstance(candidate, HifMctsNativePolicy) and candidate.retrieval_ready))
            journal('prepared', dict(run_id=run.run_id, revision=run.revision,
                action=asdict(action), pending_state=asdict(choice.pending_state),
                joint_event=event))

        if choice is None:
            from .high_score.hif_fast_native import persist_projected_events
            persist_projected_events(context,policy)

        def progress(stage, transaction):
            event.update(transactions=[transaction], execution_submit_count=transaction.get('execution_submit_count', 0))
            journal('drink_' + stage, dict(joint_event=event))
        names = tuple(_normalize_status_text(drink.name) for drink in bundle.kernel.drinks.values())
        name = _normalize_status_text(bundle.kernel.drinks[action.drink_id].name)
        context.gkh_hif_expected_empty_hand = ((run.run_id, run.revision)
                                                if not bundle.state.deck.hand else None)
        validate_core = None
        upstream_use = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
        if not upstream_use and baseline.get('verification_scope') == 'rule_maintained_information_boundary_v1':
            def validate_core(observed):
                proof = _rule_drink_core_binding(bundle,baseline,observed)
                event['preinput_rule_binding'] = proof
                return proof['accepted']
        transactions = observe_drink_detail_transaction(context,
            capture_frames=capture_frames, checkpoint=checkpoint, frame_identifier=frame_identifier,
            names=tuple(dict.fromkeys(names)), slot_index=action.source_slot,
            _use_name=name, _expected_core=baseline['core'],
            _on_use_progress=progress if choice is not None else None,
            _expected_core_validator=validate_core, _planned_upstream_use=upstream_use)
        event['transactions'] = transactions
        event['execution_submit_count'] = sum(t.get('execution_submit_count', 0) for t in transactions)
        if not transactions or transactions[-1].get('use_submitted') is not True:
            raise ValueError('HIF drink use was not submitted')
        context.gkh_hif_committed_state = None
        context.gkh_hif_committed_event = None
        if choice is not None:
            return handler(choice, event, baseline, observe_settlement)
        from .hif_hold_transaction import complete_hif_hold

        expected = complete_hif_hold(context, policy, expected)
        settlement = observe_settlement(expected, baseline, action)
        event.update(settlement=settlement, accepted=settlement.get('accepted') is True,
                     reason=settlement.get('reason'), drinks_consumed=int(settlement.get('accepted') is True))
    except StatusInspectionError as error:
        event.update(reason=str(error), transactions=error.events,
            execution_submit_count=sum(t.get('execution_submit_count', 0) for t in error.events))
    except (ValueError, KeyError, TypeError) as error:
        event['reason'] = str(error)
    finally:
        context.gkh_hif_expected_empty_hand = None
    return event
