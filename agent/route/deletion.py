"""Delete an explicitly requested physical card through three ledger steps."""
from dataclasses import asdict, dataclass

from .hif import known_hif_role
from .engine import Outcome, Decision, propose_unique, validate_observation
from .ledger import CardInstance, LedgerEffect

DELETE_KINDS = frozenset({'select_delete_card', 'open_delete_confirmation', 'execute_card_delete'})


@dataclass(frozen=True)
class DeletionRequest:
    card: CardInstance
    fee: int
    capacity_counter: str | None = None


@dataclass(frozen=True)
class DeletionMenu:
    request: DeletionRequest
    stage: str
    selected_card: CardInstance | None
    select_target_key: str
    continue_target_key: str
    confirm_target_key: str
    observed_fee: int | None = None


@dataclass(frozen=True)
class DeletionPlan:
    decision: Decision
    effect: LedgerEffect | None = None


def request_record(request):
    record = asdict(request)
    record['card']['customization'] = [list(pair) for pair in request.card.customization]
    return record


def validate_deletion_scope(host):
    owner = host.ownership or {}
    if (owner.get('owner') != 'task600' or owner.get('phase') != 'nonbattle'
            or owner.get('game_pid') != host.game_pid or host.ledger is None):
        raise ValueError('card deletion requires the current nonbattle owner and complete ledger')


def retained_request(host):
    validate_deletion_scope(host)
    retained = (host.handoff or {}).get('nonbattle_context', {}).get('card_deletion')
    if retained is not None and (not isinstance(retained, dict) or retained.get('schema_version') != 1
            or retained.get('run_id') != host.run_id or retained.get('game_pid') != host.game_pid
            or retained.get('status') not in ('active', 'complete')):
        raise ValueError('retained card deletion request belongs to another scope')
    return retained


def plan_deletion(observation, ledger, menu, retained=None):
    def stop(reason):
        return DeletionPlan(Decision(Outcome.STOP, reason))
    error = validate_observation(observation)
    if error:
        return stop(error)
    if (not isinstance(menu, DeletionMenu) or not isinstance(menu.request, DeletionRequest)
            or ledger.pending is not None or observation.run_id != ledger.state.run_id
            or observation.day != ledger.state.day or not known_hif_role(observation.idol, observation.plan) or observation.scenario != 'HIF'):
        return stop('deletion_page_or_pending_mismatch')
    request = menu.request
    if (not isinstance(request.card, CardInstance) or request.card not in ledger.state.cards
            or sum(card.instance_id == request.card.instance_id for card in ledger.state.cards) != 1):
        return stop('deletion_exact_card_instance_not_present')
    if (type(request.fee) is not int or request.fee < 0 or request.capacity_counter is not None
            and (not isinstance(request.capacity_counter, str) or not request.capacity_counter)):
        return stop('deletion_request_price_or_counter_unknown')
    if retained is not None and retained.get('status') == 'active' and retained.get('request') != request_record(request):
        return stop('deletion_original_request_changed')
    if menu.stage not in ('choice', 'selected', 'confirm'):
        return stop('deletion_stage_unknown')
    expected_scene = 'card_delete_confirm' if menu.stage == 'confirm' else 'card_delete'
    reward_scene = {'choice': 'reward_delete_choice', 'selected': 'reward_delete_selected',
        'confirm': 'reward_delete_confirm'}[menu.stage]
    if observation.scene not in (expected_scene, reward_scene):
        return stop('deletion_scene_does_not_prove_stage')
    if menu.stage != 'choice' and menu.selected_card != request.card:
        return stop('deletion_selected_instance_changed')
    if menu.stage == 'confirm':
        if retained is None or retained.get('status') != 'active':
            return stop('deletion_confirmation_requires_original_request')
        if type(menu.observed_fee) is not int or menu.observed_fee != request.fee:
            return stop('deletion_confirmation_price_changed_or_unknown')
        if request.fee > ledger.state.currency:
            return DeletionPlan(Decision(Outcome.PAUSE, 'deletion_insufficient_currency'))
        if request.capacity_counter is not None:
            counters = dict(ledger.state.counters)
            if request.capacity_counter not in counters:
                return stop('deletion_capacity_not_observed')
            if counters[request.capacity_counter] < 1:
                return DeletionPlan(Decision(Outcome.PAUSE, 'deletion_capacity_unavailable'))
    kind, key, destinations = {
        'choice': ('select_delete_card', menu.select_target_key, ('card_delete',)),
        'selected': ('open_delete_confirmation', menu.continue_target_key, ('card_delete_confirm',)),
        'confirm': ('execute_card_delete', menu.confirm_target_key, ('card_delete_result',)),
    }[menu.stage]
    rule = f'hif-delete:{ledger.state.revision}:{menu.stage}:{request_record(request)}'
    targets = tuple(target for target in observation.targets if target.key == key and target.kind == kind)
    decision = propose_unique(observation, targets, rule, destinations)
    if decision.outcome != Outcome.ACTION:
        return DeletionPlan(decision)
    effect = LedgerEffect(f'{ledger.state.run_id}:{ledger.state.revision}:{kind}:{request.card.instance_id}', key)
    if menu.stage == 'confirm':
        effect = LedgerEffect(effect.transaction_id, key, remove_cards=(request.card,), currency_delta=-request.fee,
            counter_deltas=((request.capacity_counter, -1),) if request.capacity_counter is not None else ())
    return DeletionPlan(decision, effect)


def deletion_checkpoint(host, gate, candidate_ledger):
    """Called only at the original host's atomic save, before publishing input."""
    pending = gate.pending
    original = host.hooks.transition_gate.pending
    if pending is not None and pending.target.kind in DELETE_KINDS:
        retained = retained_request(host)
        page = host._page
        if page is None or page.observation != pending.source or page.deletion is None:
            raise ValueError('card deletion request is not bound to the fresh page')
        request = request_record(page.deletion.request)
        if retained is not None and retained.get('status') == 'active' and retained['request'] != request:
            raise ValueError('card deletion request changed before saving')
        return dict(schema_version=1, run_id=host.run_id, game_pid=host.game_pid,
            status='active', request=request,
            source_revision=(retained['source_revision'] if retained and retained['status'] == 'active'
                else host.ledger.state.revision))
    if (pending is None and original is not None and original.target.kind == 'execute_card_delete'
            and host._confirmed_ledger is candidate_ledger):
        retained = retained_request(host)
        if retained is None or retained['status'] != 'active':
            raise ValueError('card deletion settlement has no original request')
        return dict(retained, status='complete', receipt_revision=candidate_ledger.state.revision)
    return None
