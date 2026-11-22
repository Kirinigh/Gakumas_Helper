"""Retain one full-bar replacement intent across discard and re-selection."""
from dataclasses import asdict, dataclass

from .hif import known_hif_role
from .engine import Outcome, Decision, propose_unique, validate_observation
from .ledger import LedgerEffect, DrinkInstance

REPLACEMENT_KINDS = frozenset({'select_full_incoming', 'open_replaced_drink', 'open_drink_discard_confirmation',
    'discard_replaced_drink', 'reselect_full_incoming', 'receive_full_incoming'})


@dataclass(frozen=True)
class DrinkReplacementRequest:
    held: tuple[DrinkInstance, ...]
    outgoing: DrinkInstance
    offers: tuple[tuple[str, str], ...]
    incoming_offer_id: str
    incoming_drink_id: str
    incoming_instance_id: str


@dataclass(frozen=True)
class DrinkReplacementMenu:
    request: DrinkReplacementRequest
    stage: str
    observed_offers: tuple[tuple[str, str], ...] | None
    selected_offer_id: str | None
    held_detail: DrinkInstance | None
    select_target_key: str
    held_target_key: str
    discard_target_key: str
    confirm_target_key: str
    receive_target_key: str
    observed_receive_cost: int | None = None


@dataclass(frozen=True)
class DrinkReplacementPlan:
    decision: Decision
    effect: LedgerEffect | None = None


def request_record(request):
    record = asdict(request)
    record['held'] = [asdict(drink) for drink in request.held]
    record['offers'] = [list(pair) for pair in request.offers]
    return record


def retained_request(host):
    owner = host.ownership or {}
    if (owner.get('owner') != 'task600' or owner.get('phase') != 'nonbattle'
            or owner.get('game_pid') != host.game_pid or host.ledger is None):
        raise ValueError('drink replacement requires the current nonbattle owner and complete ledger')
    retained = (host.handoff or {}).get('nonbattle_context', {}).get('drink_replacement')
    if retained is not None and (not isinstance(retained, dict) or retained.get('schema_version') != 1
            or retained.get('run_id') != host.run_id or retained.get('game_pid') != host.game_pid
            or retained.get('status') not in ('active', 'complete')):
        raise ValueError('retained drink replacement belongs to another scope')
    return retained


def plan_drink_replacement(observation, ledger, menu, retained=None):
    def stop(reason):
        return DrinkReplacementPlan(Decision(Outcome.STOP, reason))
    error = validate_observation(observation)
    if error:
        return stop(error)
    if (not isinstance(menu, DrinkReplacementMenu) or not isinstance(menu.request, DrinkReplacementRequest)
            or ledger.pending is not None or observation.run_id != ledger.state.run_id
            or observation.day != ledger.state.day or not known_hif_role(observation.idol, observation.plan) or observation.scenario != 'HIF'):
        return stop('drink_replacement_page_or_pending_mismatch')
    request = menu.request
    if (not isinstance(request.held, tuple) or len(request.held) != 4
            or any(not isinstance(drink, DrinkInstance) for drink in request.held)
            or any(type(drink.slot) is not int or not isinstance(drink.instance_id, str)
                or not drink.instance_id or not isinstance(drink.drink_id, str) for drink in request.held)
            or {drink.slot for drink in request.held} != {0, 1, 2, 3}
            or len({drink.instance_id for drink in request.held}) != 4 or request.outgoing not in request.held
            or any(drink.drink_id not in ledger.drink_ids for drink in request.held)):
        return stop('drink_replacement_original_four_slots_incomplete')
    if (not isinstance(request.offers, tuple) or len(request.offers) not in (3, 4)
            or any(not isinstance(pair, tuple) or len(pair) != 2
                or not isinstance(pair[0], str) or not pair[0] or not isinstance(pair[1], str)
                or pair[1] not in ledger.drink_ids
                for pair in request.offers) or len({pair[0] for pair in request.offers}) != len(request.offers)
            or (request.incoming_offer_id, request.incoming_drink_id) not in request.offers
            or not isinstance(request.incoming_instance_id, str) or not request.incoming_instance_id
            or request.incoming_instance_id in {item.instance_id for item in (*request.held, *ledger.state.cards)}):
        return stop('drink_replacement_incoming_identity_incomplete')
    if retained is not None and retained.get('status') == 'active' and retained.get('request') != request_record(request):
        return stop('drink_replacement_original_request_changed')
    if menu.stage not in ('choose_incoming', 'offer', 'held_detail', 'discard_confirm', 'reselect', 'receive'):
        return stop('drink_replacement_stage_unknown')
    if menu.stage not in ('choose_incoming', 'offer') and (retained is None or retained.get('status') != 'active'):
        return stop('drink_replacement_requires_original_request')
    if menu.stage not in ('choose_incoming', 'offer') and retained.get('free_offer_verified') is not True:
        return stop('drink_replacement_original_free_offer_not_verified')
    after_discard = menu.stage in ('reselect', 'receive')
    expected = tuple(drink for drink in request.held if not after_discard or drink != request.outgoing)
    if tuple(sorted(ledger.state.drinks, key=lambda row: row.slot)) != tuple(sorted(expected, key=lambda row: row.slot)):
        return stop('drink_replacement_current_slots_changed')
    if menu.stage in ('choose_incoming', 'offer', 'reselect', 'receive'):
        if menu.observed_offers != request.offers:
            return stop('drink_replacement_original_offer_changed')
        if menu.stage in ('offer', 'receive') and menu.selected_offer_id != request.incoming_offer_id:
            return stop('drink_replacement_selected_incoming_changed')
        if menu.stage == 'reselect' and menu.selected_offer_id is not None:
            return stop('drink_replacement_discard_selection_not_cleared')
        if menu.stage == 'choose_incoming' and menu.selected_offer_id == request.incoming_offer_id:
            return stop('drink_replacement_stage_does_not_match_selection')
    if menu.stage in ('held_detail', 'discard_confirm') and menu.held_detail != request.outgoing:
        return stop('drink_replacement_selected_held_instance_changed')
    scene = {'held_detail': 'held_drink_detail', 'discard_confirm': 'drink_discard_confirm'}.get(menu.stage, 'drink_offer')
    if observation.scene != scene:
        return stop('drink_replacement_scene_does_not_prove_stage')
    if menu.stage in ('choose_incoming', 'offer', 'receive') and (
            type(menu.observed_receive_cost) is not int or menu.observed_receive_cost != 0):
        return stop('drink_replacement_receive_is_not_observed_free')
    kind, key, destinations = {
        'choose_incoming': ('select_full_incoming', menu.select_target_key, ('drink_offer',)),
        'offer': ('open_replaced_drink', menu.held_target_key, ('held_drink_detail',)),
        'held_detail': ('open_drink_discard_confirmation', menu.discard_target_key, ('drink_discard_confirm',)),
        'discard_confirm': ('discard_replaced_drink', menu.confirm_target_key, ('drink_offer',)),
        'reselect': ('reselect_full_incoming', menu.select_target_key, ('drink_offer',)),
        'receive': ('receive_full_incoming', menu.receive_target_key, ('drink_receive_result',)),
    }[menu.stage]
    rule = f'hif-full-drink:{ledger.state.revision}:{menu.stage}:{request_record(request)}'
    targets = tuple(target for target in observation.targets if target.key == key and target.kind == kind)
    decision = propose_unique(observation, targets, rule, destinations)
    if decision.outcome != Outcome.ACTION:
        return DrinkReplacementPlan(decision)
    effect = LedgerEffect(f'{ledger.state.run_id}:{ledger.state.revision}:{kind}:{request.incoming_offer_id}', key)
    if menu.stage == 'discard_confirm':
        effect = LedgerEffect(effect.transaction_id, key, remove_drinks=(request.outgoing,))
    elif menu.stage == 'receive':
        effect = LedgerEffect(effect.transaction_id, key, add_drinks=(DrinkInstance(request.outgoing.slot,
            request.incoming_instance_id, request.incoming_drink_id),))
    return DrinkReplacementPlan(decision, effect)


def replacement_checkpoint(host, gate, candidate_ledger):
    pending = gate.pending
    original = host.hooks.transition_gate.pending
    if pending is not None and pending.target.kind in REPLACEMENT_KINDS:
        retained = retained_request(host)
        page = host._page
        if page is None or page.observation != pending.source or page.drink_replacement is None:
            raise ValueError('drink replacement is not bound to the fresh page')
        request = request_record(page.drink_replacement.request)
        if retained is not None and retained.get('status') == 'active' and retained['request'] != request:
            raise ValueError('drink replacement changed before saving')
        return dict(schema_version=1, run_id=host.run_id, game_pid=host.game_pid,
            status='active', request=request, free_offer_verified=True,
            source_revision=(retained['source_revision'] if retained and retained['status'] == 'active'
                else host.ledger.state.revision))
    if (pending is None and original is not None and original.target.kind == 'receive_full_incoming'
            and host._confirmed_ledger is candidate_ledger):
        retained = retained_request(host)
        if retained is None or retained['status'] != 'active':
            raise ValueError('drink replacement settlement has no original request')
        return dict(retained, status='complete', receipt_revision=candidate_ledger.state.revision)
    return None
