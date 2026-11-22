"""Versioned, data-only persistence of the opt-in strong policy computation.

No executable object, observation capability, deadline, or arbitrary class name
is restored. Callers validate their action/terminal links before applying the
prepared computation. The legacy v22 cache remains unchanged.
"""
from __future__ import annotations

import copy
import math
from enum import Enum
from dataclasses import asdict, fields, replace

from . import contracts
from .hif_mcts_cache import (
    same_json,
    json_value,
    definitions,
    exact_state,
    export_tree,
    restore_tree,
    public_request,
    slots_from_json,
    request_from_public,
    action_key_from_json,
)

VERSION = 'hif-strong-computation-cache-v1'


def _registry():
    from .kernel import RuleEvent, TransitionBranch
    from .planner import _Outcome
    from .hif_internal_reanalysis import Sample, Decision
    from .hif_resource_reanalysis import ResourceCursor, ResourceDecision
    from .hif_public_turn_reanalysis import PublicTurnCursor, PublicTurnPolicy
    from .hif_internal_successor_graph import NodeRecord, GraphDecision

    # Explicit business types only. No module/class supplied by a file is loaded.
    names = ('CardRef', 'VerifiedHandSupport', 'DeckBelief', 'TimedStatus',
        'TimedModifier', 'ScheduledEffectRef', 'EffectTriggerContext',
        'CardEffectsContinuation', 'ScheduledEffectsContinuation',
        'FreePlayAllContinuation', 'TurnResolutionContinuation', 'PendingChoice',
        'UtilitySpec', 'GameAction')
    records = [getattr(contracts, name) for name in names]
    records += [Decision, Sample, GraphDecision, NodeRecord, PublicTurnPolicy,
                PublicTurnCursor, ResourceDecision, ResourceCursor, RuleEvent,
                TransitionBranch, _Outcome]
    enums = [value for value in vars(contracts).values()
             if isinstance(value, type) and issubclass(value, Enum) and value is not Enum]
    return {value.__name__: value for value in records}, {value.__name__: value for value in enums}


def encode_value(value, *, context=None):
    """Encode whitelisted immutable business values, preserving tuple/map order."""
    from .hif_resource_reanalysis import ResourcePolicy
    from .hif_public_turn_reanalysis import PublicTurnPolicy

    records, enums = _registry()
    def encode(current):
        if context is not None and isinstance(current, tuple) and current == context:
            return {'type': 'runtime_context'}
        if isinstance(current, Enum):
            if type(current).__name__ not in enums:
                raise ValueError('strong_cache_enum_not_supported')
            return {'type': 'enum', 'name': type(current).__name__, 'value': current.value}
        if current is None or type(current) in (str, bool, int):
            return current
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError('strong_cache_nonfinite')
            return current
        if isinstance(current, contracts.StrategyState):
            return {'type': 'state', 'value': json_value(asdict(current))}
        if isinstance(current, contracts.DecisionRequest):
            return {'type': 'request', 'value': public_request(current),
                    'card_slots': encode(tuple(current.card_slots)),
                    'protocol_version': current.protocol_version}
        if isinstance(current, ResourcePolicy):
            return {'type': 'resource_policy', 'version': current.version,
                    'context': encode(current.context), 'decisions': encode(current.decisions)}
        if isinstance(current, PublicTurnPolicy):
            # CompleteGraph is a local evaluation subclass; persist its data.
            return {'type': 'record', 'name': 'PublicTurnPolicy', 'fields': {
                field.name: encode(getattr(current, field.name)) for field in fields(PublicTurnPolicy)}}
        if type(current).__name__ in records and type(current) is records[type(current).__name__]:
            return {'type': 'record', 'name': type(current).__name__, 'fields': {
                field.name: encode(getattr(current, field.name)) for field in fields(current)}}
        if isinstance(current, dict):
            return {'type': 'dict', 'items': [[encode(key), encode(item)] for key, item in current.items()]}
        for kind in (tuple, list, set, frozenset):
            if type(current) is kind:
                return {'type': kind.__name__, 'items': [encode(item) for item in current]}
        raise ValueError(f'strong_cache_type_not_supported:{type(current).__name__}')
    return encode(value)


def decode_value(raw, *, kernel, search=None, context=None):
    """Restore only declared business types; requests have no input capability."""
    from .hif_resource_reanalysis import VERSION as RESOURCE_VERSION
    from .hif_resource_reanalysis import ResourcePolicy

    records, enums = _registry()
    def decode(value):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if not isinstance(value, dict) or 'type' not in value:
            raise ValueError('strong_cache_value_invalid')
        kind = value['type']
        if kind == 'runtime_context' and set(value) == {'type'} and context is not None:
            return context
        if kind == 'enum' and set(value) == {'type', 'name', 'value'} and value['name'] in enums:
            return enums[value['name']](value['value'])
        if kind == 'state' and set(value) == {'type', 'value'}:
            return exact_state(value['value'])
        if kind == 'request' and set(value) == {'type', 'value', 'card_slots', 'protocol_version'}:
            request = request_from_public(value['value'], kernel)
            slots = decode(value['card_slots'])
            if (not isinstance(slots, tuple) or any(not isinstance(pair, tuple) or len(pair) != 2
                    or not isinstance(pair[0], str) or type(pair[1]) is not int or pair[1] < 0 for pair in slots)
                    or len(dict(slots)) != len(slots)
                    or set(dict(slots)) != set(dict(request.card_slots))
                    or len({slot for _, slot in slots}) != len(slots)
                    or value['protocol_version'] != request.protocol_version):
                raise ValueError('strong_cache_request_bindings_invalid')
            request = replace(request, card_slots=slots)
            request.validate()
            kernel._validate_registered_state(request.state, plan_type=request.plan_type)
            return request
        if kind == 'resource_policy' and set(value) == {'type', 'version', 'context', 'decisions'}:
            if search is None or value['version'] != RESOURCE_VERSION:
                raise ValueError('strong_cache_resource_version_invalid')
            return ResourcePolicy(value['version'], decode(value['context']),
                                  search._resource_key, decode(value['decisions']))
        if kind == 'record' and set(value) == {'type', 'name', 'fields'} and value['name'] in records:
            record = records[value['name']]
            if not isinstance(value['fields'], dict) or set(value['fields']) != {f.name for f in fields(record)}:
                raise ValueError('strong_cache_record_fields_invalid')
            result = record(**{key: decode(item) for key, item in value['fields'].items()})
            if hasattr(result, 'validate'):
                result.validate()
            return result
        if kind in ('dict', 'tuple', 'list', 'set', 'frozenset') and set(value) == {'type', 'items'}:
            if not isinstance(value['items'], list):
                raise ValueError('strong_cache_container_invalid')
            if kind == 'dict':
                result = {}
                for pair in value['items']:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError('strong_cache_map_invalid')
                    key, item = map(decode, pair)
                    if key in result:
                        raise ValueError('strong_cache_duplicate_key')
                    result[key] = item
                return result
            items = [decode(item) for item in value['items']]
            result = {'tuple': tuple, 'list': list, 'set': set, 'frozenset': frozenset}[kind](items)
            if kind in ('set', 'frozenset') and len(result) != len(items):
                raise ValueError('strong_cache_duplicate_key')
            return result
        raise ValueError('strong_cache_tag_invalid')
    return decode(raw)


def _fields(raw, names):
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ValueError('strong_cache_fields_invalid')


def _integer(value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('strong_cache_integer_invalid')
    return value


def _seeds(values, *, maximum=2**32):
    if (not isinstance(values, (list, tuple, set)) or len(set(values)) != len(values)
            or any(type(seed) is not int or not 0 <= seed < maximum for seed in values)):
        raise ValueError('strong_cache_seeds_invalid')


def _validate_link(owner, link, request, plan_count, *, cohort=False, context=None):
    """Saved commits are business proofs, never permission to repeat an input."""
    if link is None:
        return
    proof = link if cohort else link['proof']
    before, after = proof['before'], proof['after'] if cohort else proof['expected_state']
    committed = proof['committed']
    source = link['source_plan_index']
    predecessor = source if cohort else link['predecessor_plan_index']
    canonical = owner.kernel._canonical_probability_state
    if (type(committed) is not bool or not 0 <= _integer(source) <= _integer(predecessor) < plan_count
            or link['context'] != context
            # Native requests retain observed inventory/hand order while the
            # tree anchor is public-state canonical. Known draw order/prefix
            # stays significant in the existing kernel normalization.
            or canonical(after if committed else before) != canonical(request.state)):
        raise ValueError('strong_cache_continuation_proof_changed')
    slots_from_json(json_value(proof['after_slots'] if cohort else proof['expected_drink_slots']), after)
    if cohort:
        _seeds(link['seeds'], maximum=2**31)
        if len(link['seeds']) != 4:
            raise ValueError('strong_cache_cohort_size_changed')
    else:
        receipt = link['source_validation']
        if any(receipt.get(key) is not True for key in (
                'source_accepted', 'complete_root_coverage', 'complete_validation_coverage')):
            raise ValueError('strong_cache_continuation_incomplete_source')
        for key in ('training_paired_visits', 'validation_paired_visits'):
            if not receipt.get(key) or any(_integer(count) < 4 for count in receipt[key]):
                raise ValueError('strong_cache_continuation_incomplete_source')
        _validate_tables(owner.search, link['policy'])
    # Replay only the one already claimed deterministic transition. This is not
    # a terminal rollout, new scenario, or renewed planning budget.
    import time

    from .validation_cache import validation_session
    from .hif_public_turn_reanalysis import _StrictDeterministicKernel

    def guard():
        if time.perf_counter() >= owner.deadline:
            raise TimeoutError('strong_cache_restore_deadline')
    kernel = _StrictDeterministicKernel(owner.kernel.cards, owner.kernel.drinks,
        owner.kernel.items, memory_abilities=owner.kernel.memory_abilities)
    with validation_session(guard, reuse_parent=True):
        guard()
        branches = kernel.transition(before, replace(proof['action'], source_slot=None),
                                     plan_type=request.plan_type)
        guard()
    if (len(branches) != 1 or branches[0].probability != 1.
            or owner.kernel._canonical_probability_state(branches[0].state) != after):
        raise ValueError('strong_cache_continuation_transition_changed')


def _validate_tables(search, value):
    from .hif_mcts import action_key
    from .hif_internal_reanalysis import VERSION as INTERNAL_VERSION
    from .hif_internal_reanalysis import VALUE_SCOPE, Decision
    from .hif_resource_reanalysis import ResourcePolicy, ResourceDecision
    from .hif_public_turn_reanalysis import PublicTurnPolicy, public_key
    from .hif_internal_successor_graph import VERSION as GRAPH_VERSION
    from .hif_internal_successor_graph import GraphDecision

    if value is None:
        return
    if isinstance(value, PublicTurnPolicy):
        if not value.nodes or set(value.nodes) != set(value.choices) or value.root_key not in value.nodes:
            raise ValueError('strong_cache_public_graph_incomplete')
        for key, node in value.nodes.items():
            if key != public_key(node['request']):
                raise ValueError('strong_cache_public_graph_key_changed')
            request = node['request']
            from .planner import RobustHighScorePlanner
            legal = tuple(RobustHighScorePlanner._attach_source_slot(action, request)
                for action in search.kernel.legal_actions(request.state,
                    plan_type=request.plan_type, decision_kind=request.decision_kind))
            expected = {action_key(action) for action in legal}
            edges = node['edges']
            if len(edges) != len(expected) or {edge['action_key'] for edge in edges} != expected:
                raise ValueError('strong_cache_public_graph_actions_changed')
            options = value.choices[key]['options']
            if len(options) != len(expected) or {row['action_key'] for row in options} != expected:
                raise ValueError('strong_cache_public_graph_options_changed')
            maximum = max(row['mean_final_score'] for row in options)
            if (value.choices[key]['mean_final_score'] != maximum
                    or value.choices[key]['maximizing_actions'] != tuple(
                        row['action_key'] for row in options if row['mean_final_score'] == maximum)):
                raise ValueError('strong_cache_public_graph_values_changed')
            for edge in edges:
                if ('child' in edge) == ('boundary_index' in edge):
                    raise ValueError('strong_cache_public_graph_edge_invalid')
                if 'child' in edge and edge['child'] not in value.nodes:
                    raise ValueError('strong_cache_public_graph_child_missing')
        return
    if isinstance(value, ResourcePolicy):
        if not isinstance(value.decisions, tuple) or not value.decisions:
            raise ValueError('strong_cache_resource_decisions_invalid')
        for decision in value.decisions:
            if not isinstance(decision, ResourceDecision):
                raise ValueError('strong_cache_resource_decision_invalid')
            _validate_tables(search, decision.graph)
            if (decision.graph.root_key != public_key(decision.normalized_request)
                    or decision.options != decision.graph.choices[decision.graph.root_key]['options']
                    or decision.action_key not in decision.maximizing_actions):
                raise ValueError('strong_cache_resource_root_changed')
        return
    if isinstance(value, (Decision, GraphDecision)):
        identity = value.identity
        if len(identity) != 10 or identity[0] != INTERNAL_VERSION or identity[1] != search._namespace:
            raise ValueError('strong_cache_internal_context_changed')
        action_key_from_json(json_value(value.action_key))
        action_key_from_json(json_value(value.frozen_action))
        _seeds(value.planning_seeds)
        if isinstance(value, GraphDecision):
            if value.version != GRAPH_VERSION or value.complete is not True:
                raise ValueError('strong_cache_internal_graph_incomplete')
            _validate_tables(search, value.graph)
            from .hif_internal_successor_graph import _records
            if value.records != _records(search.kernel, value.graph):
                raise ValueError('strong_cache_internal_records_changed')
        elif value.value_scope != VALUE_SCOPE:
            raise ValueError('strong_cache_internal_scope_changed')
        return
    raise ValueError('strong_cache_table_type_invalid')


_SEARCH_FIELDS = ('_tree', '_rng', '_adaptive_rng', '_training_seed_history',
    '_baseline_domain_reason', '_namespace', '_expected_state', '_public_turn_policy',
    '_resource_policy', '_internal_policy', 'last_validation_cohort_evidence')


def _export_search(search, request, context):
    view = copy.copy(search)
    view._internal_policy = None
    bound = search._namespace is not None
    if not bound:
        # An exact/forced decision can finish without ever entering MCTS.
        # Retain that fact: the temporary view only reuses the strict codec.
        if search._tree or any(value is not None for value in (
                search._internal_policy, search._public_turn_policy, search._resource_policy)):
            raise ValueError('strong_cache_unbound_search_has_computation')
        canonical = search.kernel._canonical_probability_state(request.state)
        if search._expected_state is not None and search._expected_state != canonical:
            raise ValueError('strong_cache_unbound_public_state_changed')
        view._expected_state = canonical
        view._namespace = view._context(request)
    return dict(tree=export_tree(view, request, context),
        tables=encode_value((search._public_turn_policy, search._resource_policy, search._internal_policy),
                            context=search._namespace),
        cohort_evidence=json_value(search.last_validation_cohort_evidence),
        initialized=bound, expected_state=encode_value(search._expected_state))


def _prepare_search(search, raw, request, context):
    _fields(raw, ('tree', 'tables', 'cohort_evidence', 'initialized', 'expected_state'))
    if search._tree or search._namespace is not None or search._expected_state is not None:
        raise ValueError('strong_cache_restore_requires_fresh_search')
    staged = copy.copy(search)
    restore_tree(staged, raw['tree'], request, context)
    tables = decode_value(raw['tables'], kernel=search.kernel, search=search, context=staged._namespace)
    if not isinstance(tables, tuple) or len(tables) != 3:
        raise ValueError('strong_cache_tables_invalid')
    for table in tables:
        _validate_tables(staged, table)
    staged._public_turn_policy, staged._resource_policy, staged._internal_policy = tables
    expected = decode_value(raw['expected_state'], kernel=search.kernel)
    if (type(raw['initialized']) is not bool
            or expected is not None and expected != staged._expected_state
            or raw['initialized'] and expected is None):
        raise ValueError('strong_cache_search_binding_invalid')
    if not raw['initialized']:
        if staged._tree or any(table is not None for table in tables):
            raise ValueError('strong_cache_unbound_search_has_computation')
        staged._namespace, staged._expected_state = None, expected
    evidence = raw['cohort_evidence']
    _fields(evidence, staged.last_validation_cohort_evidence)
    for key in ('training_seeds', 'training_seed_history', 'validation_seeds', 'supplied_seeds'):
        _seeds(evidence[key])
    if (not set(evidence['training_seed_history']) <= staged._training_seed_history
            or set(evidence['validation_seeds']) & staged._training_seed_history):
        raise ValueError('strong_cache_cohort_training_overlap')
    staged.last_validation_cohort_evidence = copy.deepcopy(evidence)
    return staged


def _anchor(search, request):
    # Tree keys intentionally translate score and discard inert schedules; the
    # anchor retains the complete public state used by the frozen policy.
    slots = {key[1] for key in search._tree
             if key == search._state_key(search._expected_state, key[1])}
    if not slots and search._expected_state == request.state:
        slots.add(tuple(request.drink_slots))
    if len(slots) != 1:
        raise ValueError('strong_cache_frozen_anchor_ambiguous')
    state = search._expected_state
    ids = state.pending_choice.candidate_instance_ids if state.pending_choice else tuple(
        ref.instance_id for ref in state.deck.hand)
    return replace(request, state=state, drink_slots=next(iter(slots)),
        card_slots=tuple((identity, index) for index, identity in enumerate(ids)),
        decision_kind=contracts.DecisionKind.SECONDARY if state.pending_choice else contracts.DecisionKind.LIVE)


def _export_retained(owner, request):
    controller = owner._retained_policy
    if controller is None:
        return None
    frozen = controller.frozen
    result = {name: value for name, value in vars(controller).items() if name not in ('owner', 'frozen')}
    data = encode_value(result, context=owner.search._namespace)
    if frozen is None:
        return dict(data=data, frozen=None)
    frozen._verify_frozen()
    anchor = _anchor(frozen.policy.search, request)
    return dict(data=data, frozen=dict(anchor=public_request(anchor),
        search=_export_search(frozen.policy.search, anchor, owner._cache_context()),
        source_plan_index=frozen.source_plan_index, allow_complete=frozen.allow_complete,
        revealed_policy_version=frozen.revealed_policy_version,
        validation_excluded_seeds=list(frozen.validation_excluded_seeds),
        action=encode_value(frozen.action), request=encode_value(frozen.request)))


def _same_public_request(kernel, raw, request):
    """Compare full public facts, preserving known draw order and all resources."""
    saved = request_from_public(raw, kernel)
    canonical = kernel._canonical_probability_state
    return (public_request(replace(saved, state=canonical(saved.state)))
            == public_request(replace(request, state=canonical(request.state))))


def _prepare_retained(owner, raw, request, plan_count, context):
    from .hif_public_graph_parallel import _frozen_signature
    from .hif_retained_policy_competition import FrozenTreePolicy, _complete_graph

    current = owner._retained_policy
    if current is None:
        if raw is not None:
            raise ValueError('strong_cache_retained_configuration_changed')
        return None
    _fields(raw, ('data', 'frozen'))
    result = type(current)(owner)
    saved = decode_value(raw['data'], kernel=owner.kernel, search=owner.search,
                         context=context)
    _fields(saved, set(vars(result)) - {'owner', 'frozen'})
    for name in ('active', 'revealed'):
        if name in saved and type(saved[name]) is not bool:
            raise ValueError('strong_cache_retained_flag_invalid')
    for name in ('predecessor_plan_index', 'execution_source_plan_index', 'graph_execution_source_plan_index'):
        if saved[name] is not None and not 0 <= _integer(saved[name]) < plan_count:
            raise ValueError('strong_cache_retained_plan_invalid')
    if saved['context'] is not None and saved['context'] != owner._cache_context():
        raise ValueError('strong_cache_retained_context_changed')
    if 'validation_seeds' in saved:
        _seeds(saved['validation_seeds'])
    result.__dict__.update(saved)
    frozen = raw['frozen']
    if frozen is None:
        if saved['active'] or saved['predecessor_plan_index'] is not None:
            raise ValueError('strong_cache_retained_policy_missing')
        return result
    _fields(frozen, ('anchor', 'search', 'source_plan_index', 'allow_complete',
        'revealed_policy_version', 'validation_excluded_seeds', 'action', 'request'))
    if (not 0 <= _integer(frozen['source_plan_index']) < plan_count
            or type(frozen['allow_complete']) is not bool):
        raise ValueError('strong_cache_frozen_source_invalid')
    anchor = request_from_public(frozen['anchor'], owner.kernel)
    candidate = FrozenTreePolicy(owner, anchor, frozen['source_plan_index'], owner.deadline,
                                 allow_complete=frozen['allow_complete'])
    target = candidate.policy.search
    target._tree, target._expected_state, target._namespace = {}, None, None
    staged = _prepare_search(target, frozen['search'], anchor, owner._cache_context())
    for name in _SEARCH_FIELDS:
        setattr(target, name, getattr(staged, name))
    if candidate.allow_complete and target._public_turn_policy is not None:
        target._public_turn_policy = _complete_graph(target._public_turn_policy)
    candidate.source_tables = dict(public_turn=target._public_turn_policy is not None,
        resource=target._resource_policy is not None, internal=target._internal_policy is not None)
    candidate._table_signature = copy.deepcopy(candidate._tables())
    candidate._resource_key = target._resource_policy.key if target._resource_policy else None
    candidate._signature = _frozen_signature(target)
    candidate.revealed_policy_version = frozen['revealed_policy_version']
    from .hif_retained_policy_competition import REVEALED_VERSION
    if candidate.revealed_policy_version not in (None, REVEALED_VERSION):
        raise ValueError('strong_cache_revealed_version_changed')
    _seeds(frozen['validation_excluded_seeds'])
    candidate.validation_excluded_seeds = tuple(frozen['validation_excluded_seeds'])
    candidate.action = decode_value(frozen['action'], kernel=owner.kernel)
    candidate.request = decode_value(frozen['request'], kernel=owner.kernel)
    if (candidate.action is None) != (candidate.request is None):
        raise ValueError('strong_cache_frozen_binding_incomplete')
    result.frozen = candidate
    if hasattr(result, 'proof'):
        proof = result.proof
        if (not isinstance(proof, dict) or type(proof.get('committed')) is not bool
                or result._pending_request is None or result._pending_action is None
                or proof['before'] != public_request(result._pending_request)
                or proof['action'] != json_value(asdict(result._pending_action))
                or proof['source_plan_index'] != candidate.source_plan_index
                or proof['predecessor_plan_index'] != result.predecessor_plan_index
                or proof['ruleset_id'] != request.ruleset_id
                or proof['protocol_version'] != request.protocol_version
                or not _same_public_request(owner.kernel,
                    proof['after' if proof['committed'] else 'before'], request)):
            raise ValueError('strong_cache_revealed_commit_proof_changed')
    return result


_CONTROLLERS = ('_validation_cohort', '_public_turn_continuation', '_resource_continuation')
_WORK_COUNTS = ('completed_simulations', 'started_simulations', 'training_completed_simulations',
    'training_started_simulations', 'validation_completed_simulations', 'validation_started_simulations')


def _work_receipt(owner):
    totals = dict(getattr(owner, '_strong_work_totals', {}))
    last = copy.deepcopy(getattr(owner, '_strong_last_window_work', None))
    for index, plan in enumerate(owner.plans, owner._plan_offset):
        counts = {name: _integer(plan[name]) for name in _WORK_COUNTS if name in plan}
        for name, value in counts.items():
            totals[name] = totals.get(name, 0) + value
        last = dict(plan_index=index, counts=counts)
    return totals, last


def _validate_work(totals, last, plan_count):
    if not isinstance(totals, dict) or not set(totals) <= set(_WORK_COUNTS):
        raise ValueError('strong_cache_work_counts_invalid')
    for value in totals.values():
        _integer(value)
    if last is not None:
        _fields(last, ('plan_index', 'counts'))
        if (not 0 <= _integer(last['plan_index']) < plan_count
                or not isinstance(last['counts'], dict) or not set(last['counts']) <= set(totals)
                or any(_integer(value) > totals[name] for name, value in last['counts'].items())):
            raise ValueError('strong_cache_last_window_work_invalid')


def export_candidate_state(owner, request):
    """Export completed computation only; owner handles action and tail links."""
    if not owner.quality_reference:
        raise ValueError('strong_cache_requires_quality_reference')
    if owner._session is not None or owner._tail_session is not None:
        raise ValueError('mcts_cache_window_still_running')
    if owner.plans and owner.plans[-1].get('accepted') is not True:
        raise ValueError('strong_cache_incomplete_window')
    context = owner.search._namespace
    controllers = {}
    for name in _CONTROLLERS:
        controller = getattr(owner, name)
        controllers[name] = None if controller is None else encode_value({
            key: value for key, value in vars(controller).items() if key != 'kernel'}, context=context)
    totals, last = _work_receipt(owner)
    return json_value(dict(version=VERSION, definitions=definitions(owner.kernel),
        context=owner._cache_context(), plan_count=owner._plan_offset + len(owner.plans),
        search=_export_search(owner.search, request, owner._cache_context()),
        retained=_export_retained(owner, request), controllers=controllers,
        work_totals=totals, last_window_work=last))


def prepare_restore_candidate_state(owner, raw, request):
    """Validate without mutation; returned apply() installs only owned fields."""
    if (owner.plans or owner._plan_offset or owner._session is not None or owner._tail_session is not None
            or not owner.quality_reference):
        raise ValueError('strong_cache_requires_new_candidate')
    from .hif_mcts_cache import require_candidate_tree_versions

    require_candidate_tree_versions(dict(computation=raw), owner.search, complete_policy=True)
    raw = json_value(raw)
    _fields(raw, ('version', 'definitions', 'context', 'plan_count', 'search', 'retained', 'controllers',
                  'work_totals', 'last_window_work'))
    if (raw['version'] != VERSION or not same_json(raw['definitions'], definitions(owner.kernel))
            or not same_json(raw['context'], owner._cache_context())):
        raise ValueError('strong_cache_configuration_changed')
    plan_count = _integer(raw['plan_count'])
    _validate_work(raw['work_totals'], raw['last_window_work'], plan_count)
    search = _prepare_search(owner.search, raw['search'], request, owner._cache_context())
    _fields(raw['controllers'], _CONTROLLERS)
    controllers = {}
    for name in _CONTROLLERS:
        current, saved = getattr(owner, name), raw['controllers'][name]
        if (current is None) != (saved is None):
            raise ValueError('strong_cache_controller_configuration_changed')
        if current is None:
            controllers[name] = None
            continue
        data = decode_value(saved, kernel=owner.kernel, search=owner.search, context=search._namespace)
        _fields(data, set(vars(current)) - {'kernel'})
        restored = type(current)(owner.kernel)
        restored.__dict__.update(data)
        if name == '_validation_cohort':
            _validate_link(owner, restored._link, request, plan_count, cohort=True, context=search._namespace)
            _fields(restored._summary, ('version', 'reused', 'source_plan_index', 'reason', 'seeds'))
            if (restored._summary['version'] != current.version
                    or type(restored._summary['reused']) is not bool):
                raise ValueError('strong_cache_cohort_summary_invalid')
            _seeds(restored._summary['seeds'], maximum=2**31)
        else:
            if restored._active is not None and restored._pending is not None:
                raise ValueError('strong_cache_continuation_two_pending_states')
            for link in (restored._active, restored._pending):
                _validate_link(owner, link, request, plan_count, context=search._namespace)
        controllers[name] = restored
    retained = _prepare_retained(owner, raw['retained'], request, plan_count, search._namespace)
    used = False
    def apply():
        nonlocal used
        if used or owner.search._tree or owner.search._namespace is not None:
            raise ValueError('strong_cache_restore_already_applied')
        for name in _SEARCH_FIELDS:
            setattr(owner.search, name, getattr(search, name))
        for name, controller in controllers.items():
            setattr(owner, name, controller)
        owner._retained_policy = retained
        owner._plan_offset = plan_count
        owner._strong_work_totals = copy.deepcopy(raw['work_totals'])
        owner._strong_last_window_work = copy.deepcopy(raw['last_window_work'])
        used = True
        return dict(restored_nodes=len(search._tree), restored_visits=sum(
            values[0] for node in search._tree.values() for values in node.statistics.values()),
            restored_complete_policy=True, restored_plan_count=plan_count,
            restored_work_totals=copy.deepcopy(raw['work_totals']))
    return apply


def restore_candidate_state(owner, raw, request):
    return prepare_restore_candidate_state(owner, raw, request)()
