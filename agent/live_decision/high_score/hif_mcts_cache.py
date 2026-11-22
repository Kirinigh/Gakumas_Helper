"""Portable MCTS computation state, never observation or input authority."""
from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from dataclasses import asdict

from .golden import _state
from .contracts import (
    PlanType,
    ActionKind,
    GameAction,
    RiskProfile,
    UtilitySpec,
    DecisionKind,
    DecisionRequest,
    CapabilityReport,
    _jsonable,
)

# Selected retrieval now obeys HIF hand capacity. Old computed trees are stale
# even with identical catalog definitions/feature schema. As with the previous
# score-arithmetic correction, business observations keep their existing codec.
# Restored HIF freshness dispatch changes complete values and frozen policies.
VERSION = 'hif-mcts-search-cache-v4'


def _version(search):
    version = (VERSION + ':' + search._instance_view_version
               if search.equivalent_instance_reuse else VERSION)
    if search.deterministic_validation_cohort:
        from .hif_mcts import DETERMINISTIC_VALIDATION_COHORT_VERSION

        version += ':' + DETERMINISTIC_VALIDATION_COHORT_VERSION
    if search.partial_node_timing:
        from .hif_mcts import PARTIAL_NODE_TIMING_VERSION

        version += ':' + PARTIAL_NODE_TIMING_VERSION
    if search.internal_reanalysis:
        version += ':' + search._internal_reanalysis_version
    if search.internal_successor_graph:
        version += ':' + search._internal_successor_graph_version
    if search.uniform_validation_time_guard:
        version += ':' + search._uniform_validation_time_guard_version
    if search.paired_adaptive_validation:
        version += ':' + search._paired_adaptive_validation_version
    if search.parallel_public_graph and (search.paired_adaptive_validation or search.uniform_validation_time_guard):
        version += ':' + search._parallel_root_validation_version
    if search.internal_complete_root_successors:
        version += ':' + search._internal_complete_root_successors_version
    installed = getattr(search.unknown_node_policy, 'installed_score_policy', None)
    if installed is not None:
        version += ':' + installed.version
    extra = getattr(search.unknown_node_policy, 'extra_action_policy', None)
    if extra is not None:
        version += ':' + extra.version
    pair_order = getattr(search.unknown_node_policy, 'pair_order_policy', None)
    if pair_order is not None:
        version += ':' + pair_order.version
    return version


def json_value(value):
    """Detach a JSON value and reject non-finite or non-business objects."""
    return json.loads(json.dumps(_jsonable(value), allow_nan=False))


def same_json(left, right):
    """Numeric representation may round-trip; booleans cannot become numbers."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(same_json(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_json(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    return type(left) is type(right) and left == right


def same_pending_plan(cached, observed):
    """Response provenance may be attached after the pending plan was saved."""
    from .hif_strong_reference import REFERENCE_COMMIT

    rows = []
    for value in (cached, observed):
        if not isinstance(value, dict):
            return False
        value = dict(value)
        for key, expected in (('reference_commit', REFERENCE_COMMIT), ('independent_quality_passed', False)):
            if key in value and not same_json(value.pop(key), expected):
                return False
        rows.append(value)
    return same_json(*rows)


def exact_state(raw):
    # Same complete-state codec as native_settlement_resume, without its
    # historical-schema normalization. A new cache never needs old defaults.
    normalized = copy.deepcopy(raw)
    for zone in ('hand', 'draw_pile', 'discard', 'removed', 'held'):
        normalized['deck'][zone] = [card['instance_id'] for card in raw['deck'][zone]]
    state = _state(normalized)
    if not same_json(json_value(asdict(state)), raw):
        raise ValueError('mcts_cache_state_not_exact')
    state.validate()
    return state


def slots_from_json(raw, state):
    if (not isinstance(raw, list) or any(not isinstance(pair, list) or len(pair) != 2
            or type(pair[0]) is not int or not 0 <= pair[0] < 4
            or not isinstance(pair[1], str) for pair in raw)):
        raise ValueError('mcts_cache_drink_slots_invalid')
    slots = tuple(tuple(pair) for pair in raw)
    if (len({slot for slot, _ in slots}) != len(slots)
            or Counter(drink for _, drink in slots) != Counter(state.drinks)):
        raise ValueError('mcts_cache_drink_slots_invalid')
    return slots


def verified_drink_slot_map(state, source_slots, observed_slots):
    """Match unique bottles after complete settlement, never input evidence."""
    source = slots_from_json(json_value(source_slots), state)
    observed = slots_from_json(json_value(observed_slots), state)
    if source == observed:
        return {pair: pair for pair in source}
    if len({drink for _, drink in source}) != len(source):
        raise ValueError('mcts_cache_drink_rebind_ambiguous_identity')
    by_id = {drink: slot for slot, drink in observed}
    return {(slot, drink): (by_id[drink], drink) for slot, drink in source}


def rebind_retained_drink_slots(search, state, source_slots, observed_slots):
    """Relabel only the verified successor subtree; preserve all computation."""
    from .hif_mcts import _Node

    mapping = verified_drink_slot_map(state, source_slots, observed_slots)
    if tuple(source_slots) == tuple(observed_slots):
        return
    if search._expected_state != search.kernel._canonical_probability_state(state):
        raise ValueError('mcts_cache_drink_rebind_state_changed')
    rank = {tuple(pair): index for index, pair in enumerate(observed_slots)}

    def remap_key(key):
        current, slots = key
        slots_from_json(json_value(slots), current)
        if any(pair not in mapping for pair in slots):
            raise ValueError('mcts_cache_drink_rebind_descendant_changed')
        return current, tuple(sorted((mapping[pair] for pair in slots), key=rank.__getitem__))

    def remap_action(key, slots):
        if key[0] != ActionKind.USE_DRINK.value:
            return key
        pair = (key[-1], key[2])
        if pair not in slots or pair not in mapping:
            raise ValueError('mcts_cache_drink_rebind_action_changed')
        return (*key[:-1], mapping[pair][0])

    pending = [search._state_key(state, source_slots)]
    visited, tree = set(), {}
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        rebound = remap_key(key)
        if key not in search._tree:
            continue
        if rebound in tree:
            raise ValueError('mcts_cache_drink_rebind_collision')
        node = search._tree[key]
        statistics = {remap_action(action, key[1]): list(values)
                      for action, values in node.statistics.items()}
        order = tuple(remap_action(action, key[1]) for action in node.order)
        if len(statistics) != len(node.statistics) or len(set(order)) != len(order):
            raise ValueError('mcts_cache_drink_rebind_collision')
        tree[rebound] = _Node(statistics, order, {remap_key(child) for child in node.children})
        pending.extend(node.children)
    # No sampling or evaluation occurs; mutate only after the whole subtree
    # has a one-to-one relabeling. RNG, states, statistics and ordering survive.
    search._tree = tree


def public_request(request):
    return json_value(dict(state=asdict(request.state), drink_slots=request.drink_slots,
        plan_type=request.plan_type, utility=asdict(request.utility)))


def request_from_public(raw, kernel):
    if set(raw) != {'state', 'drink_slots', 'plan_type', 'utility'}:
        raise ValueError('mcts_cache_public_request_invalid')
    state = exact_state(raw['state'])
    utility = UtilitySpec(**{**raw['utility'], 'risk_profile': RiskProfile(raw['utility']['risk_profile'])})
    if not same_json(json_value(asdict(utility)), raw['utility']):
        raise ValueError('mcts_cache_utility_not_exact')
    ids = (state.pending_choice.candidate_instance_ids if state.pending_choice else
           tuple(card.instance_id for card in state.deck.hand))
    request = DecisionRequest(snapshot_id='search-cache-business-only',
        plan_type=PlanType(raw['plan_type']),
        decision_kind=DecisionKind.SECONDARY if state.pending_choice else DecisionKind.LIVE,
        state=state, utility=utility, ruleset_id=kernel.ruleset_id,
        # Deliberately incapable of authorizing a decision or input.
        capabilities=CapabilityReport(False, False, False, False, False),
        card_slots=tuple((identity, index) for index, identity in enumerate(ids)),
        drink_slots=slots_from_json(raw['drink_slots'], state))
    request.validate()
    return request


def action_from_json(raw):
    action = GameAction(**{**raw, 'kind': ActionKind(raw['kind'])})
    action.validate()
    if not same_json(json_value(asdict(action)), raw):
        raise ValueError('mcts_cache_action_not_exact')
    return action


def action_key_from_json(raw):
    if not isinstance(raw, list) or len(raw) != 5:
        raise ValueError('mcts_cache_action_key_invalid')
    action = action_from_json(dict(zip(
        ('kind', 'card_instance_id', 'drink_id', 'target_id', 'source_slot'), raw)))
    if action.source_slot is not None and (action.kind is not ActionKind.USE_DRINK
            or type(action.source_slot) is not int or not 0 <= action.source_slot < 4):
        raise ValueError('mcts_cache_action_key_invalid')
    return tuple(raw)


def definitions(kernel):
    return json_value(dict(ruleset_id=kernel.ruleset_id, **{
        name: {key: asdict(value) for key, value in sorted(getattr(kernel, name).items())}
        for name in ('cards', 'drinks', 'items', 'memory_abilities')}))


def settings(search):
    names = ('simulations', 'timeout_ms', 'seed', 'min_root_samples', 'reward_scale',
        'soft_timeout_ms', 'scenario_reuse', 'validation_samples', 'dominance_pruning',
        'adaptive_validation', 'progressive_widening', 'terminal_horizon',
        'terminal_eligibility_horizons', 'continuation_samples', 'continuation_domain',
        'continuation_incomplete_actions', 'fresh_adaptive_training', 'continuation_portfolio')
    result = {name: getattr(search, name) for name in names}
    if search.quality_reference:
        from .hif_mcts import QUALITY_REFERENCE_VERSION

        result['quality_reference'] = True
        result['quality_reference_version'] = QUALITY_REFERENCE_VERSION
    if search.revealed_policy_competition:
        from .hif_mcts import REVEALED_POLICY_COMPETITION_VERSION

        result['revealed_policy_competition'] = True
        result['revealed_policy_competition_version'] = REVEALED_POLICY_COMPETITION_VERSION
    if search.compute_sensitivity:
        from .hif_mcts import COMPUTE_SENSITIVITY_VERSION

        result['compute_sensitivity'] = True
        result['compute_sensitivity_version'] = COMPUTE_SENSITIVITY_VERSION
    if search.single_node_expansion:
        result['single_node_expansion'] = True
    if search.stop_after_card_roots:
        result['stop_after_card_roots'] = True
    if getattr(search.turn_chain_continuation, 'installed_card_use_value', False):
        result['installed_card_use_value'] = True
    if search.root_coverage_first:
        result['root_coverage_first'] = True
    if search.equivalent_instance_reuse:
        result['equivalent_instance_reuse'] = True
    if search.paired_finalist_validation:
        result['paired_finalist_validation'] = True
    if search.uniform_validation_time_guard:
        result['uniform_validation_time_guard'] = True
        result['uniform_validation_time_guard_version'] = search._uniform_validation_time_guard_version
    if search.paired_adaptive_validation:
        result['paired_adaptive_validation'] = True
        result['paired_adaptive_validation_version'] = search._paired_adaptive_validation_version
    if search.public_turn_reanalysis:
        result['public_turn_reanalysis'] = True
        result['public_turn_reanalysis_version'] = search._public_turn_reanalysis_version
        result['score_translation_contract'] = search.score_translation_contract
    if search.resource_reanalysis:
        result['resource_reanalysis'] = True
        result['resource_reanalysis_version'] = search._resource_reanalysis_version
    if search.internal_reanalysis:
        result['internal_reanalysis'] = True
        result['internal_reanalysis_version'] = search._internal_reanalysis_version
    if search.internal_successor_graph:
        result['internal_successor_graph'] = True
        result['internal_successor_graph_version'] = search._internal_successor_graph_version
    if search.internal_complete_root_successors:
        result['internal_complete_root_successors'] = True
        result['internal_complete_root_successors_version'] = search._internal_complete_root_successors_version
    if search.public_turn_budget_reservation:
        from .hif_mcts import PUBLIC_TURN_BUDGET_RESERVATION_VERSION

        result['public_turn_budget_reservation'] = True
        result['public_turn_budget_reservation_version'] = PUBLIC_TURN_BUDGET_RESERVATION_VERSION
    if search.public_validation_phase_budget:
        from .hif_mcts import PUBLIC_VALIDATION_PHASE_BUDGET_VERSION

        result['public_validation_phase_budget'] = True
        result['public_validation_phase_budget_version'] = PUBLIC_VALIDATION_PHASE_BUDGET_VERSION
    if search.parallel_public_graph and (search.paired_adaptive_validation or search.uniform_validation_time_guard):
        result['parallel_root_validation_version'] = search._parallel_root_validation_version
    if search.parallel_public_graph:
        result['parallel_public_graph'] = True
        result['parallel_public_graph_version'] = search._parallel_public_graph_version
    if getattr(search, 'direct_double_marginal_version', None) is not None:
        result['direct_double_marginal'] = True
        result['direct_double_marginal_version'] = search.direct_double_marginal_version
    if search.unknown_node_policy is not None:
        result['unknown_node_timing'] = dict(version=search.unknown_node_policy.version,
            timing_version=search.unknown_node_policy.timing_version)
    installed = getattr(search.unknown_node_policy, 'installed_score_policy', None)
    if installed is not None:
        result['installed_score_turn_chain_version'] = installed.version
    extra = getattr(search.unknown_node_policy, 'extra_action_policy', None)
    if extra is not None:
        result['extra_action_turn_chain_version'] = extra.version
    pair_order = getattr(search.unknown_node_policy, 'pair_order_policy', None)
    if pair_order is not None:
        result['pair_order_advice_version'] = pair_order.version
    if search.partial_node_timing:
        from .hif_mcts import PARTIAL_NODE_TIMING_VERSION

        result['partial_node_timing_version'] = PARTIAL_NODE_TIMING_VERSION
    if search.retrieval_training_orderer is not None:
        result['retrieval_training_order'] = dict(version=search.retrieval_training_orderer.version,
            unknown_version=search.retrieval_training_orderer.unknown_version,
            timing_version=search.retrieval_training_orderer.timing_version)
    if search.lazy_validation_seeds:
        from .hif_mcts import LAZY_VALIDATION_SEEDS_VERSION

        result['lazy_validation_seeds_version'] = LAZY_VALIDATION_SEEDS_VERSION
    if search.deterministic_validation_cohort:
        from .hif_mcts import DETERMINISTIC_VALIDATION_COHORT_VERSION

        result['deterministic_validation_cohort_version'] = DETERMINISTIC_VALIDATION_COHORT_VERSION
    return json_value(result)


def rng_from_json(raw):
    def tuples(value):
        return tuple(map(tuples, value)) if isinstance(value, list) else value
    rng = random.Random()
    rng.setstate(tuples(raw))
    return rng


def export_tree(search, request, version_context):
    if search._internal_policy is not None:
        raise ValueError('mcts_internal_policy_in_process_only')
    request = request_from_public(public_request(request), search.kernel)
    if (request.state != search._expected_state
            or search._namespace != search._context(request)):
        raise ValueError('mcts_cache_export_context_changed')
    keys = dict.fromkeys(search._tree)
    for node in search._tree.values():
        for child in node.children:
            keys.setdefault(child, None)
    indices = {key: index for index, key in enumerate(keys)}
    result = dict(version=_version(search), context=version_context,
        definitions=definitions(search.kernel), settings=settings(search),
        anchor=public_request(request), baseline_domain_reason=search._baseline_domain_reason,
        states=[dict(state=asdict(state), drink_slots=slots) for state, slots in keys],
        nodes=[dict(state_index=indices[key],
            statistics=[[action, *values] for action, values in node.statistics.items()],
            order=node.order, children=sorted(indices[child] for child in node.children))
            for key, node in search._tree.items()],
        rng=search._rng.getstate(), adaptive_rng=search._adaptive_rng.getstate())
    if search.deterministic_validation_cohort:
        result['training_seed_history'] = sorted(search._training_seed_history)
    return json_value(result)


def require_current_tree_version(raw, search=None):
    """Reject old computational semantics before any business-state decoder."""
    if not isinstance(raw, dict) or not isinstance(raw.get('version'), str) or not (
            raw['version'] == VERSION or raw['version'].startswith(VERSION + ':')):
        raise ValueError('mcts_cache_version_or_definitions_changed')
    if search is not None and raw['version'] != _version(search):
        raise ValueError('mcts_cache_version_or_definitions_changed')


def require_candidate_tree_versions(payload, search, *, complete_policy):
    tree = payload['computation']['search']['tree'] if complete_policy else payload['search']
    require_current_tree_version(tree, search)
    if complete_policy:
        retained = payload['computation'].get('retained')
        if isinstance(retained, dict) and retained.get('frozen') is not None:
            require_current_tree_version(retained['frozen']['search']['tree'])


def restore_tree(search, raw, request, version_context):
    require_current_tree_version(raw, search)
    from .hif_mcts import _Node, action_key

    raw = json_value(raw)
    expected_fields = {'version', 'context', 'definitions', 'settings', 'anchor',
        'baseline_domain_reason', 'states', 'nodes', 'rng', 'adaptive_rng'}
    if search.deterministic_validation_cohort:
        expected_fields.add('training_seed_history')
    if (set(raw) != expected_fields or raw['version'] != _version(search)
            or not same_json(raw['context'], json_value(version_context))
            or not same_json(raw['definitions'], definitions(search.kernel))
            or not same_json(raw['settings'], settings(search))):
        raise ValueError('mcts_cache_version_or_definitions_changed')
    if (search._tree or search._expected_state is not None or search._namespace is not None):
        raise ValueError('mcts_cache_restore_requires_new_search')
    training_seed_history = raw.get('training_seed_history')
    history_limit = (2**32 if search.quality_reference and search.fresh_adaptive_training
                     and search.internal_reanalysis else 2**31)
    if search.deterministic_validation_cohort and (
            not isinstance(training_seed_history, list)
            or any(type(seed) is not int or not 0 <= seed < history_limit for seed in training_seed_history)
            or training_seed_history != sorted(set(training_seed_history))):
        raise ValueError('mcts_cache_training_seed_history_invalid')
    request.validate()
    canonical = search.kernel._canonical_probability_state(request.state)
    anchor = request_from_public(raw['anchor'], search.kernel)
    if (anchor.state != canonical or anchor.utility != request.utility
            or anchor.plan_type != request.plan_type or anchor.drink_slots != request.drink_slots):
        raise ValueError('mcts_cache_public_state_changed')
    reason = raw['baseline_domain_reason']
    if reason is not None and (reason != 'joint_root_expansion_budget_ineligible'
                              or not search.continuation_domain):
        raise ValueError('mcts_cache_baseline_domain_invalid')
    keys = []
    for row in raw['states']:
        if set(row) != {'state', 'drink_slots'}:
            raise ValueError('mcts_cache_state_key_invalid')
        state = exact_state(row['state'])
        search.kernel._validate_registered_state(state, plan_type=request.plan_type)
        if state.score != 0 or search.kernel._canonical_probability_state(state) != state:
            raise ValueError('mcts_cache_state_key_not_canonical')
        # v1 sidecars may retain exhausted observation counters in their
        # keys. Validate every original field first, then use the same key
        # normalization as new search nodes. Never combine saved statistics
        # if distinct old keys collapse to one current key.
        key = search._state_key(state, slots_from_json(row['drink_slots'], state))
        if search.equivalent_instance_reuse and key[0] != state:
            raise ValueError('mcts_cache_instance_view_not_canonical')
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError('mcts_cache_duplicate_or_equivalent_state')
    tree = {}
    def index(value):
        if type(value) is not int or not 0 <= value < len(keys):
            raise ValueError('mcts_cache_node_index_invalid')
        return keys[value]
    for row in raw['nodes']:
        if set(row) != {'state_index', 'statistics', 'order', 'children'}:
            raise ValueError('mcts_cache_node_invalid')
        key = index(row['state_index'])
        if key in tree:
            raise ValueError('mcts_cache_duplicate_node')
        statistics = {}
        for entry in row['statistics']:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError('mcts_cache_statistics_invalid')
            action = action_key_from_json(entry[0])
            visits, total = entry[1:]
            if (action in statistics or type(visits) is not int or visits < 0
                    or type(total) not in (int, float) or not math.isfinite(total)
                    or visits == 0 and total != 0):
                raise ValueError('mcts_cache_statistics_invalid')
            statistics[action] = [visits, total]
        order = tuple(action_key_from_json(action) for action in row['order'])
        if not order or len(order) != len(set(order)) or set(order) != set(statistics):
            raise ValueError('mcts_cache_order_invalid')
        state, slots = key
        legal_keys = set()
        for action in search.kernel.legal_actions(state, plan_type=request.plan_type,
                decision_kind=DecisionKind.SECONDARY if state.pending_choice else DecisionKind.LIVE):
            parts = list(action_key(action))
            if action.kind is ActionKind.USE_DRINK:
                parts[-1] = next(slot for slot, drink in slots if drink == action.drink_id)
            legal_keys.add(tuple(parts))
        if state.terminal or set(statistics) != legal_keys:
            raise ValueError('mcts_cache_legal_actions_changed')
        children = {index(value) for value in row['children']}
        if len(children) != len(row['children']):
            raise ValueError('mcts_cache_children_invalid')
        tree[key] = _Node(statistics, order, children)
    if (search.deterministic_validation_cohort and not training_seed_history
            and any(values[0] for node in tree.values() for values in node.statistics.values())):
        raise ValueError('mcts_cache_training_seed_history_invalid')
    rng, adaptive_rng = rng_from_json(raw['rng']), rng_from_json(raw['adaptive_rng'])
    # All validation precedes the first mutation. Runtime identities and
    # deadlines come from this instance, never from the stored process.
    namespace = search._context(request)
    staged = copy.copy(search)
    staged._baseline_domain_reason = reason
    staged._namespace = namespace
    staged._expected_state = canonical
    staged._tree, staged._rng, staged._adaptive_rng = tree, rng, adaptive_rng
    staged._resource_policy = None
    staged._internal_policy = None
    if search.deterministic_validation_cohort:
        staged._training_seed_history = set(training_seed_history)
        staged._reset_validation_cohort_evidence()
    receipt = dict(restored_nodes=len(tree), restored_visits=sum(
        values[0] for node in tree.values() for values in node.statistics.values()))
    search.__dict__.update(staged.__dict__)
    return receipt
