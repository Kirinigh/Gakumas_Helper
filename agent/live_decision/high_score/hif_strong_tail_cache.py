"""Pure-data persistence of completed exact-tail work for the strong trial."""
from __future__ import annotations

import math
from dataclasses import replace

from .planner import RobustHighScorePlanner, _Outcome
from .contracts import PlanType, ActionKind, DecisionKind
from .hif_search_session import HifSearchSession
from .hif_terminal_cache import TerminalMarginalCache

# Exact outcomes also change with HIF freshness dispatch, independent of trees.
VERSION = 'hif-strong-complete-tail-cache-v2'


def _codec(kernel=None):
    from .hif_strong_policy_cache import decode_value, encode_value

    return encode_value, lambda raw: decode_value(raw, kernel=kernel)


def _outcomes(values):
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(v, _Outcome) or not math.isfinite(v.score)
                   or not math.isfinite(v.probability) or v.probability < 0
                   or type(v.stamina) is not int or v.stamina < 0
                   or type(v.genki) is not int or v.genki < 0 for v in values)
            or not math.isclose(sum(v.probability for v in values), 1., abs_tol=1e-8)):
        raise ValueError('strong_tail_invalid_complete_distribution')


def _namespace_metadata(namespace, planner):
    if (len(namespace) != 15 or namespace[0] != id(planner.kernel)
            or namespace[1] is not type(planner.kernel)
            or namespace[3] is not type(planner)
            or namespace[2] != planner.kernel.ruleset_id
            or namespace[4:7] != (planner.planner_version, planner.terminal_cache,
                                  planner.complete_horizon)
            or namespace[11:] != tuple(tuple(sorted(getattr(planner.kernel, name).items()))
                for name in ('cards', 'drinks', 'items', 'memory_abilities'))):
        raise ValueError('strong_tail_namespace_changed')
    return namespace[7:11]


def export_session(session, planner):
    encode, _ = _codec()
    if session._binding.locked():
        raise ValueError('strong_tail_window_still_running')
    rows = []
    for namespace, (memo, marginal) in session._namespaces.items():
        metadata = _namespace_metadata(namespace, planner)
        universes = list(marginal.universes.items())
        indices = {identity: index for index, (identity, _) in enumerate(universes)}

        def table(values):
            result = {}
            for key, value in values.items():
                if key[0] not in indices:
                    raise ValueError('strong_tail_marginal_universe_missing')
                result[(indices[key[0]], *key[1:])] = value
            return result

        for values in (*memo.values(), *marginal.values.values()):
            _outcomes(values)
        rows.append(dict(metadata=encode(metadata), memo=encode(memo),
            universes=encode(tuple(cards for _, cards in universes)),
            eligibility=encode(table(marginal.eligibility)), values=encode(table(marginal.values)),
            transitions=encode(table(marginal.transitions)), hits=marginal.hits,
            transition_hits=marginal.transition_hits))
    return dict(version=VERSION, namespaces=rows, entries=session.entry_count)


def restore_session(raw, planner, request, deadline):
    _, decode = _codec(planner.kernel)
    if (not isinstance(raw, dict) or set(raw) != {'version', 'namespaces', 'entries'}
            or raw['version'] != VERSION or not isinstance(raw['namespaces'], list)
            or type(raw['entries']) is not int or not 0 < raw['entries'] <= 32768
            or not math.isfinite(deadline)):
        raise ValueError('strong_tail_invalid_session')
    session = HifSearchSession(deadline)
    seen = set()
    for row in raw['namespaces']:
        if set(row) != {'metadata', 'memo', 'universes', 'eligibility', 'values',
                        'transitions', 'hits', 'transition_hits'}:
            raise ValueError('strong_tail_invalid_namespace_fields')
        metadata = decode(row['metadata'])
        if (not isinstance(metadata, tuple) or len(metadata) != 4
                or metadata[0] is not PlanType.SENSE or not isinstance(metadata[2], DecisionKind)
                or type(metadata[3]) is not bool or metadata[1] != request.utility):
            raise ValueError('strong_tail_invalid_namespace_context')
        # tables() supplies only new-process identities and current definitions.
        namespace_request = replace(request, plan_type=metadata[0], utility=metadata[1],
            decision_kind=metadata[2], drink_slots=request.drink_slots if metadata[3] else ())
        if bool(namespace_request.drink_slots) != metadata[3]:
            # A completed bottle action may leave a historical nonempty namespace.
            # Construct its key explicitly; no placeholder bottle enters a request.
            namespace = (id(planner.kernel), type(planner.kernel), planner.kernel.ruleset_id,
                type(planner), planner.planner_version, planner.terminal_cache,
                planner.complete_horizon, *metadata,
                *(tuple(sorted(getattr(planner.kernel, name).items()))
                  for name in ('cards', 'drinks', 'items', 'memory_abilities')))
        else:
            prior_keys = set(session._namespaces)
            session.tables(planner, namespace_request)
            added = set(session._namespaces) - prior_keys
            if len(added) != 1:
                raise ValueError('strong_tail_duplicate_namespace')
            namespace = added.pop()
        _namespace_metadata(namespace, planner)
        if namespace in seen:
            raise ValueError('strong_tail_duplicate_namespace')
        seen.add(namespace)
        memo = decode(row['memo'])
        universes = decode(row['universes'])
        if not isinstance(memo, dict) or not isinstance(universes, tuple):
            raise ValueError('strong_tail_invalid_tables')
        marginal = TerminalMarginalCache(planner.kernel)
        ids = []
        for cards in universes:
            if not isinstance(cards, tuple) or len({c.instance_id for c in cards}) != len(cards):
                raise ValueError('strong_tail_invalid_universe')
            if cards == request.state.deck.all_cards and id(request.state.deck.all_cards) not in ids:
                cards = request.state.deck.all_cards
            identity = id(cards)
            if identity in ids:
                raise ValueError('strong_tail_duplicate_universe')
            ids.append(identity)
            marginal.universes[identity] = cards

        def table(value):
            result = {}
            data = decode(value)
            if not isinstance(data, dict):
                raise ValueError('strong_tail_invalid_marginal_table')
            for key, item in data.items():
                if (not isinstance(key, tuple) or not key or type(key[0]) is not int
                        or not 0 <= key[0] < len(ids)):
                    raise ValueError('strong_tail_invalid_universe_index')
                result[(ids[key[0]], *key[1:])] = item
            return result

        marginal.eligibility = table(row['eligibility'])
        marginal.values = table(row['values'])
        marginal.transitions = table(row['transitions'])
        if any(type(value) is not bool for value in marginal.eligibility.values()):
            raise ValueError('strong_tail_invalid_eligibility')
        for values in (*memo.values(), *marginal.values.values()):
            _outcomes(values)
        for name in ('hits', 'transition_hits'):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError('strong_tail_invalid_counter')
            setattr(marginal, name, row[name])
        session._namespaces[namespace] = memo, marginal
    if session.entry_count != raw['entries']:
        raise ValueError('strong_tail_entry_count_changed')
    session.decisions = session.memo_hits = session.namespaces_created = 0
    return session


def export_tail_link(owner):
    encode, _ = _codec()
    saved = owner._actual_tail_link
    if saved is None:
        return None
    return dict(version=VERSION,
        link=encode({key: replace(value, source_slot=None) if key == 'action' else value
                     for key, value in saved.items() if key not in ('session', 'context')}),
        context=encode(saved['context']), session=export_session(saved['session'], owner.tail))


def restore_tail_link(owner, raw, request, plan_count):
    _, decode = _codec(owner.kernel)
    if raw is None:
        return None
    if (not owner.actual_tail_session_reuse or set(raw) != {'version', 'link', 'context', 'session'}
            or raw['version'] != VERSION):
        raise ValueError('strong_tail_invalid_link')
    saved = decode(raw['link'])
    if (set(saved) != {'before', 'action', 'after', 'committed', 'source_plan_index'}
            or type(saved['committed']) is not bool or type(saved['source_plan_index']) is not int
            or not 0 <= saved['source_plan_index'] < plan_count):
        raise ValueError('strong_tail_invalid_link_fields')
    canonical = owner.kernel._canonical_probability_state
    expected = saved['after'] if saved['committed'] else saved['before']
    if (expected != canonical(request.state) or expected.turns_remaining != 2
            or saved['before'].turns_remaining != 2
            or saved['action'].source_slot is not None
            or saved['action'].kind not in (ActionKind.PLAY_CARD, ActionKind.USE_DRINK)
            or not saved['committed'] and saved['after'] is not None):
        raise ValueError('strong_tail_public_state_changed')
    for state in (saved['before'], expected):
        owner.kernel._validate_registered_state(state, plan_type=request.plan_type)
    if not saved['committed']:
        saved['action'] = RobustHighScorePlanner._attach_source_slot(saved['action'], request)
    context = owner._actual_tail_context(request)
    if decode(raw['context']) != context:
        raise ValueError('strong_tail_context_changed')
    return dict(saved, context=context,
                session=restore_session(raw['session'], owner.tail, request, owner.deadline))
