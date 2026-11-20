"""Exercise gkbench action binding and replay with an original offline fixture."""

from __future__ import annotations

import json
import argparse
from collections import Counter
from dataclasses import asdict, replace

from agent.live_decision.high_score.contracts import ActionKind, GameAction


def action_bindings(request, legal, *, ordered_instances, ordered_bottle_slots):
    local_instances = dict(request.card_slots)
    local_bottles = dict(request.drink_slots)
    if (len(ordered_instances) != len(local_instances) or set(ordered_instances) != set(local_instances)
            or len(ordered_bottle_slots) != len(local_bottles) or set(ordered_bottle_slots) != set(local_bottles)):
        raise ValueError('gkbench_positions_not_uniquely_bound')
    pending = request.state.pending_choice
    if pending and (pending.action_kind is not ActionKind.MOVE_CARD
                    or pending.selections_remaining != 1 or pending.selection_optional):
        raise ValueError('gkbench_selection_shape_unverified')

    bindings = {}
    for action in legal:
        if action.kind is ActionKind.USE_DRINK and not pending:
            for index, slot in enumerate(ordered_bottle_slots):
                if local_bottles[slot] == action.drink_id:
                    bindings[('drink', index)] = replace(action, source_slot=slot)
        elif action.kind is ActionKind.END_TURN and not pending:
            bindings[('end',)] = action
        elif action.kind is ActionKind.PLAY_CARD and not pending:
            bindings[('play', ordered_instances.index(action.card_instance_id))] = action
        elif action.kind is ActionKind.MOVE_CARD and pending:
            bindings[('select', (ordered_instances.index(action.card_instance_id),))] = action
        else:
            raise ValueError('gkbench_action_kind_unverified')
    return bindings


def encode_action(action, bindings, server_legal):
    matches = [token for token, candidate in bindings.items() if candidate == action]
    if len(matches) != 1:
        raise ValueError('gkbench_action_not_uniquely_bound')
    token = matches[0]
    wire = [token[0], list(token[1])] if token[0] == 'select' else list(token)
    if wire not in server_legal:
        raise ValueError('gkbench_action_not_advertised_by_referee')
    return wire


def verify_final(final, *, score, decisions):
    if final.get('finished') is not True:
        raise ValueError('gkbench_referee_did_not_finish')
    if type(final.get('score')) is not int or final['score'] != score:
        raise ValueError('gkbench_referee_terminal_score_mismatch')
    if type(final.get('decisions')) is not int or final['decisions'] != decisions:
        raise ValueError('gkbench_referee_decision_count_mismatch')


def _normalized(value):
    return json.loads(json.dumps(value))


def replay_archive(environment, request, archive):
    counts = Counter()
    rows = archive['actions']
    for row in rows:
        if _normalized(asdict(request.state)) != row['before']:
            raise ValueError('task275_replay_before_mismatch')
        action = GameAction(**(row['action'] | {'kind': ActionKind(row['action']['kind'])}))
        instances = tuple(instance for instance, _ in sorted(request.card_slots, key=lambda pair: pair[1]))
        bottles = tuple(sorted(slot for slot, _ in request.drink_slots))
        bindings = action_bindings(request, environment.legal_actions(),
                                   ordered_instances=instances, ordered_bottle_slots=bottles)
        local_wire_legal = [[head, list(indices[0])] if head == 'select' else [head, *indices]
                            for head, *indices in bindings]
        token = encode_action(action, bindings, local_wire_legal)
        key = ('select', tuple(token[1])) if token[0] == 'select' else tuple(token)
        step = environment.step(bindings[key])
        request = step.request
        if (_normalized(asdict(request.state)) != row['after']
                or _normalized([asdict(event) for event in step.events]) != row['events']):
            raise ValueError('task275_replay_transition_mismatch')
        counts[token[0]] += 1

    if (archive['terminal'] is not True or archive['error'] is not None
            or archive['result']['stopped'] is not False or archive['result']['illegal_action'] is not False
            or archive['result']['final_score'] != archive['observed_score']):
        raise ValueError('task275_archive_not_a_completed_episode')
    verify_final({'score': request.state.score, 'finished': request.state.terminal, 'decisions': sum(counts.values())},
                 score=archive['observed_score'], decisions=archive['result']['decision_count'])
    return {'scope': 'local_archive_replay_only', 'score': request.state.score,
            'actions': dict(counts), 'hosted_referee_verified': False}


def synthetic_smoke(seed):
    # Import the rule engine only for an actual replay. The wire adapters work
    # with a caller's observed actions and do not require a simulator.
    import time

    from tools.task285_cases import load_case
    from tools.task285_seed_bench import RoutePolicy, SyntheticPrior
    from tools.task275.gkms_continuous_policy import execute_episode
    from agent.live_decision.high_score.hif_environment import HifEnvironment

    case = load_case('synthetic-draw')
    environment = HifEnvironment(case.kernel)
    request = environment.reset(case.state, seed=seed, drink_slots=case.drink_slots)
    trace = execute_episode(environment, RoutePolicy(SyntheticPrior(), case.routes[0]), request,
                            episode_id='task280-synthetic', deadline=time.perf_counter() + 5)
    # Match the JSON round trip the historical replay accepted.
    request = environment.reset(case.state, seed=seed, drink_slots=case.drink_slots)
    return replay_archive(environment, request, _normalized(trace)) | {'scope': 'synthetic_offline_replay'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-seed', default=280, type=int)
    args = parser.parse_args()
    print(json.dumps(synthetic_smoke(args.local_seed), sort_keys=True))


if __name__ == '__main__':
    main()
