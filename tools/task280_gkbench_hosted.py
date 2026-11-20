"""Hosted diagnostic that continues from the referee's actual selection."""
from __future__ import annotations

import re
import json
import time
import argparse
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from collections import Counter
from dataclasses import replace

from tools.task280_gkbench_smoke import verify_final
from agent.live_decision.high_score.contracts import ActionKind, GameAction

SERVER = 'https://hiro.matto.top/gkbench'
BLACK_VINEGAR = 'pdrink_01-3-012'
LOCAL_BLACK_VINEGAR = '22'


def _card_key(card):
    return (card['id'], card['u'], tuple(sorted(tuple(row) for row in card.get('c', ()))),
            tuple(sorted(card.get('g', ()))))


def planning_upgrade_level(observed):
    if type(observed) is not int or observed < 0:
        raise ValueError('task280_invalid_observed_upgrade_level')
    return min(observed, 1)


def planning_card_key(card):
    return _card_key(card | {'u': planning_upgrade_level(card['u'])})


def _cards(observation, *zones):
    return Counter(_card_key(card) for zone in zones for card in observation[zone])


def delegated_retrieval(before, after, *, expected=None):
    """Compare a prior intention with a public, uniquely observable transfer."""
    evidence = {'expected': expected, 'actual': None, 'status': 'unverified'}
    if any(action[0] == 'select' for action in after['legal']):
        return evidence | {'status': 'explicit_selection'}
    source_before = _cards(before, 'deck', 'grave')
    source_after = _cards(after, 'deck', 'grave')
    hand_before, hand_after = _cards(before, 'hand'), _cards(after, 'hand')
    added = hand_after - hand_before
    if (added.total() != 1 or hand_before - hand_after
            or source_before - source_after != added or source_after - source_before):
        return evidence
    actual, = added
    status = 'unpriced' if expected is None else 'match' if actual == expected else 'mismatch'
    return evidence | {'actual': actual, 'status': status}


def reconcile_local_retrieval(kernel, request, drink, preferred, receipt, *, mappings):
    """Use the observed target, never the shadow preference, as simulation state."""
    from agent.live_decision.high_score.gkms_shadow_adapter import ShadowMappingError, _card

    if receipt['actual'] is None or receipt['status'] not in {'match', 'mismatch', 'unpriced'}:
        return None
    if (drink.kind is not ActionKind.USE_DRINK or drink.drink_id != LOCAL_BLACK_VINEGAR
            or (drink.source_slot, drink.drink_id) not in request.drink_slots):
        raise ValueError('gkbench_black_vinegar_binding_unverified')
    if preferred is not None and preferred.kind is not ActionKind.MOVE_CARD:
        raise ValueError('gkbench_shadow_retrieval_action_invalid')
    branches = kernel.transition(request.state, drink, plan_type=request.plan_type)
    if len(branches) != 1 or branches[0].probability != 1.0:
        return None
    pending_state = branches[0].state
    pending = pending_state.pending_choice
    if pending is None or pending.action_kind is not ActionKind.MOVE_CARD:
        return None
    candidates = {card.instance_id: card for card in pending_state.deck.draw_pile + pending_state.deck.discard
                  if card.instance_id in pending.candidate_instance_ids}
    try:
        raw_id, upgrade, customizations, growth = receipt['actual']
        planning_key = (raw_id, planning_upgrade_level(upgrade), customizations, growth)
        matches = [instance for instance, card in candidates.items() if _card_key(_card(card, mappings)) == planning_key]
    except ShadowMappingError:
        return None
    if len(matches) != 1:
        return None
    actual = GameAction(ActionKind.MOVE_CARD, card_instance_id=matches[0])
    transitions = kernel.transition(pending_state, actual, plan_type=request.plan_type)
    if len(transitions) != 1 or transitions[0].probability != 1.0 or transitions[0].state.pending_choice:
        return None
    next_state = transitions[0].state
    next_request = replace(request, state=next_state,
                           card_slots=tuple((card.instance_id, slot) for slot, card in enumerate(next_state.deck.hand)),
                           drink_slots=tuple(pair for pair in request.drink_slots
                                             if pair != (drink.source_slot, drink.drink_id)))
    next_request.validate()
    return next_request, actual, preferred is not None and preferred.card_instance_id != actual.card_instance_id


def play_diagnostic(start, post_step, *, expected_pick):
    """Protocol-only driver; a diagnostic expectation is not a policy result."""
    observation = start
    score = observation['num']['score']
    actions = Counter()
    retrieval = None
    for index in range(65):
        started = time.perf_counter()
        if observation['i'] != index or observation['num']['score'] != score:
            raise ValueError('gkbench_referee_observation_mismatch')
        legal = observation['legal']
        if index == 0:
            if observation['drinks'].count(BLACK_VINEGAR) != 1:
                raise ValueError('expected_one_black_vinegar')
            action = ['drink', observation['drinks'].index(BLACK_VINEGAR)]
            expected = expected_pick(observation)
        else:
            action = next((candidate for kind in ('play', 'end')
                           for candidate in legal if candidate[0] == kind), None)
        if action is None or action not in legal or any(candidate[0] == 'select' for candidate in legal):
            raise ValueError('gkbench_unhandled_or_unadvertised_action')
        reply = post_step(action, round((time.perf_counter() - started) * 1000, 2))
        actions[action[0]] += 1
        next_score = reply['result']['after']['score']
        if next_score - score != reply['result']['delta'].get('score', 0):
            raise ValueError('gkbench_score_delta_mismatch')
        if reply['done']:
            if retrieval is None:
                raise ValueError('gkbench_drink_receipt_missing')
            verify_final(reply['final'], score=next_score, decisions=index + 1)
            return {'scope': 'protocol_diagnostic_not_task275', 'final': reply['final'],
                    'actions': dict(actions), 'retrieval': retrieval}
        if index == 0:
            if Counter(observation['drinks']) - Counter(reply['obs']['drinks']) != Counter({BLACK_VINEGAR: 1}):
                raise ValueError('gkbench_drink_receipt_mismatch')
            retrieval = delegated_retrieval(observation, reply['obs'], expected=expected)
        observation, score = reply['obs'], next_score
    raise ValueError('gkbench_diagnostic_step_limit')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def hosted_request(token_path):
    token = token_path.read_text(encoding='utf-8-sig').strip()
    if not re.fullmatch(r'[A-Za-z0-9_.~-]+', token):
        raise ValueError('gkbench_plain_token_required')
    opener = urllib.request.build_opener(_NoRedirect)

    def request(path, body=None, *, method=None):
        headers = {'Authorization': 'Bearer ' + token}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        query = urllib.request.Request(SERVER + path, headers=headers, method=method,
                                       data=None if body is None else json.dumps(body).encode())
        with opener.open(query, timeout=10) as response:
            if response.headers.get('Content-Type', '').startswith('application/json'):
                return json.load(response)
            return None

    return request


def run_existing_slot(eval_id, problem, token_path):
    if not re.fullmatch(r'[a-f0-9]{12}', eval_id) or not re.fullmatch(r'AS-[A-Z0-9]{8}', problem):
        raise ValueError('gkbench_invalid_evaluation_or_problem')
    request = hosted_request(token_path)
    engine = request('/api/me')['engine']
    if engine['rules_version'] != 65:
        raise ValueError('gkbench_rule_version_changed')
    evaluation_path = f'/api/evals/{eval_id}'
    evaluation = request(evaluation_path)
    if evaluation['set'] != 'hif' or any(entry['id'] == problem and entry['slot'] == 0
                                         for entry in evaluation['done']):
        raise ValueError('gkbench_diagnostic_not_an_unfinished_hif_slot')
    slot = {'id': problem, 'slot': 0}
    start = request(evaluation_path + '/episodes', slot)
    episode_path = '/api/episodes/' + start['episode']
    try:
        report = play_diagnostic(
            start['obs'],
            lambda action, elapsed: request(episode_path + '/step', {
                'action': action, 'client_ms': elapsed, 'meta': {'scope': 'protocol-diagnostic-not-strategy'},
            }),
            expected_pick=lambda obs: min(_cards(obs, 'deck', 'grave'), default=None),
        )
    except (ValueError, OSError, KeyError):
        try:
            request(episode_path, method='DELETE')
        except OSError:
            pass
        raise
    saved = request(evaluation_path)
    if not any(entry['id'] == problem and entry['slot'] == 0 and entry['score'] == report['final']['score']
               for entry in saved['done']):
        raise ValueError('gkbench_terminal_missing_from_evaluation')
    run_path = '/api/runs/' + urllib.parse.quote(saved['run'], safe='')
    run = request(run_path)
    if run['engine']['rules_version'] != 65 or run['step_seconds'] != 5:
        raise ValueError('gkbench_diagnostic_rule_or_budget_mismatch')
    return report | {'rule_version': 65, 'eval': eval_id, 'problem': problem, 'slot': 0,
                     'expected_rule': 'lexicographic_public_source_card_protocol_probe_only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--problem', required=True)
    parser.add_argument('--token-file', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_existing_slot(args.eval, args.problem, args.token_file)
    except (ValueError, OSError, KeyError) as failure:
        parser.exit(1, f'Hosted diagnostic stopped: {type(failure).__name__} {getattr(failure, "code", "")}\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
