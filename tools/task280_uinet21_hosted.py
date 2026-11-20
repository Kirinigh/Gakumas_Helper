"""Run the pinned UINet network on declared HIF Sense template slots."""

from __future__ import annotations

import re
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from collections import Counter

from tools.task280_gkbench_smoke import verify_final
from tools.task280_gkbench_hosted import BLACK_VINEGAR, hosted_request, delegated_retrieval
from agent.live_decision.high_score.gkms_continuous_policy import verify_model_output

REV = '1fc50378753bd35fb239bebb77cb315c019915ff'
MODEL_SHA256 = '5895807a112ca6eec82bbe2dd252a27b140add528e1f33db841eef7c92057824'
PROBLEMS = ('AS-4ND5SSJ7', 'AS-B38CARVV', 'AS-ERXY3JPJ')
TRACK = 'UINet-2.1-fixed-model-Sense-v65-referee-selects'


def load_policy(model_root, master):
    root = model_root.resolve(strict=True)
    model = root / 'model/gkms_uinet.onnx'
    with model.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != MODEL_SHA256:
            raise ValueError('unexpected_uinet21_weight')
    sys.path.insert(0, str(root))
    import gkms_uinet

    if not Path(gkms_uinet.__file__).resolve().is_relative_to(root):
        raise ValueError('unexpected_uinet21_module')
    policy = gkms_uinet.Policy(model, master.resolve(strict=True), threads=4)
    if policy.version != 2:
        raise ValueError('uinet21_uiobs_v2_required')
    return policy


def play_slot(start, post, policy, *, clock=time.perf_counter):
    obs, score = start, start['num']['score']
    latencies, delegated = [], []
    action_counts = Counter()
    for index in range(200):
        if (obs['v'] != 'uiobs.v2' or obs['exam']['plan'] != 2 or obs['i'] != index
                or obs['num']['score'] != score or not obs['legal']
                or any(token[0] not in {'play', 'drink', 'end'} for token in obs['legal'])):
            raise ValueError('unsupported_or_stale_sense_observation')
        started = clock()
        action = verify_model_output(policy.decide(obs), obs)
        latency = (clock() - started) * 1000
        latencies.append(latency)
        action_counts[action[0]] += 1
        response = post(action, round(latency, 2))
        following_score = response['result']['after']['score']
        if following_score - score != response['result']['delta'].get('score', 0):
            raise ValueError('referee_score_delta_mismatch')
        if response['done']:
            final = response['final']
            verify_final(final, score=following_score, decisions=index + 1)
            return dict(score=following_score, decisions=index+1, actions=dict(action_counts),
                        over_5s=sum(value > 5000 for value in latencies),
                        server_over=final['over_steps'], max_policy_ms=round(max(latencies), 2),
                        delegated=delegated)
        successor = response['obs']
        if action[0] == 'drink' and obs['drinks'][action[1]] == BLACK_VINEGAR:
            delegated.append(dict(step=index+1, **delegated_retrieval(obs, successor)))
        obs, score = successor, following_score
    raise ValueError('sense_step_limit')


def run(request, policy, *, problems, seeds, output, resume=None):
    engine = request('/api/me')['engine']
    if engine['rules_version'] != 65:
        raise ValueError('hosted_rule_version_not_65')
    params = {'model_revision': REV, 'threads': 4, 'scope': 'task280_sense_subset'}
    planned = {(problem, slot) for problem in problems for slot in range(seeds)}
    if resume is None:
        evaluation = request('/api/evals', {
            'set': 'hif', 'label': TRACK, 'algo': 'UINet-Ver2.1-Kummertranen-local-frozen',
            'params': params, 'seeds': seeds, 'only': list(problems), 'step_seconds': 5,
        })
        slots = evaluation['slots']
        if len(slots) != len(planned) or {(row['id'], row['slot']) for row in slots} != planned:
            raise ValueError('unexpected_evaluation_scope')
        already_done = set()
    else:
        evaluation = request('/api/evals/' + resume)
        manifest = request('/api/runs/' + evaluation['run'])
        if (evaluation['eval'] != resume or evaluation['label'] != TRACK or evaluation['set'] != 'hif'
                or evaluation['seeds_per_problem'] != seeds or evaluation['total'] != len(planned)
                or manifest['params'] != params or manifest['step_seconds'] != 5
                or manifest['engine']['rules_version'] != 65):
            raise ValueError('incompatible_uinet21_resume')
        already_done = {(row['id'], row['slot']) for row in evaluation['done']}
        if not already_done <= planned:
            raise ValueError('unexpected_completed_scope')
        slots = [dict(id=problem, slot=slot) for problem, slot in sorted(planned - already_done)]
    eval_id = evaluation['eval']
    eval_path = '/api/evals/' + eval_id
    print(json.dumps(dict(evaluation=eval_id, planned=len(planned))), flush=True)
    completed = len(already_done)
    with output.open('x', encoding='utf-8') as sink:
        for slot in slots:
            active = None
            try:
                start = request(eval_path + '/episodes', slot)
                active = '/api/episodes/' + start['episode']
                result = play_slot(
                    start['obs'], lambda action, latency: request(
                        active + '/step', {'action': action, 'client_ms': latency}), policy)
                active = None
                completed += 1
                result['finished'] = True
            except (ValueError, OSError, KeyError, RuntimeError, IndexError) as failure:
                result = dict(finished=False, failure=type(failure).__name__, http=getattr(failure, 'code', None))
                if isinstance(failure, ValueError) and str(failure).isascii():
                    result['reason'] = str(failure)[:180]
            finally:
                if active:
                    try:
                        request(active, method='DELETE')
                    except OSError:
                        pass
            row = dict(evaluation=eval_id, rule=65, **slot, **result)
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            print(json.dumps(row), flush=True)
    saved = request(eval_path)
    print(json.dumps(dict(evaluation=eval_id, done=saved['done'], completed=completed)), flush=True)
    actual = {(row['id'], row['slot']) for row in saved['done']}
    return completed == len(planned) and actual == planned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', nargs='+', choices=PROBLEMS, required=True)
    parser.add_argument('--seeds', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--token-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path,
                        help=f'Local UINet checkout at {REV} with its separately obtained model')
    parser.add_argument('--master', required=True, type=Path, help='Local master-data directory')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.output.exists() or len(set(args.problem)) != len(args.problem):
        parser.error('new output and unique problems are required')
    if args.resume is not None and not re.fullmatch(r'[a-f0-9]{12}', args.resume):
        parser.error('invalid evaluation identifier')
    policy = load_policy(args.model_root, args.master)
    return 0 if run(hosted_request(args.token_file), policy, problems=args.problem,
                    seeds=args.seeds, output=args.output, resume=args.resume) else 2


if __name__ == '__main__':
    raise SystemExit(main())
