"""Allocate residual validation returns using paired estimated regret.

All candidates are complete policies, identified by their original index. The
four common base worlds establish every pair; a later world updates only its
two allocated policies. The exploration score is an allocation heuristic, not
a confidence bound or a claim of optimality. A new minimax winner replaces the
retained policy only when their direct paired mean difference is positive.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy

VERSION = 'hif-paired-adaptive-validation-v3'
VALUE_SCOPE = 'PAIRED_COMPLETE_POLICY_ESTIMATED_REGRET'
DECISION_RULE = 'MINIMAX_PAIRED_MEAN_REGRET_WITH_DIRECT_RETAINED_GUARD'
SAMPLE_CAP = 32
BASE_SAMPLES = 4
DELIVERY_RESERVE_SECONDS = 1.
_RETURN_KINDS = frozenset(('sampled_terminal_score', 'terminal_evaluator_expected_score',
                           'proven_tail_expected_score'))


def _seed(value):
    return type(value) is int and 0 <= value < 2**31


def _guard(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('paired_adaptive_validation_deadline')


def _sample(row, seed):
    if not isinstance(row, dict):
        raise ValueError('paired_adaptive_validation_invalid_sample')
    score = row.get('final_score')
    depth, calls, fallbacks = (row.get(name) for name in
                              ('completed_depth', 'terminal_evaluations', 'terminal_fallbacks'))
    elapsed = row.get('elapsed_seconds')
    if (type(score) not in (int, float) or not math.isfinite(score)
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
            or row.get('return_kind') not in _RETURN_KINDS
            or any(type(value) is not int for value in (depth, calls, fallbacks))
            or not 0 <= depth < 256 or calls < 0 or fallbacks != 0
            or not _seed(row.get('planning_seed', seed)) or row.get('planning_seed', seed) != seed
            or 'complete' in row and row['complete'] is not True):
        raise ValueError('paired_adaptive_validation_invalid_sample')
    return deepcopy(dict(row, planning_seed=seed))


def evaluate(candidate_keys, base_samples, *, remaining_budget, reward_scale,
             next_seed, evaluate_return, frozen_signature, deadline, excluded_seeds=(),
             retained_index=None):
    """Return complete evidence, including charged work when a pair fails.

    ``base_samples[index]`` contains exactly four complete-return dictionaries
    with ``planning_seed`` in the same order for every policy. ``next_seed``
    receives the sorted union of caller exclusions, base seeds and new seeds.
    ``evaluate_return(index, seed, deadline)`` returns a complete-return dict;
    a normal callback return counts as completed even if its receipt is invalid.
    Only validated receipts increase candidate sample counts, and only complete
    pairs update pair statistics. No extra sample is appended to the base table.
    Base receipts include measured ``elapsed_seconds``. Before allocating a
    new seed, reserve twice the largest observed callback duration plus one
    second for delivery. This empirical estimate is not a timing guarantee;
    a started pair still fails on a real deadline or incomplete return.

    The caller supplies a signature of frozen continuation state. It is checked
    before and after the whole stage, including failure. Callback exceptions,
    invalid inputs, timeout, or a changed signature return ``complete=False``
    and never publish a winner. This function does not restore caller state.

    ``retained_index`` explicitly identifies the unique ``RETAINED_POLICY``
    key, when present. The final guard uses only this policy's existing shared
    worlds with the original minimax winner; it never changes allocation.
    """
    report = dict(version=VERSION, enabled=True, value_scope=VALUE_SCOPE,
        decision_rule=DECISION_RULE, reward_scale=reward_scale, sample_cap=SAMPLE_CAP,
        complete=False, status='failed', reason=None, errors=[], candidate_keys=(),
        base_samples=(), base_seeds=(), excluded_seeds=(), additional_pairs=[],
        requested_simulations=remaining_budget, started_simulations=0,
        completed_simulations=0, remaining_budget=remaining_budget,
        candidate_sample_counts=[], pair_statistics=[], paired_regrets=[],
        selected_index=None, frozen_signature_unchanged=None,
        minimax_selected_index=None, retained_index=None, retained_pair_count=0,
        retained_mean_difference=None, retained_guard_reason=None,
        time_guard=dict(rule='REMAINING_GT_TWO_MAX_OBSERVED_RETURNS_PLUS_DELIVERY',
            delivery_reserve_seconds=DELIVERY_RESERVE_SECONDS,
            base_observed_returns=0, additional_observed_returns=0,
            maximum_observed_return_seconds=0., last_check=None,
            stopped_before_next_pair=False))
    differences = {}
    before = None
    signature_started = False
    ready = False

    def means():
        return {pair: sum(values) / len(values) for pair, values in differences.items()}

    def difference(pair_means, left, right):
        return pair_means[left, right] if left < right else -pair_means[right, left]

    def regrets(pair_means):
        count = len(report['candidate_keys'])
        return [max((difference(pair_means, index, other) for other in range(count)
                     if other != index), default=0.) for index in range(count)]

    def fail(error):
        message = str(error) or type(error).__name__
        report['errors'].append(dict(type=type(error).__name__, reason=message))
        report.update(complete=False, status='failed', reason=message, selected_index=None,
            minimax_selected_index=None, retained_index=None, retained_pair_count=0,
            retained_mean_difference=None, retained_guard_reason=None)

    try:
        if (type(remaining_budget) is not int or remaining_budget < 0
                or type(reward_scale) not in (int, float) or not math.isfinite(reward_scale)
                or reward_scale <= 0 or type(deadline) not in (int, float)
                or not math.isfinite(deadline)
                or any(not callable(callback) for callback in
                       (next_seed, evaluate_return, frozen_signature))):
            raise ValueError('paired_adaptive_validation_invalid_configuration')
        _guard(deadline)
        keys = tuple(deepcopy(candidate_keys))
        rows = tuple(tuple(row) for row in base_samples)
        excluded = tuple(excluded_seeds)
        if (not keys or len(rows) != len(keys)
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in excluded)):
            raise ValueError('paired_adaptive_validation_invalid_candidates_or_exclusions')
        retained_keys = tuple(index for index, key in enumerate(keys)
            if isinstance(key, (tuple, list)) and key and key[0] == 'RETAINED_POLICY')
        if retained_index is None:
            if retained_keys:
                raise ValueError('paired_adaptive_validation_invalid_retained_index')
        elif (type(retained_index) is not int or not 0 <= retained_index < len(keys)
                or retained_keys != (retained_index,) or len(keys[retained_index]) != 2):
            raise ValueError('paired_adaptive_validation_invalid_retained_index')
        if any(len(row) != BASE_SAMPLES for row in rows):
            raise ValueError('paired_adaptive_validation_four_base_samples_required')
        seeds = tuple(row.get('planning_seed') if isinstance(row, dict) else None for row in rows[0])
        if any(not _seed(seed) for seed in seeds) or len(set(seeds)) != BASE_SAMPLES:
            raise ValueError('paired_adaptive_validation_invalid_base_seeds')
        base = []
        for row in rows:
            if tuple(item.get('planning_seed') if isinstance(item, dict) else None
                     for item in row) != seeds:
                raise ValueError('paired_adaptive_validation_base_seeds_not_paired')
            base.append(tuple(_sample(item, seed) for item, seed in zip(row, seeds)))
        report.update(candidate_keys=keys, base_samples=tuple(base), base_seeds=seeds,
                      excluded_seeds=excluded, candidate_sample_counts=[BASE_SAMPLES] * len(keys),
                      retained_index=retained_index)
        timing = report['time_guard']
        timing.update(base_observed_returns=sum(map(len, base)),
            maximum_observed_return_seconds=max(item['elapsed_seconds'] for row in base for item in row))
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                values = [base[right][world]['final_score'] - base[left][world]['final_score']
                          for world in range(BASE_SAMPLES)]
                if not all(math.isfinite(value) for value in (*values, sum(values))):
                    raise ValueError('paired_adaptive_validation_nonfinite_difference')
                differences[left, right] = values
        before = deepcopy(frozen_signature())
        signature_started = ready = True
        used_seeds = set(excluded).union(seeds)
        while True:
            _guard(deadline)
            counts = report['candidate_sample_counts']
            eligible = [index for index, count in enumerate(counts) if count < SAMPLE_CAP]
            if len(eligible) < 2:
                report['reason'] = 'fewer_than_two_candidates_below_cap'
                break
            if report['remaining_budget'] < 2:
                report['reason'] = 'remaining_budget_below_pair'
                break
            pair_means = means()
            current_regrets = regrets(pair_means)
            leader = min(eligible, key=lambda index: (current_regrets[index], index))
            opponents = []
            for other in eligible:
                if other == leader:
                    continue
                count = len(differences[min(leader, other), max(leader, other)])
                estimate = difference(pair_means, leader, other)
                exploration = reward_scale * math.sqrt(2 * math.log(sum(counts) + 1) / count)
                score = estimate + exploration
                if not math.isfinite(score):
                    raise ValueError('paired_adaptive_validation_nonfinite_allocation')
                opponents.append(dict(candidate_index=other, pair_count=count,
                    paired_mean_difference=estimate, exploration=exploration, score=score))
            opponent = max(opponents, key=lambda row: (row['score'], -row['candidate_index']))['candidate_index']
            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                raise TimeoutError('paired_adaptive_validation_deadline')
            maximum_seconds = timing['maximum_observed_return_seconds']
            check = dict(remaining_seconds=remaining_seconds,
                maximum_observed_return_seconds=maximum_seconds,
                estimated_pair_seconds=2 * maximum_seconds,
                required_seconds=2 * maximum_seconds + DELIVERY_RESERVE_SECONDS)
            timing['last_check'] = check
            if remaining_seconds <= check['required_seconds']:
                timing['stopped_before_next_pair'] = True
                report['reason'] = 'time_reserve_reached'
                break
            pair = dict(leader_index=leader, opponent_index=opponent,
                candidate_keys=(deepcopy(keys[leader]), deepcopy(keys[opponent])),
                counts_before=tuple(counts), paired_regrets_before=tuple(current_regrets),
                opponent_scores=opponents, seed=None, samples=[], complete=False,
                time_guard=dict(check), return_timings=[])
            report['additional_pairs'].append(pair)
            seed = next_seed(tuple(sorted(used_seeds)))
            if not _seed(seed) or seed in used_seeds:
                raise ValueError('paired_adaptive_validation_seed_conflict_or_invalid')
            pair['seed'] = seed
            used_seeds.add(seed)
            for index in (leader, opponent):
                _guard(deadline)
                report['started_simulations'] += 1
                report['remaining_budget'] -= 1
                started = time.perf_counter()
                returned = False
                try:
                    raw = evaluate_return(index, seed, deadline)
                    returned = True
                finally:
                    elapsed = time.perf_counter() - started
                    timing['additional_observed_returns'] += 1
                    timing['maximum_observed_return_seconds'] = max(
                        timing['maximum_observed_return_seconds'], elapsed)
                    pair['return_timings'].append(dict(candidate_index=index,
                        elapsed_seconds=elapsed, callback_returned=returned))
                report['completed_simulations'] += 1
                if isinstance(raw, dict):
                    raw = dict(raw, elapsed_seconds=elapsed)
                row = _sample(raw, seed)
                row['candidate_index'] = index
                pair['samples'].append(row)
                counts[index] += 1
                _guard(deadline)
            first, second = pair['samples']
            delta = second['final_score'] - first['final_score']
            values = differences[min(leader, opponent), max(leader, opponent)]
            delta = delta if leader < opponent else -delta
            if not math.isfinite(delta) or not math.isfinite(sum(values) + delta):
                raise ValueError('paired_adaptive_validation_nonfinite_difference')
            values.append(delta)
            pair['complete'] = True
        report.update(complete=True, status='complete')
    except Exception as error:
        fail(error)
    finally:
        if signature_started:
            try:
                unchanged = frozen_signature() == before
                report['frozen_signature_unchanged'] = unchanged
                if not unchanged:
                    raise ValueError('paired_adaptive_validation_frozen_policy_changed')
            except Exception as error:
                fail(error)
        if ready:
            pair_means = means()
            report['pair_statistics'] = [dict(lower_index=left, upper_index=right,
                count=len(values), difference_sum=sum(values), mean_difference=pair_means[left, right])
                for (left, right), values in differences.items()]
            report['paired_regrets'] = regrets(pair_means)
            if report['complete']:
                try:
                    _guard(deadline)
                    winner = min(range(len(keys)),
                        key=lambda index: (report['paired_regrets'][index], index))
                    report['minimax_selected_index'] = winner
                    if retained_index is None:
                        report['retained_guard_reason'] = 'no_retained_candidate'
                    elif winner == retained_index:
                        report['retained_guard_reason'] = 'minimax_selected_retained'
                    else:
                        direct = -difference(pair_means, winner, retained_index)
                        report['retained_pair_count'] = len(differences[
                            min(winner, retained_index), max(winner, retained_index)])
                        report['retained_mean_difference'] = direct
                        if direct > 0:
                            report['retained_guard_reason'] = 'direct_paired_mean_positive'
                        else:
                            report['retained_guard_reason'] = 'direct_paired_mean_nonpositive'
                            winner = retained_index
                    report['selected_index'] = winner
                except Exception as error:
                    fail(error)
    return report
