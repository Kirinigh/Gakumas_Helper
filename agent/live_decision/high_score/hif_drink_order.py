"""Compare one alternative card-first policy with a winning immediate drink."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

from .kernel import EffectKind
from .hif_mcts import action_key, _RootOption
from .contracts import ActionKind


def _setup(kernel, request, legal):
    # Compare the order of one draw that also builds concentration for later plays.
    # Ambiguous set-ups retain the original searched policy.
    held = {card.instance_id: card for card in request.state.deck.hand}
    candidates = []
    for action in legal:
        if action.kind is ActionKind.PLAY_CARD:
            definition = kernel.card_definition(request.state, held[action.card_instance_id])
            kinds = {effect.kind for effect in definition.effects}
            if {EffectKind.DRAW, EffectKind.CONCENTRATION_EFFECT_BUFF} <= kinds:
                candidates.append(action)
    return candidates[0] if len(candidates) == 1 else None


def _best_card_before_score_drink(search, request, legal, estimate):
    """Compare against the best already priced card with a free continuation."""
    definition = search.kernel.drinks.get(estimate.action.drink_id)
    if definition is None or not any(
        effect.kind is EffectKind.SCORE_BUFF and effect.amount > 0 and effect.condition is None
        and effect.duration is not None and 0 < effect.duration < request.state.turns_remaining
        for effect in definition.effects
    ):
        return None
    candidates = (row for row in estimate.candidate_statistics
                  if row["action"].kind is ActionKind.PLAY_CARD and row["action"] in legal
                  and not row.get("continuation") and row.get("validation_mean_final_score") is not None)
    best = max(candidates, key=lambda row: row["validation_mean_final_score"], default=None)
    return best["action"] if best else None


def compare_drink_order(search, request, legal, original, deadline):
    """Use only residual traversals, with new paired seeds and the same terminal evaluator."""
    action = original.action
    if (
        action is None
        or action.kind is not ActionKind.USE_DRINK
        or not original.complete_validation_coverage
        or (original.retained_policy_comparison or {}).get("selected")
    ):
        return original, None
    available = min(original.unused_simulation_budget, search.simulations - original.started_simulations)
    pairs = min(16, available // 2)
    if pairs < search.validation_samples:
        return original, None
    target = None
    try:
        if original.selected_continuation:
            if len(original.selected_continuation) != 1 or request.state.turns_remaining <= 2:
                return original, None
            (target,) = original.selected_continuation
            if target.kind is not ActionKind.MOVE_CARD:
                return original, None
            setup = _setup(search.kernel, request, legal)
        else:
            setup = _best_card_before_score_drink(search, request, legal, original)
    except (KeyError, ValueError):
        return original, None
    if setup is None:
        return original, None
    guard = original.uniform_validation_time_guard or {}
    started = time.perf_counter()
    deadline = min(search._order_deadline, deadline)
    remaining = deadline - started
    reserve = guard.get("delivery_reserve_seconds", 0.0)
    if remaining <= reserve + guard.get("max_completed_round_seconds", 0.0):
        return original, None
    # Give back the completed policy with the same observed-round and delivery
    # reserve used by the existing search.
    deadline -= reserve + guard.get("max_completed_round_seconds", 0.0)
    cohort = search.last_validation_cohort_evidence
    retained = original.retained_policy_comparison or {}
    excluded = (
        *search._training_seed_history,
        *cohort["validation_seeds"],
        *cohort["training_seeds"],
        *retained.get("excluded_training_seeds", ()),
        *retained.get("excluded_validation_seeds", ()),
    )
    seeds = search._fresh_seeds(pairs, excluded)
    incumbent = _RootOption(action, original.selected_continuation, original.selected_expected_after_first)
    report = dict(
        original=asdict(action),
        setup=asdict(setup),
        target=asdict(target) if target else None,
        card_first_continuation="free_frozen_policy",
        original_selection_changed_from_retained=original.selection_changed_from_retained,
        pairs=[],
        started_seeds=[],
        requested_pairs=pairs,
        started_simulations=0,
        completed_simulations=0,
        selected=False,
    )
    previous_deadline = search._order_deadline
    max_depth, terminal, fallbacks = original.maximum_depth, 0, 0
    try:
        search._order_deadline = min(deadline, previous_deadline)
        for seed in seeds:
            row = dict(seed=seed)
            for label in ("immediate", "after_setup"):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("mcts_window_timeout")
                if label == "immediate":
                    report["started_seeds"].append(seed)
                report["started_simulations"] += 1
                root = incumbent if label == "immediate" else action_key(setup)
                score, _, depth, calls, failed = search._validate_simulation(request, seed, root, deadline)
                report["completed_simulations"] += 1
                if not math.isfinite(score):
                    raise ValueError("incomplete drink-order terminal return")
                max_depth = max(max_depth, depth)
                terminal += calls
                fallbacks += failed
                row[label] = score
            row["difference"] = row["after_setup"] - row["immediate"]
            report["pairs"].append(row)
        report["mean_difference"] = sum(row["difference"] for row in report["pairs"]) / pairs
        report["selected"] = report["mean_difference"] > 0
    except (KeyError, ValueError, RuntimeError, TimeoutError) as error:
        report["reason"] = str(error)
    finally:
        search._order_deadline = previous_deadline
    used = report["completed_simulations"]
    began = report["started_simulations"]
    result = replace(
        original,
        elapsed_ms=original.elapsed_ms + (time.perf_counter() - started) * 1000,
        maximum_depth=max_depth,
        terminal_evaluations=original.terminal_evaluations + terminal,
        terminal_fallbacks=original.terminal_fallbacks + fallbacks,
        started_simulations=original.started_simulations + began,
        completed_simulations=original.completed_simulations + used,
        unused_simulation_budget=original.unused_simulation_budget - began,
        validation_started_simulations=original.validation_started_simulations + began,
        validation_completed_simulations=original.validation_completed_simulations + used,
    )
    if report["selected"]:
        # The challenger commits only its first card. Later decisions remain free.
        training_best = max(original.candidate_statistics, key=lambda row: row["mean_final_score"])
        result = replace(
            result, action=setup, selected_continuation=(), selected_expected_after_first=None,
            root_selection_scope="paired_drink_order",
            selection_changed_from_retained=training_best["action"] != setup or bool(training_best.get("continuation")),
        )
    return result, report
