from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.live_decision.high_score import (  # noqa: E402
    EFFECT_IR_VERSION,
    CardRef,
    PlanType,
    ActionKind,
    DeckBelief,
    EffectKind,
    GameAction,
    LogicRoute,
    DecisionKind,
    DeckKnowledge,
    PlannerConfig,
    StrategyState,
    EffectSourceKind,
    EffectValueSource,
    PythonNiaProKernel,
    ImmediateScorePolicy,
    HighScoreContractError,
    PhaseAwareHighScorePolicy,
    NiaProDynamicDeckGenerator,
    NiaProItemLifecycleGenerator,
    evaluate_policy,
    classify_logic_route,
    compile_item_catalog,
    compare_paired_results,
    summarize_policy_results,
    compile_skill_card_catalog,
    simulate_episode_with_trace,
    build_fixed_item_inventory_scenarios,
)

SKILL_CARDS = ROOT / "assets/data/skill_cards.csv"
P_ITEMS = ROOT / "assets/data/p_items.csv"


def _effect_tree(definition):
    pending = list(definition.effects + definition.persistent_effects)
    while pending:
        effect = pending.pop()
        yield effect
        pending.extend(effect.scheduled_effects)


def _definition_logic_route(card_id: str, cards) -> LogicRoute:
    card = CardRef(f"route-{card_id}", card_id)
    state = StrategyState(
        turn=1,
        turns_remaining=1,
        score=0,
        stamina=30,
        max_stamina=30,
        genki=0,
        actions_remaining=1,
        hand_limit=1,
        score_multiplier=1.0,
        deck=DeckBelief(
            all_cards=(card,),
            hand=(card,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        ),
    )
    return classify_logic_route(state, cards)


def _has_logic_signal(definition) -> bool:
    return any(
        effect.kind
        in {
            EffectKind.GOOD_IMPRESSION,
            EffectKind.MOTIVATION,
            EffectKind.GENKI,
        }
        or effect.value_source
        in {
            EffectValueSource.GOOD_IMPRESSION,
            EffectValueSource.MOTIVATION,
            EffectValueSource.GENKI,
        }
        for effect in _effect_tree(definition)
    )


def _evaluation_pool(plan_type: PlanType, logic_route: LogicRoute | None):
    catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=plan_type)
    primary_ids = tuple(
        sorted(
            (card_id for card_id in catalog.supported_card_ids if (catalog.cards[card_id].base_card_id or card_id) == card_id),
            key=int,
        )
    )
    if logic_route is None:
        acquisition_pool = primary_ids
    else:
        if plan_type is not PlanType.LOGIC:
            raise ValueError("logic_route is only valid for the Logic plan")
        acquisition_pool = tuple(
            card_id
            for card_id in primary_ids
            if _has_logic_signal(catalog.cards[card_id]) and _definition_logic_route(card_id, catalog.cards) is logic_route
        )
    pair_bases = tuple(
        card_id for card_id in acquisition_pool if card_id in catalog.upgrade_map and catalog.upgrade_map[card_id] in catalog.cards
    )
    if len(pair_bases) < 7:
        raise RuntimeError(f"{plan_type.value}/{logic_route} does not contain seven complete upgrade pairs")
    return catalog, pair_bases[:7], acquisition_pool


def _finite_item_limit(items: dict[str, object]) -> int:
    limits = [
        effect.schedule_limit for definition in items.values() for effect in definition.persistent_effects if effect.schedule_limit is not None
    ]
    return max(limits, default=1)


def _deterministic_metrics(metrics) -> dict[str, object]:
    values = asdict(metrics)
    values.pop("latency_p50_ms")
    values.pop("latency_p95_ms")
    return values


def _prepare_stage_entry_samples(samples, kernel, items, *, plan_type: PlanType):
    prepared = []
    inventory_ledger_item_ids: set[str] = set()
    scheduled_effect_item_ids: set[str] = set()
    initial_item_schedule_ref_count = 0
    for sample in samples:
        state = kernel.initialize_stage_entry(sample.request.state, plan_type=plan_type)
        actual = {
            (scheduled.card_id, scheduled.definition_effect_index)
            for scheduled in state.scheduled_effects
            if scheduled.source_kind is EffectSourceKind.ITEM
        }
        expected = {
            (item_id, effect_index) for item_id in state.items for effect_index, _effect in enumerate(items[item_id].persistent_effects)
        }
        if actual != expected:
            raise RuntimeError(
                f"{sample.episode_id} item schedule materialization mismatch: expected={sorted(expected)} actual={sorted(actual)}"
            )
        inventory_ledger_item_ids.update(state.items)
        scheduled_effect_item_ids.update(item_id for item_id, _index in actual)
        initial_item_schedule_ref_count += len(actual)
        prepared.append(replace(sample, request=replace(sample.request, state=state)))
    return (
        tuple(prepared),
        inventory_ledger_item_ids,
        scheduled_effect_item_ids,
        initial_item_schedule_ref_count,
    )


def _directed_mental_counter_probe(kernel, item_id: str) -> dict[str, object]:
    """Exercise the new mental-card item callbacks without altering policy decks.

    A single real basic card (15) is recycled through a clean twelve-turn
    synthetic stage. Its +2 good condition and genki naturally establish the
    >=5 condition and sustain nine eligible uses, exhausting the item limit.
    This is a rules trace, not a candidate-policy episode or score benchmark.
    """
    if item_id not in {"448", "449"}:
        raise ValueError(f"no directed lifecycle fixture for item: {item_id}")
    card = CardRef("directed-basic-mental", "15")
    state = kernel.initialize_stage_entry(
        StrategyState(
            turn=1, turns_remaining=12, score=0, stamina=30, max_stamina=30,
            genki=0, actions_remaining=1, hand_limit=5, score_multiplier=1,
            deck=DeckBelief(all_cards=(card,), hand=(card,), draw_pile=()),
            items=(item_id,),
        ),
        plan_type=PlanType.SENSE,
    )
    steps = []
    actual_applications = 0
    triggered = False
    reason = "maximum_decisions_exceeded"
    expected_applications = sum(
        effect.schedule_limit or 0
        for effect in kernel.items[item_id].persistent_effects
    )
    for _ in range(32):
        if state.terminal:
            reason = "terminal"
            break
        legal = kernel.legal_actions(state, plan_type=PlanType.SENSE, decision_kind=DecisionKind.CARD)
        intended = GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id)
        action = intended if intended in legal else GameAction(ActionKind.END_TURN)
        if action not in legal:
            reason = "directed_action_not_legal"
            break
        before = state
        branches = kernel.transition(state, action, plan_type=PlanType.SENSE)
        if len(branches) != 1 or branches[0].probability != 1:
            reason = "directed_fixture_requires_deterministic_transition"
            break
        branch = branches[0]
        state = branch.state
        item_events = tuple(event for event in branch.events if event.source_id == f"item:{item_id}")
        triggered |= any(event.event_type == "TRIGGER_SCHEDULED" for event in item_events)
        actual_applications += sum(event.event_type == "EXCELLENT_CONDITION" for event in item_events)
        steps.append({
            "turn": before.turn,
            "action": action.kind.value,
            "card_id": "15" if action.kind is ActionKind.PLAY_CARD else None,
            "good_condition_before": before.good_condition_turns,
            "good_condition_after": state.good_condition_turns,
            "item_schedules_before": [asdict(effect) for effect in before.scheduled_effects if effect.card_id == item_id],
            "item_schedules_after": [asdict(effect) for effect in state.scheduled_effects if effect.card_id == item_id],
            "item_events": [asdict(event) for event in item_events],
            "deck_conservation_valid": state.deck.conservation_valid,
        })
    return {
        "probe_id": f"mental-counter-item-{item_id}-v1",
        "evidence_kind": "DIRECTED_OFFLINE_RULE_TRACE",
        "included_in_policy_comparison": False,
        "item_id": item_id,
        "stage_turns": 12,
        "initial_stamina": 30,
        "initial_good_condition": 0,
        "card_ids": ["15"],
        "triggered": triggered,
        "expected_applications": expected_applications,
        "actual_applications": actual_applications,
        "terminal": state.terminal,
        "terminal_schedules_cleared": not state.scheduled_effects,
        "reason": reason,
        "passed": (
            reason == "terminal" and triggered and expected_applications > 0
            and actual_applications == expected_applications
            and not state.scheduled_effects
            and all(step["deck_conservation_valid"] for step in steps)
        ),
        "steps": steps,
    }


def _slice_report(
    *,
    plan_type: PlanType,
    logic_route: LogicRoute | None,
    seed: int,
    episodes_per_inventory: int,
    bootstrap_rounds: int,
    stage_turns: int,
) -> dict[str, object]:
    card_catalog, base_ids, acquisition_pool = _evaluation_pool(
        plan_type,
        logic_route,
    )
    item_catalog = compile_item_catalog(P_ITEMS, plan_type=plan_type)
    finite_limit = _finite_item_limit(item_catalog.items)
    if stage_turns < finite_limit:
        raise ValueError(f"stage_turns={stage_turns} is shorter than {plan_type.value} maximum finite item schedule limit {finite_limit}")
    item_ids = tuple(sorted(item_catalog.items, key=int))
    scenarios = build_fixed_item_inventory_scenarios(
        item_ids,
        pairing_seed=seed + 275,
    )
    base_generator = NiaProDynamicDeckGenerator(
        card_catalog.cards,
        plan_type=plan_type,
        base_card_ids=base_ids,
        acquisition_pool=acquisition_pool,
        upgrade_map=card_catalog.upgrade_map,
    )
    generator = NiaProItemLifecycleGenerator(
        base_generator,
        item_catalog.items,
        inventory_scenarios=scenarios,
        stage_turns=stage_turns,
    )
    samples = generator.generate(
        seed=seed,
        episode_count_per_inventory=episodes_per_inventory,
    )
    kernel = PythonNiaProKernel(card_catalog.cards, items=item_catalog.items)
    prepared, inventory_ledger_item_ids, scheduled_effect_item_ids, initial_item_schedule_ref_count = _prepare_stage_entry_samples(
        samples,
        kernel,
        item_catalog.items,
        plan_type=plan_type,
    )
    candidate = PhaseAwareHighScorePolicy(
        kernel,
        exact_turns_remaining=2,
        exact_config=PlannerConfig(
            max_decision_depth=16,
            timeout_ms=1000.0,
            require_terminal_leaf=True,
        ),
        maximum_exact_hand_size_for_multi_action=3,
    )
    candidate_simulations = []
    for sample in prepared:
        try:
            candidate_simulations.append(simulate_episode_with_trace(sample, candidate, kernel))
        except HighScoreContractError as error:
            raise RuntimeError(f"{sample.episode_id} lifecycle simulation violated the state contract: {error}") from error
    candidate_simulations = tuple(candidate_simulations)
    candidate_results = tuple(simulation.result for simulation in candidate_simulations)
    candidate_metrics = summarize_policy_results(candidate, candidate_results)
    baseline_runs = [
        evaluate_policy(prepared, ImmediateScorePolicy(kernel), kernel),
        evaluate_policy(
            prepared,
            ImmediateScorePolicy(kernel, include_resource_potential=True),
            kernel,
        ),
    ]
    baseline_metrics, baseline_results = max(
        baseline_runs,
        key=lambda item: (item[0].mean_final_score, item[0].p10_final_score),
    )
    comparison = compare_paired_results(
        candidate_metrics,
        candidate_results,
        baseline_metrics,
        baseline_results,
        bootstrap_rounds=bootstrap_rounds,
        bootstrap_seed=seed + 550,
    )

    triggered_item_ids: set[str] = set()
    unexpected_item_event_ids: set[str] = set()
    all_terminal = True
    terminal_schedules_cleared = True
    for sample, simulation in zip(prepared, candidate_simulations, strict=True):
        inventory = set(sample.request.state.items)
        event_item_ids = {
            event.source_id.removeprefix("item:")
            for event in simulation.events
            if event.source_id.startswith("item:")
        }
        triggered_item_ids.update(
            event.source_id.removeprefix("item:")
            for event in simulation.events
            if event.event_type == "TRIGGER_SCHEDULED"
            and event.source_id.startswith("item:")
        )
        unexpected_item_event_ids.update(event_item_ids - inventory)
        all_terminal &= simulation.final_state.terminal and not simulation.result.stopped
        terminal_schedules_cleared &= not simulation.final_state.scheduled_effects

    singleton_ids = {scenario.item_ids[0] for scenario in scenarios if len(scenario.item_ids) == 1}
    paired_ids = {item_id for scenario in scenarios if len(scenario.item_ids) == 2 for item_id in scenario.item_ids}
    candidate_lifecycle_passed = (
        inventory_ledger_item_ids == set(item_ids)
        and singleton_ids == set(item_ids)
        and paired_ids == set(item_ids)
        and triggered_item_ids == scheduled_effect_item_ids
        and all_terminal
        and terminal_schedules_cleared
        and not unexpected_item_event_ids
        and candidate_metrics.stop_rate == 0.0
        and candidate_metrics.illegal_action_rate == 0.0
    )
    directed_probes = tuple(
        _directed_mental_counter_probe(kernel, item_id)
        for item_id in sorted(scheduled_effect_item_ids - triggered_item_ids, key=int)
        if plan_type is PlanType.SENSE and item_id in {"448", "449"}
    )
    directed_covered_ids = {probe["item_id"] for probe in directed_probes if probe["passed"]}
    lifecycle_passed = (
        inventory_ledger_item_ids == set(item_ids)
        and singleton_ids == set(item_ids)
        and paired_ids == set(item_ids)
        and triggered_item_ids | directed_covered_ids == scheduled_effect_item_ids
        and all(probe["passed"] for probe in directed_probes)
        and all_terminal and terminal_schedules_cleared
        and not unexpected_item_event_ids
        and candidate_metrics.stop_rate == 0.0
        and candidate_metrics.illegal_action_rate == 0.0
    )
    inventory_size_counts = Counter(len(scenario.item_ids) for scenario in scenarios)
    logic_route_counts: dict[str, int] | None = None
    if plan_type is PlanType.LOGIC:
        logic_route_counts = Counter(classify_logic_route(sample.request.state, card_catalog.cards).value for sample in prepared)
        for route in LogicRoute:
            logic_route_counts.setdefault(route.value, 0)
        observed_routes = {route_name for route_name, count in logic_route_counts.items() if count}
        if logic_route is not None and observed_routes != {logic_route.value}:
            raise RuntimeError(f"generated Logic lifecycle slice drifted from {logic_route.value}: {dict(sorted(logic_route_counts.items()))}")

    return {
        "slice_id": (plan_type.value.lower() if logic_route is None else f"logic-{logic_route.value.lower().replace('_', '-')}"),
        "plan_type": plan_type.value,
        "logic_route": logic_route.value if logic_route is not None else None,
        "seed": seed,
        "stage_turns": stage_turns,
        "episodes_per_inventory": episodes_per_inventory,
        "episode_count": len(prepared),
        "unique_deck_signatures": len({sample.deck_signature for sample in prepared}),
        "acquisition_pool_size": len(acquisition_pool),
        "card_catalog": {
            "supported_rows": card_catalog.supported_rows,
            "total_rows": card_catalog.total_rows,
            "source_sha256": card_catalog.source_sha256,
        },
        "item_catalog": {
            "supported_rows": item_catalog.supported_rows,
            "total_rows": item_catalog.total_rows,
            "source_sha256": item_catalog.source_sha256,
            "maximum_finite_schedule_limit": finite_limit,
        },
        "inventory_scenarios": {
            "count": len(scenarios),
            "by_item_count": {str(item_count): count for item_count, count in sorted(inventory_size_counts.items())},
            "singleton_item_coverage": len(singleton_ids),
            "paired_item_coverage": len(paired_ids),
        },
        "lifecycle": {
            "gate_scope": "candidate lifecycle consistency plus explicitly separated directed rule coverage",
            "candidate_lifecycle_passed": candidate_lifecycle_passed,
            "candidate_trigger_coverage_complete": triggered_item_ids == scheduled_effect_item_ids,
            "candidate_untriggered_item_ids": sorted(scheduled_effect_item_ids - triggered_item_ids, key=int),
            "directed_rule_covered_item_ids": sorted(directed_covered_ids, key=int),
            "combined_rule_covered_item_count": len(triggered_item_ids | directed_covered_ids),
            "directed_rule_probes": directed_probes,
            "initialized_at_turn_one": True,
            "initial_item_schedule_ref_count": initial_item_schedule_ref_count,
            "inventory_ledger_item_count": len(inventory_ledger_item_ids),
            "scheduled_effect_item_count": len(scheduled_effect_item_ids),
            "zero_schedule_item_ids": sorted(
                set(item_ids) - scheduled_effect_item_ids,
                key=int,
            ),
            "triggered_item_count": len(triggered_item_ids),
            "triggered_item_ids": sorted(triggered_item_ids, key=int),
            "unexpected_item_event_ids": sorted(unexpected_item_event_ids, key=int),
            "all_candidate_episodes_terminal": all_terminal,
            "terminal_schedules_cleared": terminal_schedules_cleared,
            "passed": lifecycle_passed,
        },
        "logic_route_episode_counts": (dict(sorted(logic_route_counts.items())) if logic_route_counts is not None else None),
        "candidate": _deterministic_metrics(candidate_metrics),
        "candidate_stops": [
            {
                "episode_id": result.episode_id,
                "reason": result.reason,
            }
            for result in candidate_results
            if result.stopped
        ],
        "baselines": [
            _deterministic_metrics(metrics)
            for metrics, _results in baseline_runs
        ],
        "strongest_baseline": _deterministic_metrics(baseline_metrics),
        "paired_comparison": asdict(comparison),
    }


def build_report(
    *,
    seed: int,
    episodes_per_inventory: int,
    bootstrap_rounds: int,
    stage_turns: int,
) -> dict[str, object]:
    if episodes_per_inventory < 1:
        raise ValueError("episodes_per_inventory must be positive")
    slice_specs = [(PlanType.SENSE, None)]
    slice_specs.extend((PlanType.LOGIC, route) for route in LogicRoute)
    slice_specs.append((PlanType.ANOMALY, None))
    slices = tuple(
        _slice_report(
            plan_type=plan_type,
            logic_route=logic_route,
            seed=seed + index * 100_000,
            episodes_per_inventory=episodes_per_inventory,
            bootstrap_rounds=bootstrap_rounds,
            stage_turns=stage_turns,
        )
        for index, (plan_type, logic_route) in enumerate(slice_specs)
    )
    return {
        "schema_version": 2,
        "dataset_id": "task275-p-item-stage-entry-paired-v3",
        "policy_dataset_id": "task275-p-item-stage-entry-paired-v2",
        "directed_rule_probe_revision": 1,
        "dataset_scope": (
            "synthetic turn-one stage-entry states using every currently compiled "
            "P-item as singleton and in a deterministic pair inventory. Each deck seed "
            "has an explicit synthetic turn-attribute cycle shared by its paired runs. Logic is "
            "stratified into Good Impression, Motivation/Genki and Mixed deck routes"
        ),
        "effect_ir_version": EFFECT_IR_VERSION,
        "latency_scope": (
            "Wall-clock latency is intentionally excluded so the frozen score and "
            "lifecycle report is byte-reproducible; performance remains covered by "
            "the separate architecture probe."
        ),
        "seed": seed,
        "stage_turns": stage_turns,
        "episodes_per_inventory": episodes_per_inventory,
        "slices": slices,
        "all_lifecycle_gates_passed": all(slice_report["lifecycle"]["passed"] for slice_report in slices),
        "all_candidate_lifecycle_gates_passed": all(slice_report["lifecycle"]["candidate_lifecycle_passed"] for slice_report in slices),
        "all_policy_promotion_gates_passed": all(slice_report["paired_comparison"]["promotion_passed"] for slice_report in slices),
        "boundary": (
            "This report proves lifecycle consistency only for compiled offline rules: "
            "inventories are present before turn one, persistent schedules are initialized "
            "once, candidate and baselines share identical decks/items/seeds, and terminal "
            "cleanup is checked. Directed rule probes supplement uncovered callback families "
            "in separate twelve-turn synthetic traces; their scores and episodes are excluded "
            "from all candidate and baseline metrics. The original candidate-only lifecycle "
            "gate and its trigger counts remain separately reported. Neither synthetic horizon is evidence of the "
            "actual NIA PRO turn count, attribute multipliers, Live recognition, or runtime readiness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate P-item lifecycles from paired turn-one stage entries.")
    parser.add_argument("--seed", type=int, default=675000)
    parser.add_argument("--episodes-per-inventory", type=int, default=2)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--stage-turns", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".local/task275-item-lifecycle-evaluation.json",
    )
    args = parser.parse_args()
    report = build_report(
        seed=args.seed,
        episodes_per_inventory=args.episodes_per_inventory,
        bootstrap_rounds=args.bootstrap_rounds,
        stage_turns=args.stage_turns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "all_lifecycle_gates_passed": report["all_lifecycle_gates_passed"],
                "all_policy_promotion_gates_passed": report["all_policy_promotion_gates_passed"],
                "slices": {
                    slice_report["slice_id"]: {
                        "episode_count": slice_report["episode_count"],
                        "lifecycle": {
                            key: value for key, value in slice_report["lifecycle"].items()
                            if key != "directed_rule_probes"
                        },
                        "paired_comparison": slice_report["paired_comparison"],
                    }
                    for slice_report in report["slices"]
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
