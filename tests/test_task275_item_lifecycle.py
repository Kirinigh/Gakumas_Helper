from __future__ import annotations

import unittest
from pathlib import Path
from dataclasses import replace

from agent.live_decision.high_score import (
    PlanType,
    LogicRoute,
    KernelError,
    ItemDefinition,
    EffectSourceKind,
    PythonNiaProKernel,
    ImmediateScorePolicy,
    ItemInventoryScenario,
    NiaProDynamicDeckGenerator,
    NiaProItemLifecycleGenerator,
    compile_item_catalog,
    compile_skill_card_catalog,
    simulate_episode_with_trace,
    build_fixed_item_inventory_scenarios,
)
from tools.task275_item_lifecycle_evaluate import (
    _evaluation_pool,
    _finite_item_limit,
    _definition_logic_route,
    _directed_mental_counter_probe,
)

ROOT = Path(__file__).resolve().parents[1]
SKILL_CARDS = ROOT / "assets/data/skill_cards.csv"
P_ITEMS = ROOT / "assets/data/p_items.csv"


class Task275ItemLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.SENSE)
        cls.items = compile_item_catalog(P_ITEMS, plan_type=PlanType.SENSE)
        cls.base_ids = tuple(sorted(cls.cards.supported_card_ids, key=int)[:5])

    def _base_generator(self) -> NiaProDynamicDeckGenerator:
        return NiaProDynamicDeckGenerator(
            self.cards.cards,
            plan_type=PlanType.SENSE,
            base_card_ids=self.base_ids,
            acquisition_pool=self.base_ids,
            minimum_deck_size=5,
            mutation_count_range=(0, 0),
        )

    def _lifecycle_generator(
        self,
        scenarios: tuple[ItemInventoryScenario, ...],
        *,
        stage_turns: int = 2,
    ) -> NiaProItemLifecycleGenerator:
        return NiaProItemLifecycleGenerator(
            self._base_generator(),
            self.items.items,
            inventory_scenarios=scenarios,
            stage_turns=stage_turns,
        )

    def test_fixed_inventory_scenarios_are_deterministic_and_cover_pairs(self) -> None:
        first = build_fixed_item_inventory_scenarios(("1", "2", "3"), pairing_seed=275)
        second = build_fixed_item_inventory_scenarios(("1", "2", "3"), pairing_seed=275)

        self.assertEqual(first, second)
        self.assertEqual(first[0], ItemInventoryScenario("empty", ()))
        self.assertEqual(
            {scenario.item_ids[0] for scenario in first if len(scenario.item_ids) == 1},
            {"1", "2", "3"},
        )
        paired_ids = {item_id for scenario in first if len(scenario.item_ids) == 2 for item_id in scenario.item_ids}
        self.assertEqual(paired_ids, {"1", "2", "3"})
        self.assertEqual(len(first), 6)

    def test_logic_lifecycle_pools_are_stratified_by_route(self) -> None:
        for route in LogicRoute:
            with self.subTest(route=route.value):
                catalog, base_ids, acquisition_pool = _evaluation_pool(
                    PlanType.LOGIC,
                    route,
                )
                self.assertEqual(len(base_ids), 7)
                self.assertGreaterEqual(len(acquisition_pool), 7)
                self.assertEqual(
                    {_definition_logic_route(card_id, catalog.cards) for card_id in acquisition_pool},
                    {route},
                )

    def test_four_turn_horizon_covers_current_finite_item_use_limits(self) -> None:
        for plan_type in PlanType:
            with self.subTest(plan_type=plan_type.value):
                catalog = compile_item_catalog(P_ITEMS, plan_type=plan_type)
                self.assertEqual(_finite_item_limit(catalog.items), 4)

    def test_lifecycle_generator_reuses_decks_but_starts_clean_at_turn_one(self) -> None:
        scenarios = (
            ItemInventoryScenario("empty", ()),
            ItemInventoryScenario("single-3", ("3",)),
        )
        samples = self._lifecycle_generator(scenarios, stage_turns=4).generate(
            seed=275000,
            episode_count_per_inventory=2,
        )

        self.assertEqual(len(samples), 4)
        by_seed: dict[int, list[object]] = {}
        for sample in samples:
            by_seed.setdefault(sample.seed, []).append(sample)
            state = sample.request.state
            self.assertEqual(state.turn, 1)
            self.assertEqual(state.turns_remaining, 4)
            self.assertEqual(len(state.turn_attributes), state.turns_remaining)
            self.assertEqual(set(state.turn_attributes), {"vocal", "dance", "visual"})
            self.assertEqual(state.score, 0)
            self.assertEqual(state.stamina, state.max_stamina)
            self.assertEqual(state.genki, 0)
            self.assertEqual(state.scheduled_effects, ())
            self.assertFalse(state.persistent_effects_initialized)
        for paired in by_seed.values():
            self.assertEqual(len(paired), 2)
            self.assertEqual(len({sample.deck_signature for sample in paired}), 1)
            self.assertEqual(len({sample.mutation_events for sample in paired}), 1)
            self.assertEqual(len({sample.request.state.turn_attributes for sample in paired}), 1)
            self.assertEqual({sample.request.state.items for sample in paired}, {(), ("3",)})
        self.assertEqual(len({sample.episode_id for sample in samples}), len(samples))

    def test_lifecycle_generator_rejects_unknown_and_plan_mismatched_items(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown definitions"):
            NiaProItemLifecycleGenerator(
                self._base_generator(),
                self.items.items,
                inventory_scenarios=(ItemInventoryScenario("unknown", ("missing",)),),
                stage_turns=2,
            )

        mismatched = dict(self.items.items)
        mismatched["logic-only"] = ItemDefinition(
            "logic-only",
            "logic-only",
            plan=PlanType.LOGIC,
        )
        with self.assertRaisesRegex(ValueError, "not legal for SENSE"):
            NiaProItemLifecycleGenerator(
                self._base_generator(),
                mismatched,
                inventory_scenarios=(ItemInventoryScenario("logic", ("logic-only",)),),
                stage_turns=2,
            )

    def test_stage_entry_initialization_materializes_exact_typed_item_rules(self) -> None:
        sample = self._lifecycle_generator((ItemInventoryScenario("single-3", ("3",)),)).generate(
            seed=275100,
            episode_count_per_inventory=1,
        )[0]
        kernel = PythonNiaProKernel(self.cards.cards, items=self.items.items)

        initialized = kernel.initialize_stage_entry(
            sample.request.state,
            plan_type=PlanType.SENSE,
        )

        expected_count = len(self.items.items["3"].persistent_effects)
        item_schedules = tuple(scheduled for scheduled in initialized.scheduled_effects if scheduled.source_kind is EffectSourceKind.ITEM)
        self.assertTrue(initialized.persistent_effects_initialized)
        self.assertEqual(len(item_schedules), expected_count)
        self.assertEqual(
            {(scheduled.source_id, scheduled.card_id, scheduled.definition_effect_index) for scheduled in item_schedules},
            {("item:3", "3", index) for index in range(expected_count)},
        )
        self.assertFalse(sample.request.state.persistent_effects_initialized)

    def test_stage_entry_rejects_mid_lifecycle_injection_and_reinitialization(self) -> None:
        sample = self._lifecycle_generator((ItemInventoryScenario("single-3", ("3",)),)).generate(
            seed=275200,
            episode_count_per_inventory=1,
        )[0]
        kernel = PythonNiaProKernel(self.cards.cards, items=self.items.items)

        with self.assertRaisesRegex(KernelError, "requires_live_turn_one"):
            kernel.initialize_stage_entry(
                replace(sample.request.state, turn=10),
                plan_type=PlanType.SENSE,
            )
        initialized = kernel.initialize_stage_entry(
            sample.request.state,
            plan_type=PlanType.SENSE,
        )
        with self.assertRaisesRegex(KernelError, "already_initialized"):
            kernel.initialize_stage_entry(initialized, plan_type=PlanType.SENSE)

    def test_traced_episode_reaches_terminal_and_clears_item_schedules(self) -> None:
        sample = self._lifecycle_generator(
            (ItemInventoryScenario("single-3", ("3",)),),
            stage_turns=2,
        ).generate(seed=275300, episode_count_per_inventory=1)[0]
        kernel = PythonNiaProKernel(self.cards.cards, items=self.items.items)
        initialized = kernel.initialize_stage_entry(sample.request.state, plan_type=PlanType.SENSE)
        prepared = replace(sample, request=replace(sample.request, state=initialized))

        simulation = simulate_episode_with_trace(
            prepared,
            ImmediateScorePolicy(kernel, include_resource_potential=True),
            kernel,
        )

        self.assertFalse(simulation.result.stopped)
        self.assertFalse(simulation.result.illegal_action)
        self.assertTrue(simulation.final_state.terminal)
        self.assertTrue(simulation.final_state.persistent_effects_initialized)
        self.assertEqual(simulation.final_state.scheduled_effects, ())
        self.assertIn("item:3", {event.source_id for event in simulation.events})

    def test_finite_recurring_item_is_pruned_before_it_can_underflow(self) -> None:
        sample = self._lifecycle_generator(
            (ItemInventoryScenario("single-19", ("19",)),),
            stage_turns=4,
        ).generate(seed=275400, episode_count_per_inventory=1)[0]
        kernel = PythonNiaProKernel(self.cards.cards, items=self.items.items)
        initialized = kernel.initialize_stage_entry(sample.request.state, plan_type=PlanType.SENSE)
        prepared = replace(sample, request=replace(sample.request, state=initialized))

        simulation = simulate_episode_with_trace(
            prepared,
            ImmediateScorePolicy(kernel, include_resource_potential=True),
            kernel,
        )

        triggers = [event for event in simulation.events if event.source_id == "item:19" and event.event_type == "TRIGGER_SCHEDULED"]
        self.assertFalse(simulation.result.stopped)
        self.assertTrue(simulation.final_state.terminal)
        self.assertEqual(len(triggers), 2)
        self.assertEqual(simulation.final_state.scheduled_effects, ())

    def test_directed_mental_callbacks_establish_condition_and_exhaust_real_item_limit(self) -> None:
        kernel = PythonNiaProKernel(self.cards.cards, items=self.items.items)
        for item_id in ("448", "449"):
            with self.subTest(item_id=item_id):
                report = _directed_mental_counter_probe(kernel, item_id)
                self.assertTrue(report["passed"])
                self.assertFalse(report["included_in_policy_comparison"])
                self.assertEqual(report["initial_good_condition"], 0)
                self.assertEqual(report["card_ids"], ["15"])
                self.assertEqual((report["actual_applications"], report["expected_applications"]), (3, 3))
                self.assertTrue(report["terminal_schedules_cleared"])
                played = [step for step in report["steps"] if step["action"] == "PLAY_CARD"]
                self.assertEqual(len(played), 12)
                counters = [step["item_schedules_after"][0]["effect_counter"] for step in played]
                self.assertEqual(counters, [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                applications = [
                    step["turn"] for step in played
                    if any(event["event_type"] == "EXCELLENT_CONDITION" for event in step["item_events"])
                ]
                self.assertEqual(applications, [6, 9, 12])
                self.assertEqual(played[-1]["item_schedules_after"][0]["remaining_uses"], 0)

    def test_directed_rule_probe_is_repeatable_and_does_not_mutate_paired_samples(self) -> None:
        generator = self._lifecycle_generator((ItemInventoryScenario("single-448", ("448",)),))
        before = generator.generate(seed=675000, episode_count_per_inventory=2)
        kernel = PythonNiaProKernel(self.cards.cards, items=self.items.items)
        first = _directed_mental_counter_probe(kernel, "448")
        self.assertEqual(first, _directed_mental_counter_probe(kernel, "448"))
        self.assertEqual(before, generator.generate(seed=675000, episode_count_per_inventory=2))
        with self.assertRaisesRegex(ValueError, "no directed lifecycle fixture"):
            _directed_mental_counter_probe(kernel, "3")


if __name__ == "__main__":
    unittest.main()
