import unittest
from pathlib import Path
from dataclasses import replace

from agent.live_decision.high_score import (
    Stance,
    CardRef,
    CardZone,
    PlanType,
    ActionKind,
    DeckBelief,
    EffectKind,
    EffectSpec,
    GameAction,
    LogicRoute,
    KernelError,
    CardCostKind,
    DatasetSplit,
    DecisionKind,
    ConditionKind,
    DeckKnowledge,
    PendingChoice,
    PlannerConfig,
    SchedulePhase,
    TimedModifier,
    CardDefinition,
    DecisionOutcome,
    EffectCondition,
    EffectSourceKind,
    PythonNiaProKernel,
    ScheduledEffectRef,
    ImmediateScorePolicy,
    HighScoreContractError,
    RobustHighScorePlanner,
    PhaseAwareHighScorePolicy,
    NiaProDynamicDeckGenerator,
    evaluate_policy,
    load_golden_suite,
    classify_logic_route,
    compile_item_catalog,
    compile_drink_catalog,
    compare_paired_results,
    compile_skill_card_catalog,
    build_architecture_probe_report,
)
from agent.live_decision.high_score.challenger import (
    ActionFeatureEncoder,
    LinearBehaviorCloningPolicy,
    collect_expert_choices,
    train_linear_behavior_cloning,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/fixtures/live_decision/task275_anomaly_golden_v1.json"
SKILL_CARDS = ROOT / "assets/data/skill_cards.csv"
P_DRINKS = ROOT / "assets/data/p_drinks.csv"
P_ITEMS = ROOT / "assets/data/p_items.csv"


class Task275HighScoreContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_golden_suite(GOLDEN)
        cls.kernel = PythonNiaProKernel(cls.suite.cards, cls.suite.drinks, ruleset_id=cls.suite.ruleset_id)
        cls.planner = RobustHighScorePlanner(
            cls.kernel,
            PlannerConfig(max_decision_depth=8, timeout_ms=2000.0, require_terminal_leaf=True),
        )

    def test_anomaly_golden_suite_matches_complete_horizon_decisions(self) -> None:
        self.assertEqual(len(self.suite.cases), 5)
        for case in self.suite.cases:
            with self.subTest(case=case.case_id):
                response = self.planner.decide(case.request)
                self.assertEqual(response.outcome, case.expectation.outcome)
                self.assertEqual(response.action, case.expectation.action)
                if case.expectation.expected_final_score is not None:
                    self.assertIsNotNone(response.value)
                    assert response.value is not None
                    self.assertAlmostEqual(response.value.expected_final_score, case.expectation.expected_final_score)
                if case.expectation.p10_final_score is not None:
                    self.assertIsNotNone(response.value)
                    assert response.value is not None
                    self.assertAlmostEqual(response.value.p10_final_score, case.expectation.p10_final_score)
                self.assertEqual(response.to_dict()["has_action_intent"], response.outcome is DecisionOutcome.DECIDE)

    def test_conserve_release_event_order_and_rounding_are_frozen(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "release-chain-beats-immediate")
        branches = self.kernel.transition(
            case.request.state,
            GameAction(kind=ActionKind.PLAY_CARD, card_instance_id="release-1"),
            plan_type=case.request.plan_type,
        )
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assertEqual(branch.state.stance, Stance.AGGRESSIVE_1)
        self.assertEqual(branch.state.passion, 12.5)
        self.assertEqual(branch.state.genki, 5)
        self.assertEqual(branch.state.actions_remaining, 1)
        self.assertEqual(branch.state.score, 42)
        event_types = [event.event_type for event in branch.events]
        self.assertLess(event_types.index("MOVE_CARD"), event_types.index("STANCE"))
        self.assertLess(event_types.index("STANCE"), event_types.index("SCORE"))

    def test_known_composition_draw_expands_probability_mass(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "unknown-order-draw-uses-chance-branches")
        branches = self.kernel.transition(
            case.request.state,
            GameAction(kind=ActionKind.PLAY_CARD, card_instance_id="draw-1"),
            plan_type=case.request.plan_type,
        )
        self.assertEqual(len(branches), 2)
        self.assertAlmostEqual(sum(branch.probability for branch in branches), 1.0)
        self.assertEqual({branch.events[-1].detail for branch in branches}, {"burst-2", "small-1"})
        self.assertEqual({branch.probability for branch in branches}, {0.5})

    def test_equivalent_draws_merge_without_changing_probability_mass(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "unknown-order-draw-uses-chance-branches")
        burst = next(card for card in case.request.state.deck.draw_pile if card.card_id == "an-burst")
        duplicate = CardRef("burst-duplicate", "an-burst")
        deck = replace(
            case.request.state.deck,
            all_cards=case.request.state.deck.hand + (burst, duplicate),
            draw_pile=(burst, duplicate),
        )
        state = replace(case.request.state, deck=deck)
        branches = self.kernel.transition(
            state,
            GameAction(kind=ActionKind.PLAY_CARD, card_instance_id="draw-1"),
            plan_type=case.request.plan_type,
        )
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0].probability, 1.0)
        self.assertIn("an-burst", {card.card_id for card in branches[0].state.deck.hand})

    def test_multidraw_merges_order_permutations_into_combinations(self) -> None:
        card_ids = tuple(self.suite.cards)[:7]
        cards = tuple(CardRef(f"combination-{index}", card_id) for index, card_id in enumerate(card_ids))
        deck = DeckBelief(
            all_cards=cards,
            hand=(),
            draw_pile=cards,
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            hand_limit=5,
            deck=deck,
        )
        branches = self.kernel.transition(
            state,
            GameAction(kind=ActionKind.END_TURN),
            plan_type=self.suite.cases[0].request.plan_type,
        )
        self.assertEqual(len(branches), 21)
        self.assertAlmostEqual(sum(branch.probability for branch in branches), 1.0)

    def test_deck_conservation_failure_is_never_promoted_to_probability(self) -> None:
        a = CardRef("a", "an-small")
        b = CardRef("b", "an-small")
        belief = DeckBelief(
            all_cards=(a, b),
            hand=(a,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        self.assertFalse(belief.conservation_valid)
        with self.assertRaises(HighScoreContractError):
            belief.validate()

    def test_unregistered_card_anywhere_in_deck_fails_closed(self) -> None:
        known = CardRef("registered-card", "an-small")
        unknown = CardRef("unregistered-card", "unsupported-card")
        deck = DeckBelief(
            all_cards=(known, unknown),
            hand=(known,),
            draw_pile=(unknown,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(self.suite.cases[0].request.state, deck=deck)
        with self.assertRaisesRegex(KernelError, "unregistered_card_in_deck"):
            self.kernel.legal_actions(
                state,
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.CARD,
            )

    def test_nonterminal_search_cutoff_stops_instead_of_using_partial_ranking(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "charge-enables-full-power-burst")
        state = replace(case.request.state, turns_remaining=2, future_score_multipliers=(1.0,))
        request = replace(case.request, state=state)
        planner = RobustHighScorePlanner(
            self.kernel,
            PlannerConfig(max_decision_depth=1, timeout_ms=2000.0, require_terminal_leaf=True),
        )
        response = planner.decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.STOP_TASK)
        self.assertIn("search_horizon_not_terminal", response.reason)

    def test_phase_aware_policy_uses_one_step_before_exact_endgame(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "charge-enables-full-power-burst")
        policy = PhaseAwareHighScorePolicy(
            self.kernel,
            exact_turns_remaining=2,
            exact_config=PlannerConfig(max_decision_depth=16, timeout_ms=2000.0),
        )
        early = policy.decide(replace(case.request, state=replace(case.request.state, turns_remaining=3)))
        exact = policy.decide(case.request)
        self.assertEqual(early.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(early.trace)
        assert early.trace is not None
        self.assertEqual(early.trace.planner, "one-step-state-potential")
        self.assertEqual(exact.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(exact.trace)
        assert exact.trace is not None
        self.assertEqual(exact.trace.planner, "bounded-expectimax")

        template = case.request.state.deck.hand[0]
        cards = tuple(CardRef(f"complex-{index}", template.card_id) for index in range(5))
        complex_deck = DeckBelief(all_cards=cards, hand=cards, draw_pile=())
        complex_state = replace(
            case.request.state,
            turns_remaining=2,
            actions_remaining=2,
            deck=complex_deck,
        )
        complex_request = replace(
            case.request,
            state=complex_state,
            card_slots=tuple((card.instance_id, index) for index, card in enumerate(cards)),
        )
        guarded = policy.decide(complex_request)
        self.assertEqual(guarded.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(guarded.trace)
        assert guarded.trace is not None
        self.assertEqual(guarded.trace.planner, "one-step-state-potential")
        self.assertIn("phase_policy:exact_complexity_guard", guarded.trace.reason_codes)

        one_turn_guarded = policy.decide(
            replace(complex_request, state=replace(complex_state, turns_remaining=1))
        )
        self.assertEqual(one_turn_guarded.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(one_turn_guarded.trace)
        assert one_turn_guarded.trace is not None
        self.assertEqual(one_turn_guarded.trace.planner, "one-step-state-potential")

        expanding_cards = (
            CardRef("expanding-draw", "an-draw"),
            *(CardRef(f"expanding-small-{index}", "an-small") for index in range(4)),
        )
        expanding_deck = DeckBelief(all_cards=expanding_cards, hand=expanding_cards, draw_pile=())
        expanding_state = replace(
            case.request.state,
            turns_remaining=2,
            actions_remaining=1,
            deck=expanding_deck,
        )
        expanding_request = replace(
            case.request,
            state=expanding_state,
            card_slots=tuple((card.instance_id, index) for index, card in enumerate(expanding_cards)),
        )
        expanding_guarded = policy.decide(expanding_request)
        self.assertEqual(expanding_guarded.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(expanding_guarded.trace)
        assert expanding_guarded.trace is not None
        self.assertEqual(expanding_guarded.trace.planner, "one-step-state-potential")
        self.assertIn("phase_policy:exact_complexity_guard", expanding_guarded.trace.reason_codes)

        refill_cards = tuple(CardRef(f"refill-{index}", "an-small") for index in range(5))
        refill_deck = DeckBelief(
            all_cards=refill_cards,
            hand=refill_cards[:3],
            draw_pile=refill_cards[3:],
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        refill_state = replace(
            case.request.state,
            turns_remaining=2,
            actions_remaining=1,
            hand_limit=5,
            deck=refill_deck,
        )
        refill_request = replace(
            case.request,
            state=refill_state,
            card_slots=tuple(
                (card.instance_id, index) for index, card in enumerate(refill_deck.hand)
            ),
        )
        refill_guarded = policy.decide(refill_request)
        self.assertEqual(refill_guarded.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(refill_guarded.trace)
        assert refill_guarded.trace is not None
        self.assertEqual(refill_guarded.trace.planner, "one-step-state-potential")
        self.assertIn("phase_policy:exact_complexity_guard", refill_guarded.trace.reason_codes)

        pending = PendingChoice(
            ActionKind.HOLD_CARD,
            "external-source",
            tuple(card.instance_id for card in cards),
        )
        secondary_state = replace(complex_state, turns_remaining=1, pending_choice=pending)
        secondary_request = replace(
            complex_request,
            state=secondary_state,
            decision_kind=DecisionKind.SECONDARY,
        )
        secondary_guarded = policy.decide(secondary_request)
        self.assertEqual(secondary_guarded.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(secondary_guarded.trace)
        assert secondary_guarded.trace is not None
        self.assertEqual(secondary_guarded.trace.planner, "one-step-state-potential")

        strict_policy = PhaseAwareHighScorePolicy(
            self.kernel,
            exact_turns_remaining=2,
            exact_config=PlannerConfig(max_decision_depth=1, timeout_ms=2000.0),
        )
        strict_request = replace(
            case.request,
            state=replace(case.request.state, turns_remaining=3),
            utility=replace(case.request.utility, baseline_p10=0.0),
        )
        strict = strict_policy.decide(strict_request)
        self.assertEqual(strict.outcome, DecisionOutcome.STOP_TASK)
        self.assertIn("search_horizon_not_terminal", strict.reason)

    def test_phase_policy_guards_future_large_secondary_draw(self) -> None:
        simple = CardDefinition(
            card_id="future-secondary-simple",
            name="Future secondary simple card",
            plan=PlanType.ANOMALY,
            effects=(EffectSpec(EffectKind.SCORE, amount=1),),
        )
        secondary = CardDefinition(
            card_id="future-secondary-card",
            name="Future secondary branching card",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.HOLD_SELECTED,
                    target_zones=(CardZone.DRAW_PILE, CardZone.DISCARD),
                ),
            ),
        )
        kernel = PythonNiaProKernel({simple.card_id: simple, secondary.card_id: secondary})
        hand = tuple(CardRef(f"future-hand-{index}", simple.card_id) for index in range(5))
        draw = (CardRef("future-secondary", secondary.card_id),) + tuple(
            CardRef(f"future-draw-{index}", simple.card_id) for index in range(4)
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            score=0,
            stance=Stance.NEUTRAL,
            actions_remaining=1,
            hand_limit=5,
            deck=DeckBelief(
                all_cards=hand + draw,
                hand=hand,
                draw_pile=draw,
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
        )
        request = replace(
            self.suite.cases[0].request,
            state=state,
            card_slots=tuple((card.instance_id, index) for index, card in enumerate(hand)),
        )
        policy = PhaseAwareHighScorePolicy(
            kernel,
            exact_turns_remaining=2,
            exact_config=PlannerConfig(max_decision_depth=16, timeout_ms=500.0),
        )
        response = policy.decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(response.trace)
        assert response.trace is not None
        self.assertEqual(response.trace.planner, "one-step-state-potential")
        self.assertIn("phase_policy:exact_complexity_guard", response.trace.reason_codes)

    def test_phase_policy_guards_future_expanding_draw(self) -> None:
        simple = CardDefinition(
            card_id="future-expansion-simple",
            name="Future expansion simple card",
            plan=PlanType.ANOMALY,
            effects=(EffectSpec(EffectKind.SCORE, amount=1),),
        )
        expanding = CardDefinition(
            card_id="future-expansion-card",
            name="Future expanding card",
            plan=PlanType.ANOMALY,
            effects=(EffectSpec(EffectKind.DRAW, amount=3),),
        )
        kernel = PythonNiaProKernel({simple.card_id: simple, expanding.card_id: expanding})
        hand = tuple(CardRef(f"future-expansion-hand-{index}", simple.card_id) for index in range(5))
        draw = (CardRef("future-expansion", expanding.card_id),) + tuple(
            CardRef(f"future-expansion-draw-{index}", simple.card_id) for index in range(4)
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            score=0,
            stance=Stance.NEUTRAL,
            actions_remaining=1,
            hand_limit=5,
            deck=DeckBelief(
                all_cards=hand + draw,
                hand=hand,
                draw_pile=draw,
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
        )
        request = replace(
            self.suite.cases[0].request,
            state=state,
            card_slots=tuple((card.instance_id, index) for index, card in enumerate(hand)),
        )
        response = PhaseAwareHighScorePolicy(
            kernel,
            exact_turns_remaining=2,
            exact_config=PlannerConfig(max_decision_depth=16, timeout_ms=500.0),
        ).decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(response.trace)
        assert response.trace is not None
        self.assertEqual(response.trace.planner, "one-step-state-potential")
        self.assertIn("phase_policy:exact_complexity_guard", response.trace.reason_codes)

    def test_one_step_policy_respects_resource_floors(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)
        card = CardRef("resource-floor", "370")
        deck = DeckBelief(all_cards=(card,), hand=(card,), draw_pile=())
        base_request = self.suite.cases[0].request
        state = replace(
            base_request.state,
            deck=deck,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
        )
        request = replace(
            base_request,
            state=state,
            utility=replace(base_request.utility, minimum_stamina=20),
            card_slots=((card.instance_id, 0),),
        )
        response = ImmediateScorePolicy(kernel, include_resource_potential=True).decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.DECIDE)
        self.assertEqual(response.action, GameAction(ActionKind.END_TURN))

    def test_phase_policy_guards_large_secondary_reachable_from_root(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("reachable-secondary-source", "528")
        candidates = tuple(CardRef(f"reachable-secondary-{index}", "370") for index in range(4))
        deck = DeckBelief(
            all_cards=(source, *candidates),
            hand=(source,),
            draw_pile=candidates,
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            deck=deck,
        )
        request = replace(
            self.suite.cases[0].request,
            state=state,
            card_slots=((source.instance_id, 0),),
        )
        policy = PhaseAwareHighScorePolicy(
            kernel,
            exact_config=PlannerConfig(max_decision_depth=16, timeout_ms=2000.0),
        )
        response = policy.decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.DECIDE)
        self.assertIsNotNone(response.trace)
        assert response.trace is not None
        self.assertEqual(response.trace.planner, "one-step-state-potential")
        self.assertIn("phase_policy:exact_complexity_guard", response.trace.reason_codes)

    def test_phase_policy_guards_large_free_play_choice_reachable_from_root(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("reachable-free-source", "793")
        candidates = tuple(CardRef(f"reachable-free-{index}", "370") for index in range(4))
        deck = DeckBelief(
            all_cards=(source, *candidates),
            hand=(source,),
            draw_pile=candidates,
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            deck=deck,
        )
        request = replace(
            self.suite.cases[0].request,
            state=state,
            card_slots=((source.instance_id, 0),),
        )
        response = PhaseAwareHighScorePolicy(
            kernel,
            exact_config=PlannerConfig(max_decision_depth=16, timeout_ms=2000.0),
        ).decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.DECIDE)
        assert response.trace is not None
        self.assertEqual(response.trace.planner, "one-step-state-potential")
        self.assertIn("phase_policy:exact_complexity_guard", response.trace.reason_codes)

    def test_equivalent_secondary_cards_in_different_zones_are_not_merged(self) -> None:
        card_in_hand = CardRef("hold-hand", "an-small")
        card_in_discard = CardRef("hold-discard", "an-small")
        deck = DeckBelief(
            all_cards=(card_in_hand, card_in_discard),
            hand=(card_in_hand,),
            draw_pile=(),
            discard=(card_in_discard,),
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            pending_choice=PendingChoice(
                ActionKind.HOLD_CARD,
                "hold-source",
                (card_in_hand.instance_id, card_in_discard.instance_id),
            ),
        )
        actions = (
            GameAction(ActionKind.HOLD_CARD, card_instance_id=card_in_hand.instance_id),
            GameAction(ActionKind.HOLD_CARD, card_instance_id=card_in_discard.instance_id),
        )
        merged = RobustHighScorePlanner._merge_equivalent_actions(state, actions)
        self.assertEqual(merged, actions)

    def test_selected_card_without_unique_source_slot_stops(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "release-chain-beats-immediate")
        response = self.planner.decide(replace(case.request, card_slots=()))
        self.assertEqual(response.outcome, DecisionOutcome.STOP_TASK)
        self.assertIn("selected_card_has_no_unique_source_slot", response.reason)

    def test_architecture_gate_selects_only_currently_executable_candidate(self) -> None:
        assessments = {
            "providers": [{
                "provider": "gakumas-tools-engine",
                "license": "BSD-3-Clause",
                "acquisition": "ACQUIRED",
                "same_fixture_status": "NOT_EXECUTED",
            }],
        }
        report = build_architecture_probe_report(self.suite, assessments, rounds=2)
        candidates = {candidate["candidate"]: candidate for candidate in report["candidates"]}
        self.assertEqual(candidates["python"]["status"], "PASS")
        self.assertEqual(candidates["adapter"]["status"], "FAIL")
        self.assertEqual(candidates["hybrid"]["status"], "FAIL")
        self.assertEqual(report["selection"]["selected"], "python")

    def test_dynamic_deck_generator_is_reproducible_and_group_split(self) -> None:
        generator = NiaProDynamicDeckGenerator(
            self.suite.cards,
            plan_type=self.suite.cases[0].request.plan_type,
            base_card_ids=(
                "an-conserve",
                "an-release",
                "an-charge",
                "an-burst",
                "an-immediate",
                "an-draw",
                "an-small",
            ),
            acquisition_pool=("an-conserve", "an-release", "an-charge", "an-burst", "an-immediate", "an-draw"),
        )
        first = generator.generate(seed=1000, episode_count=100)
        second = generator.generate(seed=1000, episode_count=100)
        self.assertEqual(first, second)
        self.assertGreater(len({sample.deck_signature for sample in first}), 20)
        split_by_signature: dict[str, DatasetSplit] = {}
        mutation_kinds = set()
        for sample in first:
            prior = split_by_signature.setdefault(sample.deck_signature, sample.split)
            self.assertEqual(sample.split, prior)
            sample.request.state.deck.validate()
            mutation_kinds.update(event.kind.value for event in sample.mutation_events)
        self.assertEqual(mutation_kinds, {"ACQUIRE", "UPGRADE", "DELETE"})

    def test_dynamic_deck_generator_uses_plan_specific_state_distributions(self) -> None:
        for plan_type, card_ids in (
            (PlanType.SENSE, ("5", "6", "7", "8", "15")),
            (PlanType.LOGIC, ("9", "10", "11", "12", "19")),
        ):
            catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=plan_type)
            generator = NiaProDynamicDeckGenerator(
                catalog.cards,
                plan_type=plan_type,
                base_card_ids=card_ids,
                acquisition_pool=card_ids,
                mutation_count_range=(0, 0),
            )
            samples = generator.generate(seed=1250, episode_count=20)
            for sample in samples:
                state = sample.request.state
                self.assertEqual(state.stance, Stance.NEUTRAL)
                self.assertEqual(state.full_power, 0)
                self.assertEqual(state.cumulative_full_power, 0)
                self.assertEqual(state.passion_buff, 0)
                if plan_type is PlanType.SENSE:
                    self.assertEqual(state.good_impression_turns, 0)
                    self.assertEqual(state.motivation, 0)
                else:
                    self.assertEqual(state.concentration, 0)
                    self.assertEqual(state.good_condition_turns, 0)
                    self.assertEqual(state.excellent_condition_turns, 0)

    def test_logic_route_classifier_distinguishes_both_subroutes(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.LOGIC)

        def state_for(card_id: str):
            card = CardRef(f"logic-route-{card_id}", card_id)
            return replace(
                self.suite.cases[0].request.state,
                deck=DeckBelief(
                    all_cards=(card,),
                    hand=(card,),
                    draw_pile=(),
                    knowledge=DeckKnowledge.KNOWN_ORDER,
                ),
                stance=Stance.NEUTRAL,
            )

        self.assertEqual(
            classify_logic_route(state_for("9"), catalog.cards),
            LogicRoute.GOOD_IMPRESSION,
        )
        self.assertEqual(
            classify_logic_route(state_for("21"), catalog.cards),
            LogicRoute.MOTIVATION_GENKI,
        )
        self.assertEqual(
            classify_logic_route(state_for("19"), catalog.cards),
            LogicRoute.MIXED,
        )

    def test_dynamic_deck_paired_evaluation_uses_true_kernel_score(self) -> None:
        generator = NiaProDynamicDeckGenerator(
            self.suite.cards,
            plan_type=self.suite.cases[0].request.plan_type,
            base_card_ids=(
                "an-conserve",
                "an-release",
                "an-charge",
                "an-burst",
                "an-immediate",
                "an-draw",
                "an-small",
            ),
            acquisition_pool=("an-conserve", "an-release", "an-charge", "an-burst", "an-immediate", "an-draw"),
        )
        samples = generator.generate(seed=2000, episode_count=120)
        candidate_metrics, candidate_results = evaluate_policy(samples, self.planner, self.kernel)
        baselines = [
            evaluate_policy(samples, ImmediateScorePolicy(self.kernel), self.kernel),
            evaluate_policy(samples, ImmediateScorePolicy(self.kernel, include_resource_potential=True), self.kernel),
        ]
        baseline_metrics, baseline_results = max(baselines, key=lambda item: (item[0].mean_final_score, item[0].p10_final_score))
        comparison = compare_paired_results(
            candidate_metrics,
            candidate_results,
            baseline_metrics,
            baseline_results,
            bootstrap_rounds=300,
        )
        self.assertEqual(candidate_metrics.illegal_action_rate, 0.0)
        self.assertEqual(candidate_metrics.stop_rate, 0.0)
        self.assertGreaterEqual(candidate_metrics.mean_final_score, baseline_metrics.mean_final_score)
        self.assertGreaterEqual(candidate_metrics.p10_final_score, baseline_metrics.p10_final_score)
        self.assertTrue(comparison.p10_guard_passed)

    def test_isolated_behavior_cloning_challenger_keeps_kernel_action_mask(self) -> None:
        generator = NiaProDynamicDeckGenerator(
            self.suite.cards,
            plan_type=self.suite.cases[0].request.plan_type,
            base_card_ids=(
                "an-conserve",
                "an-release",
                "an-charge",
                "an-burst",
                "an-immediate",
                "an-draw",
                "an-small",
            ),
            acquisition_pool=(
                "an-conserve",
                "an-release",
                "an-charge",
                "an-burst",
                "an-immediate",
                "an-draw",
            ),
        )
        samples = generator.generate(seed=2250, episode_count=16)
        collection = collect_expert_choices(samples, self.planner, self.kernel)
        self.assertEqual(collection.stopped_episode_count, 0)
        self.assertEqual(collection.illegal_expert_action_count, 0)
        self.assertGreater(len(collection.examples), 10)
        encoder = ActionFeatureEncoder(self.kernel)
        model = train_linear_behavior_cloning(
            collection.examples,
            encoder,
            epoch_count=40,
        )
        self.assertGreater(model.pair_count, len(collection.examples))
        self.assertGreater(model.training_choice_accuracy, 0.4)
        policy = LinearBehaviorCloningPolicy(self.kernel, model)
        request = samples[0].request
        response = policy.decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.DECIDE)
        assert response.action is not None
        legal = self.kernel.legal_actions(
            request.state,
            plan_type=request.plan_type,
            decision_kind=request.decision_kind,
        )
        self.assertIn(replace(response.action, source_slot=None), legal)
        assert response.trace is not None
        self.assertIn(
            "offline_challenger_no_input_authority",
            response.trace.reason_codes,
        )

        blocked = policy.decide(
            replace(
                request,
                capabilities=replace(
                    request.capabilities,
                    execution_ready=False,
                ),
            )
        )
        self.assertEqual(blocked.outcome, DecisionOutcome.STOP_TASK)
        self.assertIsNone(blocked.action)

    def test_real_anomaly_catalog_compiler_never_silently_drops_effects(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        self.assertGreater(catalog.total_rows, 100)
        self.assertEqual(catalog.supported_rows, 231)
        self.assertEqual(catalog.supported_rows, catalog.total_rows)
        self.assertEqual(len(catalog.source_sha256), 64)
        for card_id in (
            "368",
            "370",
            "372",
            "374",
            "376",
            "386",
            "390",
            "394",
            "404",
            "406",
            "410",
            "411",
            "416",
            "432",
            "433",
            "434",
            "435",
            "436",
            "437",
            "444",
            "448",
            "449",
            "440",
            "441",
            "486",
            "487",
            "476",
            "477",
            "510",
            "511",
            "512",
            "513",
            "522",
            "523",
            "527",
            "528",
            "529",
            "546",
            "547",
            "554",
            "555",
            "556",
            "557",
            "558",
            "559",
            "560",
            "561",
            "538",
            "539",
            "540",
            "541",
            "542",
            "543",
            "544",
            "545",
            "576",
            "577",
            "590",
            "591",
            "636",
            "637",
            "424",
            "425",
            "442",
            "443",
            "492",
            "493",
            "496",
            "497",
            "504",
            "505",
            "438",
            "439",
            "518",
            "519",
            "656",
            "657",
            "660",
            "661",
            "698",
            "699",
            "700",
            "701",
            "604",
            "688",
            "702",
            "703",
            "716",
            "717",
            "718",
            "719",
            "735",
            "736",
            "745",
            "746",
            "749",
            "750",
            "753",
            "754",
            "803",
            "804",
            "805",
            "806",
            "834",
            "414",
            "415",
            "594",
            "595",
            "638",
            "639",
            "839",
            "840",
            "841",
            "842",
            "412",
            "413",
            "428",
            "429",
            "430",
            "431",
            "480",
            "481",
            "612",
            "613",
            "634",
            "635",
            "767",
            "768",
            "797",
            "798",
            "855",
            "856",
            "450",
            "451",
            "494",
            "495",
            "514",
            "515",
            "548",
            "549",
            "552",
            "553",
            "737",
            "738",
            "779",
            "780",
            "788",
            "827",
            "828",
            "847",
            "848",
            "550",
            "551",
            "728",
            "773",
            "774",
            "454",
            "455",
            "582",
            "583",
            "836",
            "721",
            "727",
            "787",
            "829",
            "849",
            "850",
            "793",
            "794",
            "851",
            "852",
        ):
            self.assertIn(card_id, catalog.cards)
        self.assertEqual(catalog.unsupported, ())
        self.assertEqual(catalog.cards["604"].effects[-1].full_power_scale, 1.0)
        move_effect = next(
            effect for effect in catalog.cards["476"].effects if effect.kind is EffectKind.MOVE_RANDOM_TO_HAND
        )
        self.assertIsNotNone(move_effect.card_selector)
        assert move_effect.card_selector is not None
        self.assertEqual(move_effect.card_selector.card_id, "383")
        self.assertEqual(
            move_effect.card_selector.zones,
            (CardZone.DRAW_PILE, CardZone.DISCARD),
        )
        self.assertEqual(catalog.cards["805"].effects[-1].kind, EffectKind.EXCHANGE_HAND)
        self.assertEqual(
            tuple(effect.kind for effect in catalog.cards["793"].effects),
            (
                EffectKind.EXTRA_ACTION,
                EffectKind.DOUBLE_CARD_EFFECT,
                EffectKind.FREE_PLAY_SELECTED,
            ),
        )
        self.assertEqual(
            catalog.cards["793"].effects[-1].target_zones,
            (CardZone.DRAW_PILE, CardZone.DISCARD),
        )
        self.assertEqual(catalog.cards["793"].rarity, "SSR")
        self.assertEqual(catalog.upgrade_map["368"], "369")
        self.assertEqual(catalog.upgrade_map["793"], "794")
        self.assertEqual(len(catalog.upgrade_map), 111)
        self.assertEqual(catalog.dependency_card_ids, ("23",))
        self.assertNotIn("23", catalog.supported_card_ids)
        self.assertEqual(catalog.cards["23"].card_type, "trouble")
        self.assertTrue(catalog.cards["23"].limited)
        self.assertEqual(catalog.cards["23"].effects, ())
        add_effect = next(
            effect for effect in catalog.cards["702"].effects if effect.kind is EffectKind.ADD_CARD_TO_DECK
        )
        self.assertEqual(add_effect.card_id, "23")
        self.assertEqual(catalog.cards["410"].effects[2].kind, EffectKind.HALF_COST_TURNS)
        self.assertEqual(catalog.cards["410"].effects[2].amount, 2)
        self.assertEqual(catalog.cards["527"].effects[1].selection_count, 2)
        self.assertFalse(catalog.cards["527"].effects[1].selection_optional)
        self.assertEqual(catalog.cards["753"].effects[0].selection_count, 2)
        self.assertTrue(catalog.cards["753"].effects[0].selection_optional)
        self.assertEqual(catalog.cards["538"].effects[1].kind, EffectKind.ENTHUSIASM_BONUS_BUFF)
        self.assertEqual(catalog.cards["538"].effects[1].amount, 5)
        self.assertIsNone(catalog.cards["538"].effects[1].duration)
        self.assertEqual(
            catalog.cards["636"].effects[1].kind,
            EffectKind.ENTHUSIASM_MULTIPLIER_BUFF,
        )
        self.assertEqual(catalog.cards["636"].effects[1].duration, 3)
        self.assertEqual(
            catalog.cards["718"].effects[1].kind,
            EffectKind.FULL_POWER_CHARGE_BUFF,
        )
        self.assertEqual(catalog.cards["718"].effects[1].amount, 0.5)
        self.assertEqual(catalog.cards["803"].effects[2].duration, 2)
        self.assertEqual(catalog.cards["424"].effects[-1].kind, EffectKind.STANCE_LOCK_TURNS)
        self.assertEqual(catalog.cards["438"].effects[1].kind, EffectKind.NULLIFY_COST_CARDS)
        self.assertEqual(
            catalog.cards["518"].condition.kind,
            ConditionKind.PRESERVATION_COUNT_MIN,
        )
        self.assertEqual(
            catalog.cards["656"].condition.kind,
            ConditionKind.STANCE_CHANGE_COUNT_MIN,
        )
        self.assertEqual(
            catalog.cards["698"].effects[-1].condition.kind,
            ConditionKind.TROUBLE_CARD_COUNT_MIN,
        )
        self.assertEqual(
            catalog.cards["660"].effects[-1].schedule_phase,
            SchedulePhase.AFTER_START_OF_TURN,
        )
        self.assertEqual(
            catalog.cards["700"].effects[1].schedule_phase,
            SchedulePhase.START_OF_TURN,
        )
        self.assertIsNone(catalog.cards["442"].effects[0].schedule_limit)
        self.assertIsNone(catalog.cards["442"].effects[0].schedule_ttl)
        self.assertEqual(catalog.cards["496"].effects[-1].schedule_limit, 2)
        self.assertEqual(catalog.cards["504"].effects[0].schedule_ttl, 5)
        self.assertEqual(
            catalog.cards["540"].effects[-1].scheduled_effects[0].stance,
            Stance.LEISURE,
        )
        self.assertEqual(catalog.cards["834"].effects[-1].schedule_limit, 3)
        before_start = catalog.cards["414"].effects[-1]
        self.assertEqual(before_start.schedule_phase, SchedulePhase.BEFORE_START_OF_TURN)
        self.assertEqual(before_start.schedule_limit, 1)
        self.assertEqual(before_start.scheduled_effects[0].condition.kind, ConditionKind.ALL)
        self.assertEqual(
            tuple(condition.kind for condition in before_start.scheduled_effects[0].condition.all_of),
            (ConditionKind.PREVIOUS_STANCE_IS, ConditionKind.STANCE_IS),
        )
        end_of_turn = catalog.cards["638"].effects[-1]
        self.assertEqual(end_of_turn.schedule_phase, SchedulePhase.END_OF_TURN)
        self.assertEqual(end_of_turn.delay_turns, 0)
        self.assertEqual(
            end_of_turn.scheduled_effects[0].condition.kind,
            ConditionKind.FULL_POWER_CHARGE_MAX,
        )
        self.assertEqual(catalog.cards["841"].effects[-1].schedule_limit, 4)
        direct_growth = catalog.cards["412"].persistent_effects[0]
        self.assertEqual(direct_growth.schedule_phase, SchedulePhase.STANCE_CHANGED)
        self.assertIsNone(direct_growth.schedule_limit)
        self.assertEqual(
            direct_growth.scheduled_effects[0].condition.kind,
            ConditionKind.IS_DIRECT_EFFECT,
        )
        self.assertTrue(direct_growth.scheduled_effects[0].card_selector.source_only)
        self.assertEqual(catalog.cards["428"].persistent_effects[0].schedule_limit, 1)
        modulo_growth = catalog.cards["855"].persistent_effects[0].scheduled_effects[0]
        self.assertEqual(modulo_growth.condition.kind, ConditionKind.ALL)
        self.assertEqual(modulo_growth.condition.all_of[-1].modulus, 3)
        self.assertEqual(modulo_growth.condition.all_of[-1].remainder, 2)
        card_used = catalog.cards["548"].effects[-1]
        self.assertEqual(card_used.schedule_phase, SchedulePhase.CARD_USED)
        self.assertEqual(
            card_used.scheduled_effects[0].condition.kind,
            ConditionKind.USED_CARD_EFFECT_TAG,
        )
        self.assertEqual(card_used.scheduled_effects[0].condition.value, "strength")
        filtered = catalog.cards["514"].effects[-1]
        self.assertEqual(filtered.trigger_card_type, "mental")
        after_card = catalog.cards["450"].persistent_effects[0]
        self.assertEqual(after_card.schedule_phase, SchedulePhase.AFTER_CARD_USED)
        self.assertEqual(after_card.schedule_limit, 2)
        self.assertEqual(catalog.cards["780"].base_card_id, "779")
        trend_growth = catalog.cards["550"].persistent_effects[0].scheduled_effects[0]
        self.assertEqual(trend_growth.card_modification.typed_cost_delta, -1)
        entertainer_growth = catalog.cards["773"].persistent_effects[0].scheduled_effects[0]
        self.assertEqual(entertainer_growth.card_modification.typed_cost_delta, 1)
        score_buff = catalog.cards["728"].effects[-1].scheduled_effects[0]
        self.assertEqual(score_buff.kind, EffectKind.SCORE_BUFF)
        self.assertEqual(score_buff.amount, 1.5)
        self.assertEqual(score_buff.duration, 1)
        self.assertEqual(catalog.cards["454"].upgrade_card_id, "455")
        upgrade_hand = catalog.cards["454"].effects[-1].scheduled_effects[0]
        self.assertEqual(upgrade_hand.kind, EffectKind.UPGRADE_HAND)
        self.assertEqual(
            catalog.cards["582"].effects[-1].schedule_phase,
            SchedulePhase.FULL_POWER_CHARGE_INCREASED,
        )
        self.assertEqual(
            catalog.cards["836"].effects[0].schedule_phase,
            SchedulePhase.BEFORE_CARD_USED,
        )
        self.assertEqual(
            catalog.cards["721"].persistent_effects[-1].schedule_phase,
            SchedulePhase.CARD_MOVED_TO_HAND,
        )
        self.assertEqual(
            catalog.cards["727"].effects[1].kind,
            EffectKind.FREE_PLAY_ALL,
        )
        self.assertEqual(
            catalog.cards["787"].effects[-1].kind,
            EffectKind.FIXED_STAMINA_TO_ZERO,
        )
        self.assertEqual(
            catalog.cards["829"].effects[-1].scheduled_effects[0].kind,
            EffectKind.PAID_PLAY_SELECTED,
        )
        self.assertEqual(
            catalog.cards["849"].effects[0].score_enthusiasm_multiplier,
            2.0,
        )
        self.assertEqual(
            catalog.cards["849"].persistent_effects[0].scheduled_effects[0].kind,
            EffectKind.FREE_PLAY_RANDOM,
        )

    def test_sense_and_logic_foundation_catalogs_are_atomic_and_typed(self) -> None:
        sense = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.SENSE)
        logic = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.LOGIC)
        self.assertEqual((sense.supported_rows, sense.total_rows), (228, 266))
        self.assertEqual((logic.supported_rows, logic.total_rows), (216, 294))
        self.assertEqual(len(sense.upgrade_map), 111)
        self.assertEqual(len(logic.upgrade_map), 106)
        self.assertEqual(sense.cards["80"].cost_kind, CardCostKind.STAMINA)
        self.assertEqual(sense.cards["86"].cost_kind, CardCostKind.CONCENTRATION)
        self.assertEqual(
            sense.cards["116"].cost_kind,
            CardCostKind.GOOD_CONDITION,
        )
        self.assertEqual(logic.cards["96"].cost_kind, CardCostKind.MOTIVATION)
        self.assertEqual(
            logic.cards["138"].cost_kind,
            CardCostKind.GOOD_IMPRESSION,
        )
        unsupported = {card.card_id: card for card in logic.unsupported}
        self.assertIn("831", unsupported)
        self.assertTrue(
            any(
                reason.startswith("generated_dependency_not_compiled:832:")
                for reason in unsupported["831"].reasons
            )
        )

    def test_drink_catalog_is_atomic_plan_scoped_and_typed(self) -> None:
        for plan_type in PlanType:
            with self.subTest(plan_type=plan_type):
                catalog = compile_drink_catalog(P_DRINKS, plan_type=plan_type)
                self.assertEqual((catalog.supported_rows, catalog.total_rows), (15, 16))
                self.assertEqual({drink.drink_id for drink in catalog.unsupported}, {"19"})
                self.assertIn(
                    "action_not_compiled:addRandomUpgradedSSRCardToHand",
                    catalog.unsupported[0].reasons,
                )
                self.assertEqual(len(catalog.source_sha256), 64)
                for definition in catalog.drinks.values():
                    definition.validate()
                    self.assertIn(definition.plan, {None, plan_type})

        sense = compile_drink_catalog(P_DRINKS, plan_type=PlanType.SENSE)
        move_random = sense.drinks["13"].effects[0]
        self.assertEqual(move_random.kind, EffectKind.MOVE_RANDOM_TO_HAND)
        self.assertIsNotNone(move_random.card_selector)
        assert move_random.card_selector is not None
        self.assertEqual(move_random.card_selector.source_type, "pIdol")
        self.assertEqual(sense.drinks["22"].effects[0].kind, EffectKind.MOVE_SELECTED_TO_HAND)
        self.assertEqual(sense.drinks["22"].effects[-1].kind, EffectKind.DOUBLE_COST_TURNS)

    def test_drink_source_selector_and_anomaly_hold_continuation(self) -> None:
        cards = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        drinks = compile_drink_catalog(P_DRINKS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(cards.cards, drinks.drinks)
        p_idol = CardRef("drink-p-idol", "446")
        produce = CardRef("drink-produce", "380")
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            deck=DeckBelief(
                all_cards=(p_idol, produce),
                hand=(),
                draw_pile=(p_idol,),
                discard=(produce,),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
            stance=Stance.NEUTRAL,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
            pending_choice=None,
            resolution_stack=(),
            scheduled_effects=(),
            persistent_effects_initialized=False,
            half_cost_turns=0,
            half_cost_fresh=False,
            drinks=("13",),
        )
        legal = kernel.legal_actions(
            state,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.DRINK,
        )
        self.assertEqual(legal, (GameAction(ActionKind.USE_DRINK, drink_id="13"),))
        used = kernel.transition(
            state,
            GameAction(ActionKind.USE_DRINK, drink_id="13"),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual([card.instance_id for card in used.deck.hand], [p_idol.instance_id])
        self.assertEqual([card.instance_id for card in used.deck.discard], [produce.instance_id])
        self.assertEqual(used.half_cost_turns, 3)
        self.assertTrue(used.half_cost_fresh)
        self.assertEqual(used.drinks, ())

        held_candidate = CardRef("drink-hold", "370")
        hold_state = replace(
            state,
            deck=DeckBelief(
                all_cards=(held_candidate,),
                hand=(),
                draw_pile=(),
                discard=(held_candidate,),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
            drinks=("17",),
        )
        requested = kernel.transition(
            hold_state,
            GameAction(ActionKind.USE_DRINK, drink_id="17"),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNotNone(requested.pending_choice)
        assert requested.pending_choice is not None
        self.assertEqual(requested.pending_choice.action_kind, ActionKind.HOLD_CARD)
        self.assertEqual(requested.pending_choice.source_id, "drink:17")
        self.assertEqual(requested.drinks, ())
        resolved = kernel.transition(
            requested,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=held_candidate.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNone(resolved.pending_choice)
        self.assertEqual([card.instance_id for card in resolved.deck.held], [held_candidate.instance_id])
        self.assertEqual(resolved.stance, Stance.CONSERVE_2)

    def test_sense_drink_move_selected_double_cost_and_schedule(self) -> None:
        cards = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.SENSE)
        drinks = compile_drink_catalog(P_DRINKS, plan_type=PlanType.SENSE)
        kernel = PythonNiaProKernel(cards.cards, drinks.drinks)
        candidate = CardRef("drink-move", "5")
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=4,
            deck=DeckBelief(
                all_cards=(candidate,),
                hand=(),
                draw_pile=(),
                discard=(candidate,),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
            stance=Stance.NEUTRAL,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
            hand_limit=0,
            pending_choice=None,
            resolution_stack=(),
            scheduled_effects=(),
            persistent_effects_initialized=False,
            double_cost_turns=0,
            double_cost_fresh=False,
            nullify_cost_cards=0,
            drinks=("22",),
        )
        requested = kernel.transition(
            state,
            GameAction(ActionKind.USE_DRINK, drink_id="22"),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertIsNotNone(requested.pending_choice)
        assert requested.pending_choice is not None
        self.assertEqual(requested.pending_choice.action_kind, ActionKind.MOVE_CARD)
        self.assertEqual(requested.pending_choice.source_id, "drink:22")
        resolved = kernel.transition(
            requested,
            GameAction(ActionKind.MOVE_CARD, card_instance_id=candidate.instance_id),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual([card.instance_id for card in resolved.deck.hand], [candidate.instance_id])
        self.assertEqual(resolved.nullify_cost_cards, 2)
        self.assertEqual(resolved.stamina, 18)
        self.assertEqual(resolved.double_cost_turns, 1)
        self.assertTrue(resolved.double_cost_fresh)

        first_end = kernel.transition(
            resolved,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual(first_end.double_cost_turns, 1)
        self.assertFalse(first_end.double_cost_fresh)
        second_end = kernel.transition(
            first_end,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual(second_end.double_cost_turns, 0)

        schedule_state = replace(
            state,
            drinks=("20",),
            concentration=0,
        )
        scheduled = kernel.transition(
            schedule_state,
            GameAction(ActionKind.USE_DRINK, drink_id="20"),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual(len(scheduled.scheduled_effects), 1)
        self.assertEqual(scheduled.scheduled_effects[0].source_id, "drink:20")
        self.assertEqual(scheduled.scheduled_effects[0].source_kind, EffectSourceKind.DRINK)
        ticked = kernel.transition(
            scheduled,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual(ticked.concentration, 1)
        self.assertEqual(len(ticked.scheduled_effects), 1)

    def test_logic_drink_card_rarity_condition_uses_typed_trigger_context(self) -> None:
        cards = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.LOGIC)
        drinks = compile_drink_catalog(P_DRINKS, plan_type=PlanType.LOGIC)
        kernel = PythonNiaProKernel(cards.cards, drinks.drinks)
        active = CardRef("drink-rarity-active", "9")
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            deck=DeckBelief(
                all_cards=(active,),
                hand=(active,),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
            stance=Stance.NEUTRAL,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
            pending_choice=None,
            resolution_stack=(),
            scheduled_effects=(),
            persistent_effects_initialized=False,
            double_card_effects_remaining=0,
            double_cost_turns=0,
            double_cost_fresh=False,
            drinks=("25",),
        )
        prepared = kernel.transition(
            state,
            GameAction(ActionKind.USE_DRINK, drink_id="25"),
            plan_type=PlanType.LOGIC,
        )[0].state
        self.assertEqual(prepared.stamina, 18)
        self.assertEqual(prepared.double_cost_turns, 1)
        self.assertEqual(prepared.scheduled_effects[0].source_kind, EffectSourceKind.DRINK)
        played = kernel.transition(
            prepared,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=active.instance_id),
            plan_type=PlanType.LOGIC,
        )[0].state
        self.assertEqual(played.stamina, 8)
        self.assertEqual(played.double_card_effects_remaining, 1)
        self.assertEqual(played.scheduled_effects, ())

    def test_item_catalog_is_atomic_plan_scoped_and_persistent(self) -> None:
        expected = {
            PlanType.SENSE: (74, 140),
            PlanType.LOGIC: (69, 159),
            PlanType.ANOMALY: (55, 105),
        }
        for plan_type, counts in expected.items():
            with self.subTest(plan_type=plan_type):
                catalog = compile_item_catalog(P_ITEMS, plan_type=plan_type)
                self.assertEqual((catalog.supported_rows, catalog.total_rows), counts)
                self.assertEqual(
                    catalog.supported_rows + len(catalog.unsupported),
                    catalog.total_rows,
                )
                self.assertEqual(len(catalog.source_sha256), 64)
                for definition in catalog.items.values():
                    definition.validate()
                    self.assertIn(definition.plan, {None, plan_type})
                    self.assertTrue(
                        all(
                            effect.kind is EffectKind.SCHEDULE
                            for effect in definition.persistent_effects
                        )
                    )

        sense = compile_item_catalog(P_ITEMS, plan_type=PlanType.SENSE)
        self.assertIn("3", sense.items)
        self.assertEqual(sense.items["3"].persistent_effects[0].schedule_limit, 1)
        self.assertTrue(
            any(
                "effect_not_compiled:" in reason
                for item in sense.unsupported
                for reason in item.reasons
            )
        )

    def test_item_effect_materializes_with_typed_source_and_fires(self) -> None:
        cards = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.SENSE)
        items = compile_item_catalog(P_ITEMS, plan_type=PlanType.SENSE)
        kernel = PythonNiaProKernel(cards.cards, items=items.items)
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            deck=DeckBelief(
                all_cards=(),
                hand=(),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
            concentration=5,
            actions_remaining=1,
            hand_limit=0,
            pending_choice=None,
            resolution_stack=(),
            scheduled_effects=(),
            persistent_effects_initialized=False,
            items=("3",),
            field_effects=(),
        )
        advanced = kernel.transition(
            state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.SENSE,
        )[0]
        self.assertEqual(advanced.state.concentration, 7)
        self.assertEqual(advanced.state.actions_remaining, 2)
        self.assertEqual(len(advanced.state.scheduled_effects), 1)
        self.assertEqual(advanced.state.scheduled_effects[0].remaining_uses, 0)
        self.assertEqual(advanced.state.scheduled_effects[0].source_kind, EffectSourceKind.ITEM)
        self.assertIn(
            "item:3",
            {event.source_id for event in advanced.events},
        )

        recurring = replace(state, concentration=0, items=("19",))
        first = kernel.transition(
            recurring,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual(first.concentration, 1)
        self.assertEqual(len(first.scheduled_effects), 1)
        self.assertEqual(first.scheduled_effects[0].source_id, "item:19")
        self.assertEqual(first.scheduled_effects[0].source_kind, EffectSourceKind.ITEM)

    def test_item_plan_unknown_inventory_and_field_effects_fail_closed(self) -> None:
        cards = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        sense_items = compile_item_catalog(P_ITEMS, plan_type=PlanType.SENSE)
        kernel = PythonNiaProKernel(cards.cards, items=sense_items.items)
        base = replace(
            self.suite.cases[0].request.state,
            deck=DeckBelief(
                all_cards=(),
                hand=(),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
            items=("3",),
            field_effects=(),
        )
        with self.assertRaisesRegex(KernelError, "item_plan_mismatch_in_inventory:3"):
            kernel.legal_actions(
                base,
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.CARD,
            )
        with self.assertRaisesRegex(KernelError, "unregistered_item_in_inventory:missing"):
            kernel.legal_actions(
                replace(base, items=("missing",)),
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.CARD,
            )
        with self.assertRaisesRegex(KernelError, "unsupported_field_effects"):
            kernel.legal_actions(
                replace(base, items=(), field_effects=("unmodeled-field",)),
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.CARD,
            )

        unknown_drink_schedule = ScheduledEffectRef(
            source_id="drink:missing",
            card_id="missing",
            definition_effect_index=0,
            source_kind=EffectSourceKind.DRINK,
        )
        with self.assertRaisesRegex(KernelError, "unregistered_drink_schedule:missing"):
            kernel.legal_actions(
                replace(
                    base,
                    items=(),
                    scheduled_effects=(unknown_drink_schedule,),
                ),
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.CARD,
            )

        sense_drinks = compile_drink_catalog(P_DRINKS, plan_type=PlanType.SENSE)
        mixed_kernel = PythonNiaProKernel(
            cards.cards,
            drinks=sense_drinks.drinks,
        )
        with self.assertRaisesRegex(KernelError, "drink_plan_mismatch_in_schedule:20"):
            mixed_kernel.legal_actions(
                replace(
                    base,
                    items=(),
                    scheduled_effects=(
                        replace(
                            unknown_drink_schedule,
                            source_id="drink:20",
                            card_id="20",
                        ),
                    ),
                ),
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.CARD,
            )

        with self.assertRaisesRegex(
            HighScoreContractError,
            "scheduled drink effects cannot be persistent",
        ):
            replace(unknown_drink_schedule, persistent=True).validate()
        with self.assertRaisesRegex(
            HighScoreContractError,
            "scheduled item effects must be persistent",
        ):
            ScheduledEffectRef(
                source_id="item:3",
                card_id="3",
                definition_effect_index=0,
                source_kind=EffectSourceKind.ITEM,
            ).validate()

    def test_plan_specific_resources_reject_negative_observations(self) -> None:
        state = self.suite.cases[0].request.state
        for field_name in (
            "concentration",
            "good_condition_turns",
            "excellent_condition_turns",
            "good_impression_turns",
            "motivation",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(HighScoreContractError):
                    replace(state, **{field_name: -1}).validate()

    def test_sense_score_uses_concentration_good_and_perfect_condition(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.SENSE)
        kernel = PythonNiaProKernel(catalog.cards)
        challenge = CardRef("sense-challenge", "5")
        deck = DeckBelief(
            all_cards=(challenge,),
            hand=(challenge,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            concentration=3,
            good_condition_turns=2,
            excellent_condition_turns=1,
            score_multiplier=1.0,
            parameter_gain_bonus_pct=0,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=challenge.instance_id),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual(played.score, 48)
        self.assertEqual(played.stamina, 23)

        focus = CardRef("sense-focus", "34")
        focus_state = replace(
            state,
            deck=DeckBelief(
                all_cards=(focus,),
                hand=(focus,),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_ORDER,
            ),
            concentration=4,
            good_condition_turns=0,
            excellent_condition_turns=0,
        )
        focused = kernel.transition(
            focus_state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=focus.instance_id),
            plan_type=PlanType.SENSE,
        )[0].state
        self.assertEqual(focused.score, 23)

    def test_logic_routes_apply_motivation_genki_and_declared_cost_resource(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.LOGIC)
        kernel = PythonNiaProKernel(catalog.cards)
        motivation_card = CardRef("logic-motivation", "21")
        motivation_state = replace(
            self.suite.cases[0].request.state,
            deck=DeckBelief(
                all_cards=(motivation_card,),
                hand=(motivation_card,),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_ORDER,
            ),
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            motivation=2,
            score_multiplier=1.0,
            parameter_gain_bonus_pct=0,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        motivated = kernel.transition(
            motivation_state,
            GameAction(
                ActionKind.PLAY_CARD,
                card_instance_id=motivation_card.instance_id,
            ),
            plan_type=PlanType.LOGIC,
        )[0].state
        self.assertEqual(motivated.genki, 3)
        self.assertEqual(motivated.motivation, 4)

        genki_score = CardRef("logic-genki-score", "11")
        genki_state = replace(
            motivation_state,
            deck=DeckBelief(
                all_cards=(genki_score,),
                hand=(genki_score,),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_ORDER,
            ),
            stamina=10,
            genki=5,
            motivation=0,
        )
        scored = kernel.transition(
            genki_state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=genki_score.instance_id),
            plan_type=PlanType.LOGIC,
        )[0]
        self.assertEqual(scored.state.score, 5)
        self.assertEqual(scored.state.stamina, 5)
        self.assertEqual(scored.state.genki, 5)
        self.assertNotIn(
            "COST_GENKI",
            [event.event_type for event in scored.events],
        )

    def test_scheduled_upgrade_hand_rewrites_stable_instances_after_refill(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("upgrade-hand-source", "454")
        target = CardRef("upgrade-hand-target", "368")
        deck = DeckBelief(
            all_cards=(source, target),
            hand=(source, target),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=1,
            turns_remaining=2,
            stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )

        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertTrue(
            any(
                effect.trigger_phase is SchedulePhase.TURN
                for effect in played.scheduled_effects
            )
        )
        next_turn = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        upgraded = next(
            card
            for card in next_turn.state.deck.all_cards
            if card.instance_id == target.instance_id
        )
        self.assertEqual(upgraded.card_id, "369")
        self.assertTrue(upgraded.upgraded)
        self.assertEqual(next_turn.state.deck.hand, (upgraded,))
        self.assertIn("UPGRADE_HAND", [event.event_type for event in next_turn.events])

    def test_direct_full_power_increase_triggers_nonrecursive_bonus(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("charge-watch-source", "582")
        trigger = CardRef("charge-watch-trigger", "374")
        deck = DeckBelief(
            all_cards=(source, trigger),
            hand=(source, trigger),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            genki=0,
            full_power=0,
            cumulative_full_power=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )

        installed = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(installed.stance, Stance.CONSERVE_2)
        branch = kernel.transition(
            replace(installed, actions_remaining=1),
            GameAction(ActionKind.PLAY_CARD, card_instance_id=trigger.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.full_power, 3)
        self.assertEqual(branch.state.cumulative_full_power, 3)
        self.assertEqual(
            [event.event_type for event in branch.events].count("TRIGGER_SCHEDULED"),
            1,
        )
        self.assertEqual(
            [event.event_type for event in branch.events].count("FULL_POWER"),
            2,
        )

    def test_before_card_used_nullifies_payment_after_original_legality_gate(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("before-card-source", "836")
        target = CardRef("before-card-target", "576")
        deck = DeckBelief(
            all_cards=(source, target),
            hand=(source, target),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=2,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            stance=Stance.AGGRESSIVE_1,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            nullify_cost_cards=0,
        )

        installed = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        branch = kernel.transition(
            installed,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.stamina, 2)
        self.assertEqual(branch.state.nullify_cost_cards, 0)
        before_ref = next(
            effect
            for effect in branch.state.scheduled_effects
            if effect.trigger_phase is SchedulePhase.BEFORE_CARD_USED
        )
        self.assertEqual(before_ref.remaining_uses, 4)
        event_types = [event.event_type for event in branch.events]
        self.assertLess(event_types.index("TRIGGER_SCHEDULED"), event_types.index("COST_STAMINA"))
        self.assertLess(event_types.index("NULLIFY_COST_CARDS"), event_types.index("NULLIFY_CARD_COST"))

    def test_card_moved_to_hand_events_draw_and_resume_turn_after_hold(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        nova = CardRef("move-nova", "721")
        filler = CardRef("move-filler", "368")
        nova_deck = DeckBelief(
            all_cards=(nova, filler),
            hand=(),
            draw_pile=(nova, filler),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        nova_state = replace(
            self.suite.cases[0].request.state,
            deck=nova_deck,
            turns_remaining=2,
            hand_limit=1,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        nova_branch = kernel.transition(
            nova_state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(
            {card.instance_id for card in nova_branch.state.deck.hand},
            {nova.instance_id, filler.instance_id},
        )
        self.assertEqual(
            [event.event_type for event in nova_branch.events].count("DRAW"),
            2,
        )

        expert = CardRef("move-expert", "727")
        hold_target = CardRef("move-hold-target", "368")
        expert_deck = DeckBelief(
            all_cards=(expert, hold_target),
            hand=(),
            draw_pile=(expert,),
            discard=(hold_target,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        expert_state = replace(
            nova_state,
            deck=expert_deck,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        pending = kernel.transition(
            expert_state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert pending.pending_choice is not None
        self.assertEqual(pending.pending_choice.action_kind, ActionKind.HOLD_CARD)
        self.assertTrue(pending.resolution_stack)
        serialized = replace(
            self.suite.cases[0].request,
            decision_kind=DecisionKind.SECONDARY,
            state=pending,
            card_slots=tuple(
                (instance_id, index)
                for index, instance_id in enumerate(
                    pending.pending_choice.candidate_instance_ids
                )
            ),
        ).to_dict()
        self.assertTrue(
            all(
                "continuation_kind" in continuation
                for continuation in serialized["state"]["resolution_stack"]
            )
        )
        resolved = kernel.transition(
            pending,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=hold_target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNone(resolved.pending_choice)
        self.assertEqual(resolved.resolution_stack, ())
        self.assertEqual(resolved.deck.held, (hold_target,))
        self.assertIn(expert, resolved.deck.hand)

        empty_deck = DeckBelief(
            all_cards=(expert,),
            hand=(),
            draw_pile=(expert,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        empty_state = replace(
            expert_state,
            deck=empty_deck,
            hand_limit=1,
        )
        empty_branch = kernel.transition(
            empty_state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertIsNone(empty_branch.state.pending_choice)
        self.assertEqual(empty_branch.state.resolution_stack, ())
        self.assertIn(
            "HOLD_SELECTED_EMPTY",
            [event.event_type for event in empty_branch.events],
        )

    def test_exchange_hand_keeps_mixed_secondary_chance_branches_resumable(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("exchange-source", "805")
        remainder = CardRef("exchange-remainder", "368")
        expert = CardRef("exchange-expert", "727")
        filler = CardRef("exchange-filler", "382")
        deck = DeckBelief(
            all_cards=(source, remainder, expert, filler),
            hand=(source, remainder),
            draw_pile=(expert, filler),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            turns_remaining=2,
            hand_limit=2,
            actions_remaining=1,
            stance=Stance.CONSERVE_1,
            passion=0,
            parameter_gain_bonus_pct=0,
            score_multiplier=1.0,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        branches = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )
        self.assertAlmostEqual(sum(branch.probability for branch in branches), 1.0)
        self.assertEqual(
            {branch.state.pending_choice is not None for branch in branches},
            {False, True},
        )
        pending_branch = next(
            branch for branch in branches if branch.state.pending_choice is not None
        )
        pending_choice = pending_branch.state.pending_choice
        assert pending_choice is not None
        held_id = pending_choice.candidate_instance_ids[0]
        resolved = kernel.transition(
            pending_branch.state,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=held_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNone(resolved.pending_choice)
        self.assertEqual(resolved.resolution_stack, ())
        self.assertIn(held_id, {card.instance_id for card in resolved.deck.held})

    def test_expert_free_plays_all_held_then_resumes_actions(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        expert = CardRef("expert-source", "727")
        first = CardRef("expert-first", "368")
        second = CardRef("expert-second", "382")
        deck = DeckBelief(
            all_cards=(expert, first, second),
            hand=(expert,),
            draw_pile=(),
            held=(first, second),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            turns_remaining=2,
            hand_limit=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            strength_times=2,
            preservation_times=2,
            passion=0,
            parameter_gain_bonus_pct=0,
            score_multiplier=1.0,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        first_pending = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=expert.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert first_pending.pending_choice is not None
        self.assertEqual(first_pending.pending_choice.candidate_instance_ids, (first.instance_id,))
        second_pending = kernel.transition(
            first_pending,
            GameAction(ActionKind.FREE_PLAY, card_instance_id=first.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert second_pending.pending_choice is not None
        self.assertEqual(second_pending.pending_choice.candidate_instance_ids, (second.instance_id,))
        resolved = kernel.transition(
            second_pending,
            GameAction(ActionKind.FREE_PLAY, card_instance_id=second.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNone(resolved.pending_choice)
        self.assertEqual(resolved.resolution_stack, ())
        self.assertEqual(resolved.deck.held, ())
        self.assertEqual(resolved.actions_remaining, 1)
        self.assertEqual(resolved.score, 35)
        self.assertTrue(
            any(
                effect.trigger_phase is SchedulePhase.TURN
                for effect in resolved.scheduled_effects
            )
        )

    def test_grand_finale_hold_growth_repeats_score_and_zeroes_stamina(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        finale = CardRef("finale-source", "787")
        chooser = CardRef("finale-chooser", "368")
        deck = DeckBelief(
            all_cards=(finale, chooser),
            hand=(finale,),
            draw_pile=(chooser,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            pending_choice=PendingChoice(
                ActionKind.HOLD_CARD,
                chooser.instance_id,
                (finale.instance_id,),
            ),
            scheduled_effects=(),
            persistent_effects_initialized=False,
            stance=Stance.NEUTRAL,
        )
        held_state = kernel.transition(
            state,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=finale.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        grown = held_state.deck.held[0]
        self.assertEqual(grown.score_repeat_bonus, 1)
        playable = replace(
            held_state,
            deck=held_state.deck.replace_zones(hand=(grown,), held=()),
            score=0,
            stamina=10,
            max_stamina=10,
            genki=0,
            actions_remaining=1,
            cumulative_full_power=2,
            passion=0,
            score_multiplier=1.0,
            parameter_gain_bonus_pct=0,
        )
        branch = kernel.transition(
            playable,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=finale.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.score, 114)
        self.assertEqual(branch.state.stamina, 0)
        self.assertEqual(
            [event.event_type for event in branch.events].count("SCORE"),
            2,
        )

    def test_start_turn_paid_selection_resumes_full_power_pipeline(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        star = CardRef("paid-star", "829")
        target = CardRef("paid-target", "382")
        deck = DeckBelief(
            all_cards=(star, target),
            hand=(star, target),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            turns_remaining=3,
            hand_limit=1,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            full_power=5,
            cumulative_full_power=5,
            passion=0,
            score_multiplier=1.0,
            parameter_gain_bonus_pct=0,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        installed = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=star.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        pending = kernel.transition(
            installed,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert pending.pending_choice is not None
        self.assertEqual(pending.pending_choice.action_kind, ActionKind.PAID_PLAY)
        self.assertEqual(pending.stance, Stance.FULL_POWER)
        resolved = kernel.transition(
            pending,
            GameAction(ActionKind.PAID_PLAY, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertIsNone(resolved.state.pending_choice)
        self.assertEqual(resolved.state.resolution_stack, ())
        self.assertEqual(resolved.state.actions_remaining, 2)
        self.assertEqual(resolved.state.score, 27)
        self.assertLess(
            [event.event_type for event in resolved.events].index("PAID_PLAY"),
            [event.event_type for event in resolved.events].index("FULL_POWER_READY"),
        )

        unusable = CardRef("paid-unusable", "721")
        unusable_deck = DeckBelief(
            all_cards=(star, unusable),
            hand=(star, unusable),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        unusable_state = replace(
            state,
            deck=unusable_deck,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        unusable_installed = kernel.transition(
            unusable_state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=star.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        unusable_pending = kernel.transition(
            unusable_installed,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        discarded = kernel.transition(
            unusable_pending,
            GameAction(ActionKind.PAID_PLAY, card_instance_id=unusable.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        discarded_event_types = [event.event_type for event in discarded.events]
        self.assertEqual(discarded.state.score, 0)
        self.assertIn("PAID_PLAY_UNUSABLE_DISCARD", discarded_event_types)
        self.assertLess(
            discarded_event_types.index("PAID_PLAY_UNUSABLE_DISCARD"),
            discarded_event_types.index("DRAW"),
        )

    def test_removed_reach_card_replays_with_action_enthusiasm_multiplier(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("reach-source", "849")
        deck = DeckBelief(
            all_cards=(source,),
            hand=(source,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            turns_remaining=3,
            hand_limit=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            preservation_times=2,
            passion=10,
            score_multiplier=1.0,
            parameter_gain_bonus_pct=0,
            scheduled_effects=(),
            persistent_effects_initialized=False,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(played.score, 21)
        self.assertEqual(played.genki, 1)
        self.assertIn(source, played.deck.removed)
        branch = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.score, 42)
        self.assertEqual(branch.state.genki, 2)
        self.assertIn(source, branch.state.deck.removed)
        persistent = next(
            effect
            for effect in branch.state.scheduled_effects
            if effect.trigger_phase is SchedulePhase.END_OF_TURN
        )
        self.assertEqual(persistent.remaining_uses, 2)
        self.assertIn("FREE_PLAY_RANDOM", [event.event_type for event in branch.events])

    def test_typed_cost_growth_applies_to_the_declared_cost_resource(self) -> None:
        stamina_definition = CardDefinition(
            "typed-stamina",
            "Typed stamina",
            PlanType.ANOMALY,
            stamina_cost=5,
        )
        full_power_definition = CardDefinition(
            "typed-full-power",
            "Typed full power",
            PlanType.ANOMALY,
            full_power_cost=2,
        )
        kernel = PythonNiaProKernel(
            {
                stamina_definition.card_id: stamina_definition,
                full_power_definition.card_id: full_power_definition,
            }
        )
        stamina_card = CardRef(
            "typed-stamina-instance",
            stamina_definition.card_id,
            typed_cost_delta=1,
        )
        full_power_card = CardRef(
            "typed-full-power-instance",
            full_power_definition.card_id,
            typed_cost_delta=-1,
        )
        deck = DeckBelief(
            all_cards=(stamina_card, full_power_card),
            hand=(stamina_card, full_power_card),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=10,
            max_stamina=10,
            genki=0,
            full_power=1,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
        )

        legal_ids = {
            action.card_instance_id
            for action in kernel.legal_actions(
                state,
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.CARD,
            )
            if action.kind is ActionKind.PLAY_CARD
        }
        self.assertEqual(
            legal_ids,
            {stamina_card.instance_id, full_power_card.instance_id},
        )
        stamina_branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=stamina_card.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(stamina_branch.state.stamina, 4)
        full_power_branch = kernel.transition(
            state,
            GameAction(
                ActionKind.PLAY_CARD,
                card_instance_id=full_power_card.instance_id,
            ),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(full_power_branch.state.full_power, 0)

    def test_score_buff_multiplies_post_stance_score_and_expires_by_turn(self) -> None:
        source = CardRef("score-buff-instance", "score-buff-card")
        definition = CardDefinition(
            source.card_id,
            "Score buff",
            PlanType.ANOMALY,
            effects=(
                EffectSpec(EffectKind.SCORE_BUFF, amount=1.5, duration=1),
                EffectSpec(EffectKind.SCORE, amount=10),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        state = replace(
            self.suite.cases[0].request.state,
            deck=DeckBelief(
                all_cards=(source,),
                hand=(source,),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_ORDER,
            ),
            score=0,
            stamina=10,
            max_stamina=10,
            actions_remaining=1,
            turns_remaining=3,
            stance=Stance.AGGRESSIVE_1,
            score_buffs=(),
        )

        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(played.score, 50)
        self.assertEqual(
            played.score_buffs,
            (TimedModifier(amount=1.5, remaining_turns=1, fresh=True),),
        )
        first_end = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertFalse(first_end.score_buffs[0].fresh)
        second_end = kernel.transition(
            first_end,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(second_end.score_buffs, ())

    def test_persistent_direct_stance_growth_tracks_source_instance(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("growth-source", "412")
        trigger = CardRef("stance-trigger", "368")
        deck = DeckBelief(
            all_cards=(source, trigger),
            hand=(trigger,),
            draw_pile=(source,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.NEUTRAL,
            previous_stance=Stance.NEUTRAL,
            actions_remaining=1,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            stance_changed_by_direct_effect_times=0,
        )

        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=trigger.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]

        grown = next(card for card in branch.state.deck.all_cards if card.instance_id == source.instance_id)
        self.assertEqual(grown.score_bonus, 5)
        self.assertTrue(branch.state.persistent_effects_initialized)
        self.assertEqual(branch.state.stance_changed_by_direct_effect_times, 1)
        self.assertEqual(len(branch.state.scheduled_effects), 1)
        self.assertTrue(branch.state.scheduled_effects[0].persistent)
        self.assertIsNone(branch.state.scheduled_effects[0].remaining_uses)
        event_types = [event.event_type for event in branch.events]
        self.assertLess(event_types.index("STANCE"), event_types.index("TRIGGER_SCHEDULED"))
        self.assertLess(event_types.index("TRIGGER_SCHEDULED"), event_types.index("CARD_MODIFICATION"))

    def test_persistent_full_power_growth_exhausts_per_instance_limit(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("full-power-growth", "428")
        deck = DeckBelief(
            all_cards=(source,),
            hand=(),
            draw_pile=(source,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            turns_remaining=2,
            hand_limit=0,
            stance=Stance.NEUTRAL,
            previous_stance=Stance.NEUTRAL,
            full_power=10,
            cumulative_full_power=10,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            full_power_times=0,
            stance_changed_by_direct_effect_times=0,
        )

        branch = kernel.transition(
            state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]

        grown = next(card for card in branch.state.deck.all_cards if card.instance_id == source.instance_id)
        self.assertEqual(grown.score_bonus, 20)
        self.assertEqual(branch.state.stance, Stance.FULL_POWER)
        self.assertEqual(branch.state.full_power_times, 1)
        self.assertEqual(branch.state.stance_changed_by_direct_effect_times, 0)
        self.assertEqual(len(branch.state.scheduled_effects), 1)
        self.assertEqual(branch.state.scheduled_effects[0].remaining_uses, 0)
        self.assertTrue(branch.state.scheduled_effects[0].persistent)

    def test_persistent_modulo_growth_observes_preincrement_counter(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("modulo-growth", "855")
        trigger = CardRef("modulo-trigger", "368")
        deck = DeckBelief(
            all_cards=(source, trigger),
            hand=(trigger,),
            draw_pile=(source,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.NEUTRAL,
            previous_stance=Stance.NEUTRAL,
            actions_remaining=1,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            stance_changed_by_direct_effect_times=2,
        )

        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=trigger.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]

        grown = next(card for card in branch.state.deck.all_cards if card.instance_id == source.instance_id)
        self.assertEqual(grown.score_bonus, 10)
        self.assertEqual(branch.state.stance_changed_by_direct_effect_times, 3)
        self.assertEqual(branch.state.scheduled_effects[0].remaining_uses, 2)

    def test_stance_family_upgrade_does_not_trigger_persistent_growth(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("upgrade-growth", "412")
        trigger = CardRef("upgrade-trigger", "368")
        deck = DeckBelief(
            all_cards=(source, trigger),
            hand=(trigger,),
            draw_pile=(source,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.AGGRESSIVE_1,
            previous_stance=Stance.NEUTRAL,
            actions_remaining=1,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            strength_times=1,
            stance_changed_by_direct_effect_times=1,
        )

        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=trigger.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]

        unchanged = next(card for card in branch.state.deck.all_cards if card.instance_id == source.instance_id)
        self.assertEqual(branch.state.stance, Stance.AGGRESSIVE_2)
        self.assertEqual(unchanged.score_bonus, 0)
        self.assertEqual(branch.state.strength_times, 1)
        self.assertEqual(branch.state.stance_changed_by_direct_effect_times, 1)

    def test_card_used_growth_excludes_transient_used_card(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("card-used-source", "548")
        trigger = CardRef("card-used-trigger", "368")
        filler = CardRef("card-used-filler", "374")
        deck = DeckBelief(
            all_cards=(source, trigger, filler),
            hand=(source, trigger, filler),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            previous_stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            pending_after_card_ids=(),
        )
        installed = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(len(installed.scheduled_effects), 1)
        self.assertEqual(installed.scheduled_effects[0].trigger_phase, SchedulePhase.CARD_USED)

        branch = kernel.transition(
            replace(installed, actions_remaining=1),
            GameAction(ActionKind.PLAY_CARD, card_instance_id=trigger.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        cards = {card.instance_id: card for card in branch.state.deck.all_cards}
        self.assertEqual(cards[filler.instance_id].score_bonus, 2)
        self.assertEqual(cards[trigger.instance_id].score_bonus, 0)
        event_types = [event.event_type for event in branch.events]
        self.assertLess(event_types.index("TRIGGER_SCHEDULED"), event_types.index("STANCE"))

    def test_after_card_used_growth_includes_used_card_destination(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("after-card-source", "494")
        trigger = CardRef("after-card-trigger", "368")
        deck = DeckBelief(
            all_cards=(source, trigger),
            hand=(source, trigger),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            previous_stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            pending_after_card_ids=(),
        )
        installed = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state

        branch = kernel.transition(
            replace(installed, actions_remaining=1),
            GameAction(ActionKind.PLAY_CARD, card_instance_id=trigger.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        grown = next(
            card
            for card in branch.state.deck.all_cards
            if card.instance_id == trigger.instance_id
        )
        self.assertEqual(grown.score_bonus, 2)
        event_types = [event.event_type for event in branch.events]
        self.assertLess(event_types.index("STANCE"), event_types.index("TRIGGER_SCHEDULED"))

    def test_persistent_after_card_used_matches_upgraded_base_id(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("base-id-source", "780", upgraded=True)
        deck = DeckBelief(
            all_cards=(source,),
            hand=(source,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            stance=Stance.FULL_POWER,
            previous_stance=Stance.NEUTRAL,
            full_power=8,
            cumulative_full_power=10,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            pending_after_card_ids=(),
        )

        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]

        grown = next(
            card
            for card in branch.state.deck.all_cards
            if card.instance_id == source.instance_id
        )
        self.assertEqual(branch.state.deck.held, (grown,))
        self.assertEqual(grown.score_bonus, 6)
        self.assertEqual(branch.state.pending_after_card_ids, ())
        self.assertTrue(
            any(
                effect.persistent
                and effect.trigger_phase is SchedulePhase.AFTER_CARD_USED
                for effect in branch.state.scheduled_effects
            )
        )

    def test_card_used_type_filter_observes_pre_action_stance(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("filtered-source", "514")
        mental = CardRef("filtered-mental", "374")
        target = CardRef("filtered-target", "368")
        deck = DeckBelief(
            all_cards=(source, mental, target),
            hand=(source, mental),
            draw_pile=(target,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            stance=Stance.CONSERVE_1,
            previous_stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            pending_after_card_ids=(),
        )
        installed = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(installed.stance, Stance.AGGRESSIVE_2)

        branch = kernel.transition(
            replace(installed, actions_remaining=1),
            GameAction(ActionKind.PLAY_CARD, card_instance_id=mental.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        grown = next(
            card
            for card in branch.state.deck.all_cards
            if card.instance_id == target.instance_id
        )
        self.assertEqual(grown.score_bonus, 7)

    def test_nested_free_play_drains_after_card_events_inner_first(self) -> None:
        outer = CardDefinition(
            "after-outer",
            "After outer",
            PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.FREE_PLAY_SELECTED,
                    target_zones=(CardZone.DISCARD,),
                ),
            ),
        )
        target_definition = CardDefinition(
            "after-target",
            "After target",
            PlanType.ANOMALY,
        )
        observer = CardDefinition(
            "after-observer",
            "After observer",
            PlanType.ANOMALY,
            persistent_effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=1),),
                    delay_turns=0,
                    schedule_phase=SchedulePhase.AFTER_CARD_USED,
                    schedule_limit=None,
                ),
            ),
        )
        kernel = PythonNiaProKernel(
            {
                outer.card_id: outer,
                target_definition.card_id: target_definition,
                observer.card_id: observer,
            }
        )
        source = CardRef("after-outer-instance", outer.card_id)
        target = CardRef("after-target-instance", target_definition.card_id)
        watcher = CardRef("after-observer-instance", observer.card_id)
        deck = DeckBelief(
            all_cards=(source, target, watcher),
            hand=(source,),
            draw_pile=(watcher,),
            discard=(target,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            pending_after_card_ids=(),
        )
        pending = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(pending.pending_after_card_ids, (source.instance_id,))
        self.assertEqual(pending.score, 0)

        resolved = kernel.transition(
            pending,
            GameAction(ActionKind.FREE_PLAY, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(resolved.state.score, 2)
        self.assertEqual(resolved.state.pending_after_card_ids, ())
        self.assertEqual(
            [event.event_type for event in resolved.events].count("TRIGGER_SCHEDULED"),
            2,
        )

    def test_card_used_phase_conditions_share_pretrigger_snapshot(self) -> None:
        first_observer = CardDefinition(
            "snapshot-first",
            "Snapshot first",
            PlanType.ANOMALY,
            persistent_effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(
                        EffectSpec(EffectKind.STANCE, stance=Stance.AGGRESSIVE_1),
                    ),
                    delay_turns=0,
                    schedule_phase=SchedulePhase.CARD_USED,
                    schedule_limit=None,
                ),
            ),
        )
        second_observer = CardDefinition(
            "snapshot-second",
            "Snapshot second",
            PlanType.ANOMALY,
            persistent_effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(
                        EffectSpec(
                            EffectKind.GENKI,
                            amount=3,
                            condition=EffectCondition(
                                ConditionKind.STANCE_FAMILY,
                                value="preservation",
                            ),
                        ),
                    ),
                    delay_turns=0,
                    schedule_phase=SchedulePhase.CARD_USED,
                    schedule_limit=None,
                ),
            ),
        )
        trigger_definition = CardDefinition(
            "snapshot-trigger",
            "Snapshot trigger",
            PlanType.ANOMALY,
        )
        kernel = PythonNiaProKernel(
            {
                first_observer.card_id: first_observer,
                second_observer.card_id: second_observer,
                trigger_definition.card_id: trigger_definition,
            }
        )
        first = CardRef("snapshot-first-instance", first_observer.card_id)
        second = CardRef("snapshot-second-instance", second_observer.card_id)
        trigger = CardRef("snapshot-trigger-instance", trigger_definition.card_id)
        deck = DeckBelief(
            all_cards=(first, second, trigger),
            hand=(trigger,),
            draw_pile=(first, second),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            stance=Stance.CONSERVE_1,
            previous_stance=Stance.NEUTRAL,
            scheduled_effects=(),
            persistent_effects_initialized=False,
            pending_after_card_ids=(),
        )

        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=trigger.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.stance, Stance.AGGRESSIVE_1)
        self.assertEqual(branch.state.genki, 3)

    def test_timed_modifier_merge_lifecycle_and_state_key(self) -> None:
        source = CardRef("modifier-source", "modifier-card")
        definition = CardDefinition(
            card_id="modifier-card",
            name="Timed modifier setup",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(EffectKind.ENTHUSIASM_BONUS_BUFF, amount=2, duration=2),
                EffectSpec(EffectKind.ENTHUSIASM_BONUS_BUFF, amount=3, duration=2),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        deck = DeckBelief(
            all_cards=(source,),
            hand=(source,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=4,
            deck=deck,
            actions_remaining=1,
            hand_limit=0,
            passion=0,
            passion_buff=0,
            passion_gain_bonus_pct=0.0,
            enthusiasm_bonus_buffs=(),
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(
            played.enthusiasm_bonus_buffs,
            (TimedModifier(amount=5, remaining_turns=2, fresh=True),),
        )

        first_end = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(
            first_end.enthusiasm_bonus_buffs,
            (TimedModifier(amount=5, remaining_turns=2, fresh=False),),
        )
        second_end = kernel.transition(
            first_end,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(second_end.enthusiasm_bonus_buffs[0].remaining_turns, 1)
        third_end = kernel.transition(
            second_end,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(third_end.enthusiasm_bonus_buffs, ())

        state_key = RobustHighScorePlanner._semantic_state_key(played)
        self.assertNotEqual(
            state_key,
            RobustHighScorePlanner._semantic_state_key(
                replace(played, created_card_count=played.created_card_count + 1)
            ),
        )
        self.assertNotEqual(
            state_key,
            RobustHighScorePlanner._semantic_state_key(
                replace(
                    played,
                    enthusiasm_bonus_buffs=(
                        TimedModifier(amount=6, remaining_turns=2, fresh=True),
                    ),
                )
            ),
        )
        serialized = replace(
            self.suite.cases[0].request,
            state=played,
            card_slots=(),
        ).to_dict()
        self.assertEqual(
            serialized["state"]["enthusiasm_bonus_buffs"],
            [{"amount": 5, "remaining_turns": 2, "fresh": True}],
        )

    def test_real_anomaly_enthusiasm_buff_applies_before_score_rounding(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        setup = CardRef("enthusiasm-setup", "636")
        release = CardRef("enthusiasm-release", "370")
        deck = DeckBelief(
            all_cards=(setup, release),
            hand=(setup, release),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            hand_limit=0,
            deck=deck,
            stance=Stance.NEUTRAL,
            full_power=0,
            cumulative_full_power=0,
            passion=0,
            passion_buff=0,
            passion_gain_bonus_pct=0.0,
            parameter_gain_bonus_pct=0.0,
            enthusiasm_bonus_buffs=(),
            enthusiasm_multiplier_buffs=(),
        )
        setup_state = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=setup.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(
            setup_state.enthusiasm_multiplier_buffs,
            (TimedModifier(amount=0.5, remaining_turns=3, fresh=True),),
        )
        released = kernel.transition(
            setup_state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=release.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(released.passion, 12)
        self.assertEqual(released.score, 54)
        self.assertEqual(released.genki, 5)
        self.assertEqual(released.actions_remaining, 1)

    def test_full_power_charge_buff_affects_only_later_positive_gain(self) -> None:
        setup = CardRef("charge-buff-setup", "charge-buff")
        gain = CardRef("charge-buff-gain", "charge-gain")
        definitions = {
            "charge-buff": CardDefinition(
                card_id="charge-buff",
                name="Charge buff setup",
                plan=PlanType.ANOMALY,
                effects=(
                    EffectSpec(EffectKind.FULL_POWER, amount=3),
                    EffectSpec(EffectKind.FULL_POWER_CHARGE_BUFF, amount=0.5, duration=3),
                    EffectSpec(EffectKind.EXTRA_ACTION, amount=1),
                ),
            ),
            "charge-gain": CardDefinition(
                card_id="charge-gain",
                name="Charge gain",
                plan=PlanType.ANOMALY,
                effects=(EffectSpec(EffectKind.FULL_POWER, amount=3),),
            ),
        }
        kernel = PythonNiaProKernel(definitions)
        deck = DeckBelief(
            all_cards=(setup, gain),
            hand=(setup, gain),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            deck=deck,
            actions_remaining=1,
            hand_limit=0,
            full_power=0,
            cumulative_full_power=0,
            full_power_charge_buffs=(),
        )
        setup_state = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=setup.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(setup_state.full_power, 3)
        self.assertEqual(setup_state.cumulative_full_power, 3)
        self.assertEqual(
            setup_state.full_power_charge_buffs,
            (TimedModifier(amount=0.5, remaining_turns=3, fresh=True),),
        )
        gained = kernel.transition(
            setup_state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=gain.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(gained.full_power, 7.5)
        self.assertEqual(gained.cumulative_full_power, 7.5)

    def test_stance_lock_blocks_direct_change_and_delays_full_power_entry(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("stance-lock-source", "424")
        stance_card = CardRef("stance-lock-play", "370")
        hold_target = CardRef("stance-lock-hold", "374")
        deck = DeckBelief(
            all_cards=(source, stance_card, hold_target),
            hand=(source, stance_card),
            draw_pile=(hold_target,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=4,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            hand_limit=0,
            deck=deck,
            stance=Stance.NEUTRAL,
            full_power=0,
            cumulative_full_power=0,
            passion=0,
            passion_buff=0,
            passion_gain_bonus_pct=0.0,
            stance_lock_turns=0,
            stance_lock_fresh=False,
        )
        pending = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        locked = kernel.transition(
            pending,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=hold_target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(locked.stance_lock_turns, 1)
        self.assertTrue(locked.stance_lock_fresh)
        blocked = kernel.transition(
            locked,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=stance_card.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(blocked.state.stance, Stance.NEUTRAL)
        self.assertEqual(blocked.state.score, 15)
        self.assertIn("STANCE_BLOCKED", {event.event_type for event in blocked.events})

        charged = replace(
            blocked.state,
            full_power=10,
            cumulative_full_power=10,
        )
        first_end = kernel.transition(
            charged,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(first_end.stance, Stance.NEUTRAL)
        self.assertEqual(first_end.stance_lock_turns, 1)
        self.assertFalse(first_end.stance_lock_fresh)
        second_end = kernel.transition(
            first_end,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(second_end.stance, Stance.FULL_POWER)
        self.assertEqual(second_end.stance_lock_turns, 0)
        self.assertEqual(second_end.full_power, 0)
        self.assertEqual(second_end.full_power_times, first_end.full_power_times + 1)

    def test_nullified_stamina_cost_is_legal_and_consumed_only_by_stamina_card(self) -> None:
        setup = CardRef("nullify-setup", "nullify-setup")
        full_power_card = CardRef("nullify-full-power", "nullify-full-power")
        costly = CardRef("nullify-costly", "nullify-costly")
        definitions = {
            "nullify-setup": CardDefinition(
                card_id="nullify-setup",
                name="Nullify setup",
                plan=PlanType.ANOMALY,
                effects=(
                    EffectSpec(EffectKind.NULLIFY_COST_CARDS, amount=2),
                    EffectSpec(EffectKind.EXTRA_ACTION, amount=2),
                ),
            ),
            "nullify-full-power": CardDefinition(
                card_id="nullify-full-power",
                name="Full-power cost",
                plan=PlanType.ANOMALY,
                full_power_cost=1,
            ),
            "nullify-costly": CardDefinition(
                card_id="nullify-costly",
                name="Stamina cost",
                plan=PlanType.ANOMALY,
                stamina_cost=99,
                effects=(EffectSpec(EffectKind.SCORE, amount=10),),
            ),
        }
        kernel = PythonNiaProKernel(definitions)
        deck = DeckBelief(
            all_cards=(setup, full_power_card, costly),
            hand=(setup, full_power_card, costly),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            deck=deck,
            score=0,
            stamina=0,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            hand_limit=0,
            stance=Stance.NEUTRAL,
            full_power=1,
            cumulative_full_power=1,
            passion=0,
            nullify_cost_cards=0,
        )
        prepared = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=setup.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(prepared.nullify_cost_cards, 2)
        full_power_play = kernel.transition(
            prepared,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=full_power_card.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(full_power_play.nullify_cost_cards, 2)
        legal = kernel.legal_actions(
            full_power_play,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.CARD,
        )
        self.assertIn(costly.instance_id, {action.card_instance_id for action in legal})
        result = kernel.transition(
            full_power_play,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=costly.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(result.state.stamina, 0)
        self.assertEqual(result.state.nullify_cost_cards, 1)
        self.assertIn("NULLIFY_CARD_COST", {event.event_type for event in result.events})

    def test_stance_counters_drive_real_card_conditions_without_counting_family_upgrade(self) -> None:
        aggressive_one = CardRef("counter-aggressive-one", "counter-aggressive")
        aggressive_two = CardRef("counter-aggressive-two", "counter-aggressive")
        preserve = CardRef("counter-preserve", "counter-preserve")
        definitions = {
            "counter-aggressive": CardDefinition(
                card_id="counter-aggressive",
                name="Aggressive transition",
                plan=PlanType.ANOMALY,
                effects=(
                    EffectSpec(EffectKind.STANCE, stance=Stance.AGGRESSIVE_1),
                    EffectSpec(EffectKind.EXTRA_ACTION, amount=1),
                ),
            ),
            "counter-preserve": CardDefinition(
                card_id="counter-preserve",
                name="Preservation transition",
                plan=PlanType.ANOMALY,
                effects=(
                    EffectSpec(EffectKind.STANCE, stance=Stance.CONSERVE_1),
                    EffectSpec(EffectKind.EXTRA_ACTION, amount=1),
                ),
            ),
        }
        kernel = PythonNiaProKernel(definitions)
        deck = DeckBelief(
            all_cards=(aggressive_one, aggressive_two, preserve),
            hand=(aggressive_one, aggressive_two, preserve),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            deck=deck,
            actions_remaining=1,
            hand_limit=0,
            stance=Stance.NEUTRAL,
            passion=0,
            strength_times=0,
            preservation_times=0,
            full_power_times=0,
            stance_changed_by_direct_effect_times=0,
        )
        for card in (aggressive_one, aggressive_two, preserve):
            state = kernel.transition(
                state,
                GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id),
                plan_type=PlanType.ANOMALY,
            )[0].state
        self.assertEqual(state.strength_times, 1)
        self.assertEqual(state.preservation_times, 1)
        self.assertEqual(state.stance_changed_by_direct_effect_times, 2)

        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        condition_kernel = PythonNiaProKernel(catalog.cards)
        toughness = CardRef("condition-toughness", "518")
        branding = CardRef("condition-branding", "656")
        condition_deck = DeckBelief(
            all_cards=(toughness, branding),
            hand=(toughness, branding),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        condition_state = replace(
            self.suite.cases[0].request.state,
            deck=condition_deck,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            strength_times=1,
            preservation_times=2,
            full_power_times=0,
        )
        legal = condition_kernel.legal_actions(
            condition_state,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.CARD,
        )
        self.assertEqual(
            {action.card_instance_id for action in legal if action.card_instance_id},
            {toughness.instance_id, branding.instance_id},
        )

    def test_neo_impact_free_play_skips_cost_and_doubles_selected_card_effects(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("neo-source", "793")
        target = CardRef("neo-target", "370")
        deck = DeckBelief(
            all_cards=(source, target),
            hand=(source,),
            draw_pile=(),
            discard=(target,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            full_power=0,
            cumulative_full_power=0,
            passion=0,
            parameter_gain_bonus_pct=0.0,
            score_multiplier=1.0,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert played.pending_choice is not None
        self.assertEqual(played.pending_choice.action_kind, ActionKind.FREE_PLAY)
        self.assertEqual(played.pending_choice.candidate_instance_ids, (target.instance_id,))
        self.assertEqual(played.double_card_effects_remaining, 1)
        self.assertEqual(played.actions_remaining, 1)
        self.assertEqual(played.deck.removed, (source,))

        branch = kernel.transition(
            played,
            GameAction(ActionKind.FREE_PLAY, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertIsNone(branch.state.pending_choice)
        self.assertEqual(branch.state.stamina, 30)
        self.assertEqual(branch.state.actions_remaining, 1)
        self.assertEqual(branch.state.double_card_effects_remaining, 0)
        self.assertEqual(branch.state.score, 68)
        self.assertEqual(branch.state.stance, Stance.AGGRESSIVE_2)
        self.assertEqual(branch.state.deck.removed, (source, target))
        self.assertEqual(
            [event.event_type for event in branch.events].count("SCORE"),
            2,
        )

    def test_free_play_unmet_condition_moves_card_without_consuming_double_effect(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("neo-condition-source", "793")
        target = CardRef("neo-condition-target", "406")
        deck = DeckBelief(
            all_cards=(source, target),
            hand=(source,),
            draw_pile=(target,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            full_power=0,
            cumulative_full_power=0,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        branch = kernel.transition(
            played,
            GameAction(ActionKind.FREE_PLAY, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.score, 0)
        self.assertEqual(branch.state.stamina, 30)
        self.assertEqual(branch.state.actions_remaining, 1)
        self.assertEqual(branch.state.double_card_effects_remaining, 1)
        self.assertEqual(branch.state.deck.removed, (source, target))
        self.assertIn("FREE_PLAY_CONDITION_SKIPPED", {event.event_type for event in branch.events})

    def test_free_play_masks_candidate_requiring_unsupported_nested_continuation(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("free-mask-source", "793")
        nested = CardRef("free-mask-nested", "793")
        supported = CardRef("free-mask-supported", "370")
        deck = DeckBelief(
            all_cards=(source, nested, supported),
            hand=(),
            draw_pile=(nested,),
            discard=(supported,),
            removed=(source,),
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            pending_choice=PendingChoice(
                ActionKind.FREE_PLAY,
                source.instance_id,
                (nested.instance_id, supported.instance_id),
            ),
            double_card_effects_remaining=1,
        )
        self.assertEqual(
            kernel.legal_actions(
                state,
                plan_type=PlanType.ANOMALY,
                decision_kind=DecisionKind.SECONDARY,
            ),
            (GameAction(ActionKind.FREE_PLAY, card_instance_id=supported.instance_id),),
        )
        with self.assertRaisesRegex(KernelError, "free_play_target_secondary_not_ready"):
            kernel.transition(
                state,
                GameAction(ActionKind.FREE_PLAY, card_instance_id=nested.instance_id),
                plan_type=PlanType.ANOMALY,
            )

    def test_free_play_masks_doubled_hold_with_intermediate_draw(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("free-zone-source", "793")
        target = CardRef("free-zone-target", "528")
        first = CardRef("free-zone-first", "368")
        second = CardRef("free-zone-second", "382")
        deck = DeckBelief(
            all_cards=(source, target, first, second),
            hand=(),
            draw_pile=(target, first, second),
            removed=(source,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            pending_choice=PendingChoice(
                ActionKind.FREE_PLAY,
                source.instance_id,
                (target.instance_id, first.instance_id, second.instance_id),
            ),
            double_card_effects_remaining=1,
        )

        legal = kernel.legal_actions(
            state,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.SECONDARY,
        )
        self.assertNotIn(
            GameAction(ActionKind.FREE_PLAY, card_instance_id=target.instance_id),
            legal,
        )
        with self.assertRaisesRegex(KernelError, "free_play_target_secondary_not_ready"):
            kernel.transition(
                state,
                GameAction(ActionKind.FREE_PLAY, card_instance_id=target.instance_id),
                plan_type=PlanType.ANOMALY,
            )

    def test_doubled_card_resumes_both_effect_passes_across_hold_choices(self) -> None:
        outer = CardDefinition("outer", "Outer", PlanType.ANOMALY)
        target_definition = CardDefinition(
            "double-hold",
            "Double hold",
            PlanType.ANOMALY,
            effects=(
                EffectSpec(EffectKind.SCORE, amount=10),
                EffectSpec(EffectKind.HOLD_SELECTED, target_zones=(CardZone.DRAW_PILE,)),
            ),
            rarity="SSR",
        )
        filler = CardDefinition("filler", "Filler", PlanType.ANOMALY)
        kernel = PythonNiaProKernel(
            {
                outer.card_id: outer,
                target_definition.card_id: target_definition,
                filler.card_id: filler,
            }
        )
        source = CardRef("outer-source", outer.card_id)
        target = CardRef("double-hold-target", target_definition.card_id)
        first = CardRef("hold-first", filler.card_id)
        second = CardRef("hold-second", filler.card_id)
        deck = DeckBelief(
            all_cards=(source, target, first, second),
            hand=(),
            draw_pile=(first, second),
            discard=(target,),
            removed=(source,),
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stance=Stance.NEUTRAL,
            pending_choice=PendingChoice(ActionKind.FREE_PLAY, source.instance_id, (target.instance_id,)),
            double_card_effects_remaining=1,
        )
        first_pass = kernel.transition(
            state,
            GameAction(ActionKind.FREE_PLAY, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert first_pass.pending_choice is not None
        self.assertEqual(first_pass.score, 10)
        self.assertEqual(first_pass.pending_choice.remaining_effect_passes, 1)

        second_pass = kernel.transition(
            first_pass,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=first.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert second_pass.pending_choice is not None
        self.assertEqual(second_pass.score, 20)
        self.assertEqual(second_pass.pending_choice.candidate_instance_ids, (second.instance_id,))
        self.assertEqual(second_pass.pending_choice.remaining_effect_passes, 0)

        resolved = kernel.transition(
            second_pass,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=second.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNone(resolved.pending_choice)
        self.assertEqual(resolved.deck.held, (first, second))
        self.assertEqual(resolved.score, 20)
        self.assertTrue(resolved.deck.conservation_valid)

    def test_nonterminal_free_play_effect_resumes_outer_card(self) -> None:
        source_definition = CardDefinition(
            "nested-free",
            "Nested free",
            PlanType.ANOMALY,
            effects=(
                EffectSpec(EffectKind.FREE_PLAY_SELECTED, target_zones=(CardZone.DISCARD,)),
                EffectSpec(EffectKind.SCORE, amount=10),
            ),
        )
        target_definition = CardDefinition("nested-target", "Nested target", PlanType.ANOMALY)
        kernel = PythonNiaProKernel(
            {
                source_definition.card_id: source_definition,
                target_definition.card_id: target_definition,
            }
        )
        source = CardRef("nested-free-source", source_definition.card_id)
        target = CardRef("nested-free-target", target_definition.card_id)
        deck = DeckBelief(
            all_cards=(source, target),
            hand=(source,),
            draw_pile=(),
            discard=(target,),
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            actions_remaining=1,
            stance=Stance.NEUTRAL,
            passion=0,
            parameter_gain_bonus_pct=0,
            score_multiplier=1.0,
        )
        pending = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert pending.pending_choice is not None
        self.assertEqual(pending.pending_choice.continuation_index, 1)
        branch = kernel.transition(
            pending,
            GameAction(ActionKind.FREE_PLAY, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.score, 10)
        self.assertIsNone(branch.state.pending_choice)
        self.assertEqual(branch.state.resolution_stack, ())
        self.assertEqual(branch.state.pending_after_card_ids, ())

    def test_legend_rarity_does_not_consume_or_apply_double_effect(self) -> None:
        definition = CardDefinition(
            "legend-card",
            "Legend card",
            PlanType.ANOMALY,
            effects=(EffectSpec(EffectKind.SCORE, amount=10),),
            rarity="L",
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        card = CardRef("legend-instance", definition.card_id)
        deck = DeckBelief(all_cards=(card,), hand=(card,), draw_pile=())
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            stance=Stance.NEUTRAL,
            actions_remaining=1,
            double_card_effects_remaining=1,
        )
        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.score, 10)
        self.assertEqual(branch.state.double_card_effects_remaining, 1)
        self.assertEqual([event.event_type for event in branch.events].count("SCORE"), 1)

    def test_move_random_to_hand_filters_zones_and_preserves_probability_mass(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("move-source", "476")
        growth_target = CardRef("move-growth-target", "382")
        draw_target = CardRef("move-draw-target", "383", upgraded=True)
        discard_target = CardRef("move-discard-target", "383", upgraded=True, score_bonus=3)
        unrelated = CardRef("move-unrelated", "370")
        deck = DeckBelief(
            all_cards=(source, growth_target, draw_target, discard_target, unrelated),
            hand=(source,),
            draw_pile=(growth_target, draw_target, unrelated),
            discard=(discard_target,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
        )
        branches = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )
        self.assertEqual(len(branches), 2)
        self.assertAlmostEqual(sum(branch.probability for branch in branches), 1.0)
        self.assertEqual({branch.probability for branch in branches}, {0.5})
        self.assertEqual(
            {branch.events[-2].detail for branch in branches},
            {
                "DRAW_PILE->HAND:move-draw-target",
                "DISCARD->HAND:move-discard-target",
            },
        )
        for branch in branches:
            self.assertIn("383", {card.card_id for card in branch.state.deck.hand})
            modified = next(card for card in branch.state.deck.all_cards if card.instance_id == growth_target.instance_id)
            self.assertEqual(modified.score_repeat_bonus, 1)
            self.assertTrue(branch.state.deck.conservation_valid)

    def test_move_random_to_hand_without_candidate_is_explicit_noop(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("move-empty-source", "476")
        growth_target = CardRef("move-empty-growth-target", "382")
        deck = DeckBelief(
            all_cards=(source, growth_target),
            hand=(source,),
            draw_pile=(growth_target,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
        )
        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.events[-2].event_type, "MOVE_RANDOM_TO_HAND_EMPTY")
        self.assertEqual(branch.state.deck.hand, ())
        self.assertTrue(branch.state.deck.conservation_valid)

    def test_exchange_hand_discards_remainder_and_draws_same_count(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("exchange-source", "805")
        old_hand = (CardRef("exchange-old-1", "370"), CardRef("exchange-old-2", "372"))
        replacements = (CardRef("exchange-new-1", "374"), CardRef("exchange-new-2", "376"))
        deck = DeckBelief(
            all_cards=(source, *old_hand, *replacements),
            hand=(source, *old_hand),
            draw_pile=replacements,
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.CONSERVE_1,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
        )
        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(branch.state.deck.hand, replacements)
        self.assertEqual(branch.state.deck.discard, old_hand)
        self.assertEqual(branch.state.deck.removed, (source,))
        self.assertEqual(
            [event.event_type for event in branch.events[-3:]],
            ["EXCHANGE_HAND", "DRAW", "DRAW"],
        )
        self.assertTrue(branch.state.deck.conservation_valid)

    def test_exchange_hand_reshuffles_only_after_draw_pile_is_exhausted(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("exchange-reshuffle-source", "805")
        old_hand = (
            CardRef("exchange-reshuffle-old-1", "370"),
            CardRef("exchange-reshuffle-old-2", "372"),
        )
        first_draw = CardRef("exchange-reshuffle-first", "374")
        deck = DeckBelief(
            all_cards=(source, *old_hand, first_draw),
            hand=(source, *old_hand),
            draw_pile=(first_draw,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.CONSERVE_1,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
        )
        branches = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )
        self.assertEqual(len(branches), 2)
        self.assertEqual({branch.probability for branch in branches}, {0.5})
        for branch in branches:
            self.assertIn(first_draw, branch.state.deck.hand)
            self.assertEqual(len(branch.state.deck.hand), 2)
            self.assertEqual(len(branch.state.deck.draw_pile), 1)
            self.assertEqual(branch.state.deck.discard, ())
            self.assertEqual(branch.state.deck.knowledge, DeckKnowledge.KNOWN_COMPOSITION)
            self.assertTrue(branch.state.deck.conservation_valid)

    def test_next_turn_schedule_fires_once_and_is_removed(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("scheduled-stance-source", "436")
        deck = DeckBelief(
            all_cards=(source,),
            hand=(source,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            deck=deck,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(played.state.stance, Stance.AGGRESSIVE_1)
        self.assertEqual(played.state.score, 56)
        self.assertEqual(len(played.state.scheduled_effects), 1)
        self.assertEqual(played.events[-1].event_type, "SCHEDULE_EFFECT")

        ended = kernel.transition(
            played.state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(ended.state.stance, Stance.CONSERVE_1)
        self.assertEqual(ended.state.scheduled_effects, ())
        self.assertEqual(
            [event.event_type for event in ended.events],
            ["END_TURN", "TRIGGER_SCHEDULED", "STANCE"],
        )

    def test_next_turn_schedule_runs_after_normal_hand_refill(self) -> None:
        definition = CardDefinition(
            card_id="scheduled-draw",
            name="Scheduled draw probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.DRAW, amount=1),),
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("scheduled-draw-source", definition.card_id)
        draw_cards = tuple(CardRef(f"scheduled-draw-{index}", definition.card_id) for index in range(3))
        deck = DeckBelief(
            all_cards=(source, *draw_cards),
            hand=(source,),
            draw_pile=draw_cards,
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=2,
            deck=deck,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        ended = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(ended.state.deck.hand, draw_cards)
        self.assertEqual(
            [event.event_type for event in ended.events],
            ["END_TURN", "DRAW", "DRAW", "TRIGGER_SCHEDULED", "DRAW"],
        )

    def test_start_of_turn_schedule_runs_before_normal_hand_refill(self) -> None:
        definition = CardDefinition(
            card_id="scheduled-start-draw",
            name="Start scheduled draw probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.DRAW, amount=1),),
                    schedule_phase=SchedulePhase.START_OF_TURN,
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("scheduled-start-source", definition.card_id)
        first = CardRef("scheduled-start-first", definition.card_id)
        second = CardRef("scheduled-start-second", definition.card_id)
        deck = DeckBelief(
            all_cards=(source, first, second),
            hand=(source,),
            draw_pile=(first, second),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=1,
            deck=deck,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        ended = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(ended.state.deck.hand, (first,))
        self.assertEqual(ended.state.deck.draw_pile, (second,))
        self.assertEqual(
            [event.event_type for event in ended.events],
            ["END_TURN", "TRIGGER_SCHEDULED", "DRAW"],
        )

    def test_before_start_schedule_observes_full_power_exit(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("before-start-source", "414")
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            score=0,
            score_multiplier=1.0,
            stance=Stance.FULL_POWER,
            previous_stance=Stance.NEUTRAL,
            full_power=0,
            cumulative_full_power=0,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            deck=DeckBelief(all_cards=(source,), hand=(source,), draw_pile=()),
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(played.stance, Stance.FULL_POWER)
        self.assertEqual(played.full_power, 3)

        ended = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(ended.state.stance, Stance.CONSERVE_2)
        self.assertEqual(ended.state.previous_stance, Stance.NEUTRAL)
        self.assertEqual(ended.state.scheduled_effects, ())
        event_types = [event.event_type for event in ended.events]
        self.assertLess(event_types.index("EXIT_FULL_POWER"), event_types.index("TRIGGER_SCHEDULED"))
        self.assertLess(event_types.index("TRIGGER_SCHEDULED"), event_types.index("STANCE"))

    def test_end_of_turn_schedule_fires_on_current_boundary_and_repeats(self) -> None:
        definition = CardDefinition(
            card_id="end-of-turn-schedule",
            name="End-of-turn schedule probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=10),),
                    delay_turns=0,
                    schedule_phase=SchedulePhase.END_OF_TURN,
                    schedule_limit=2,
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("end-of-turn-source", definition.card_id)
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            deck=DeckBelief(all_cards=(source,), hand=(source,), draw_pile=()),
        )
        current = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state

        first = kernel.transition(
            current,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(first.state.score, 10)
        self.assertEqual(first.state.scheduled_effects[0].turns_until_fire, 0)
        self.assertEqual(first.state.scheduled_effects[0].remaining_uses, 1)
        self.assertEqual(
            [event.event_type for event in first.events],
            ["END_TURN", "TRIGGER_SCHEDULED", "SCORE"],
        )

        second = kernel.transition(
            first.state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(second.state.score, 20)
        self.assertEqual(second.state.scheduled_effects, ())

    def test_after_start_schedule_observes_full_power_after_refill(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("after-start-source", "660")
        filler = CardRef("after-start-filler", "370")
        deck = DeckBelief(
            all_cards=(source, filler),
            hand=(source,),
            draw_pile=(filler,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            full_power=7,
            cumulative_full_power=7,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=1,
            deck=deck,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        ended = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(ended.state.stance, Stance.FULL_POWER)
        self.assertEqual(ended.state.score, 45)
        self.assertEqual(ended.state.deck.hand, (filler,))
        self.assertEqual(
            [event.event_type for event in ended.events],
            [
                "END_TURN",
                "ENTER_FULL_POWER",
                "DRAW",
                "FULL_POWER_READY",
                "TRIGGER_SCHEDULED",
                "SCORE",
            ],
        )

    def test_recurring_schedule_limit_fires_exact_number_of_times(self) -> None:
        definition = CardDefinition(
            card_id="limited-recurring-schedule",
            name="Limited recurring schedule probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=10),),
                    schedule_phase=SchedulePhase.START_OF_TURN,
                    schedule_limit=2,
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("limited-recurring-source", definition.card_id)
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=4,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            deck=DeckBelief(all_cards=(source,), hand=(source,), draw_pile=()),
        )
        current = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        scores: list[int] = []
        scheduled_counts: list[int] = []
        for _index in range(3):
            current = kernel.transition(
                current,
                GameAction(ActionKind.END_TURN),
                plan_type=PlanType.ANOMALY,
            )[0].state
            scores.append(current.score)
            scheduled_counts.append(len(current.scheduled_effects))
        self.assertEqual(scores, [10, 20, 20])
        self.assertEqual(scheduled_counts, [1, 0, 0])

    def test_schedule_limit_is_not_consumed_when_condition_misses(self) -> None:
        condition = EffectCondition(ConditionKind.STANCE_FAMILY, value="fullPower")
        definition = CardDefinition(
            card_id="conditional-recurring-schedule",
            name="Conditional recurring schedule probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(
                        EffectSpec(EffectKind.SCORE, amount=10, condition=condition),
                    ),
                    schedule_phase=SchedulePhase.START_OF_TURN,
                    schedule_limit=1,
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("conditional-recurring-source", definition.card_id)
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            full_power=0,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            deck=DeckBelief(all_cards=(source,), hand=(source,), draw_pile=()),
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        missed = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(missed.score, 0)
        self.assertEqual(missed.scheduled_effects[0].remaining_uses, 1)

        armed = replace(missed, full_power=10)
        fired = kernel.transition(
            armed,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(fired.stance, Stance.FULL_POWER)
        self.assertEqual(fired.score, 30)
        self.assertEqual(fired.scheduled_effects, ())

    def test_schedule_ttl_expires_after_phase_opportunities(self) -> None:
        definition = CardDefinition(
            card_id="ttl-schedule",
            name="TTL schedule probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=3),),
                    schedule_phase=SchedulePhase.START_OF_TURN,
                    schedule_limit=None,
                    schedule_ttl=2,
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("ttl-schedule-source", definition.card_id)
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=4,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            deck=DeckBelief(all_cards=(source,), hand=(source,), draw_pile=()),
        )
        current = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        scores: list[int] = []
        for _index in range(3):
            current = kernel.transition(
                current,
                GameAction(ActionKind.END_TURN),
                plan_type=PlanType.ANOMALY,
            )[0].state
            scores.append(current.score)
        self.assertEqual(scores, [3, 6, 6])
        self.assertEqual(current.scheduled_effects, ())

    def test_trouble_count_condition_excludes_removed_cards(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("trouble-condition-source", "698")
        first = CardRef("trouble-condition-first", "23")
        second = CardRef("trouble-condition-second", "23")

        def play_with_second_zone(zone: CardZone):
            zones = {
                CardZone.DISCARD: {"discard": (second,)},
                CardZone.REMOVED: {"removed": (second,)},
            }[zone]
            deck = DeckBelief(
                all_cards=(source, first, second),
                hand=(source,),
                draw_pile=(first,),
                knowledge=DeckKnowledge.KNOWN_ORDER,
                **zones,
            )
            state = replace(
                self.suite.cases[0].request.state,
                deck=deck,
                stance=Stance.NEUTRAL,
                stamina=30,
                max_stamina=30,
                actions_remaining=1,
            )
            return kernel.transition(
                state,
                GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
                plan_type=PlanType.ANOMALY,
            )[0]

        active = play_with_second_zone(CardZone.DISCARD)
        removed = play_with_second_zone(CardZone.REMOVED)
        self.assertEqual(active.state.deck.hand, (first,))
        self.assertIn("DRAW", {event.event_type for event in active.events})
        self.assertEqual(removed.state.deck.hand, ())
        self.assertNotIn("DRAW", {event.event_type for event in removed.events})
        self.assertIn("CONDITION_SKIPPED", {event.event_type for event in removed.events})

    def test_next_turn_schedule_does_not_fire_after_terminal_boundary(self) -> None:
        definition = CardDefinition(
            card_id="terminal-schedule",
            name="Terminal schedule probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(
                    EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=100),),
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("terminal-schedule-source", definition.card_id)
        deck = DeckBelief(all_cards=(source,), hand=(source,), draw_pile=())
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=1,
            score=0,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
            hand_limit=0,
            deck=deck,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        ended = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertTrue(ended.terminal)
        self.assertEqual(ended.score, 0)
        self.assertEqual(ended.scheduled_effects, ())

        invalid = replace(
            state,
            scheduled_effects=(ScheduledEffectRef(source.instance_id, "wrong-card", 0),),
        )
        with self.assertRaises(HighScoreContractError):
            invalid.validate()

    def test_target_modification_after_hold_selection_updates_card_instance(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("growth-source", "487")
        target = CardRef("growth-target", "370")
        deck = DeckBelief(
            all_cards=(source, target),
            hand=(source,),
            draw_pile=(target,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            actions_remaining=1,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNotNone(played.pending_choice)

        resolved = kernel.transition(
            played,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertIsNone(resolved.state.pending_choice)
        self.assertEqual(resolved.state.actions_remaining, 1)
        self.assertEqual(resolved.state.full_power, 2)
        self.assertEqual(resolved.state.deck.held[0].score_bonus, 5)
        self.assertEqual(
            next(card for card in resolved.state.deck.all_cards if card.instance_id == target.instance_id).score_bonus,
            5,
        )
        self.assertTrue(resolved.state.deck.conservation_valid)
        self.assertEqual(resolved.events[-1].event_type, "CARD_MODIFICATION")
        self.assertEqual(resolved.events[-1].detail, target.instance_id)

    def test_type_target_modifies_only_matching_targetable_cards(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("active-growth-source", "745")
        active = CardRef("active-growth-target", "384")
        mental = CardRef("mental-growth-target", "440")
        removed = CardRef("removed-growth-target", "384")
        deck = DeckBelief(
            all_cards=(source, active, mental, removed),
            hand=(source, active, mental),
            draw_pile=(),
            removed=(removed,),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            full_power=3,
            stamina=30,
            max_stamina=30,
            actions_remaining=2,
        )
        result = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        cards = {card.instance_id: card for card in result.deck.all_cards}
        self.assertEqual(cards[source.instance_id].score_bonus, 0)
        self.assertEqual(cards[active.instance_id].score_bonus, 4)
        self.assertEqual(cards[mental.instance_id].score_bonus, 0)
        self.assertEqual(cards[removed.instance_id].score_bonus, 0)

    def test_effect_target_cost_reduction_is_used_by_legality_and_payment(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("cost-growth-source", "558")
        target = CardRef("cost-growth-target", "370")
        other = CardRef("cost-growth-other", "384")
        deck = DeckBelief(
            all_cards=(source, target, other),
            hand=(source, target, other),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.AGGRESSIVE_1,
            stamina=6,
            max_stamina=30,
            genki=0,
            actions_remaining=2,
        )
        modified = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        cards = {card.instance_id: card for card in modified.deck.all_cards}
        self.assertEqual(cards[target.instance_id].score_bonus, 37)
        self.assertEqual(cards[target.instance_id].stamina_cost_delta, -7)
        self.assertEqual(cards[other.instance_id].score_bonus, 0)
        self.assertEqual(modified.stamina, 0)
        legal = kernel.legal_actions(
            modified,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.CARD,
        )
        self.assertIn(target.instance_id, {action.card_instance_id for action in legal})
        played = kernel.transition(
            modified,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(played.state.stamina, 0)
        self.assertIn(0, [event.amount for event in played.events if event.event_type == "COST_STAMINA"])

    def test_anomaly_stance_and_half_cost_round_up_payment_and_legality(self) -> None:
        definition = CardDefinition(
            "cost-seven",
            "Cost seven",
            PlanType.ANOMALY,
            stamina_cost=7,
            effects=(EffectSpec(EffectKind.SCORE, amount=1),),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        expected_costs = {
            (Stance.NEUTRAL, 0, 0): 7,
            (Stance.AGGRESSIVE_1, 0, 0): 14,
            (Stance.AGGRESSIVE_2, 0, 0): 14,
            (Stance.CONSERVE_1, 0, 0): 4,
            (Stance.CONSERVE_2, 0, 0): 2,
            (Stance.NEUTRAL, 1, 0): 4,
            (Stance.AGGRESSIVE_1, 1, 0): 7,
            (Stance.CONSERVE_1, 1, 0): 2,
            (Stance.CONSERVE_2, 1, 0): 1,
            (Stance.NEUTRAL, 0, 1): 14,
            (Stance.AGGRESSIVE_1, 0, 1): 28,
            (Stance.CONSERVE_1, 0, 1): 7,
            (Stance.CONSERVE_2, 0, 1): 4,
            (Stance.NEUTRAL, 1, 1): 7,
            (Stance.AGGRESSIVE_1, 1, 1): 14,
            (Stance.CONSERVE_1, 1, 1): 4,
            (Stance.CONSERVE_2, 1, 1): 2,
        }
        for index, ((stance, half_cost_turns, double_cost_turns), expected_cost) in enumerate(
            expected_costs.items()
        ):
            with self.subTest(
                stance=stance,
                half_cost_turns=half_cost_turns,
                double_cost_turns=double_cost_turns,
            ):
                card = CardRef(f"cost-seven-{index}", definition.card_id)
                deck = DeckBelief(
                    all_cards=(card,),
                    hand=(card,),
                    draw_pile=(),
                    knowledge=DeckKnowledge.KNOWN_COMPOSITION,
                )
                affordable = replace(
                    self.suite.cases[0].request.state,
                    deck=deck,
                    stance=stance,
                    half_cost_turns=half_cost_turns,
                    half_cost_fresh=False,
                    double_cost_turns=double_cost_turns,
                    double_cost_fresh=False,
                    stamina=expected_cost,
                    max_stamina=30,
                    genki=0,
                    actions_remaining=1,
                )
                legal = kernel.legal_actions(
                    affordable,
                    plan_type=PlanType.ANOMALY,
                    decision_kind=DecisionKind.CARD,
                )
                self.assertIn(card.instance_id, {action.card_instance_id for action in legal})
                branch = kernel.transition(
                    affordable,
                    GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id),
                    plan_type=PlanType.ANOMALY,
                )[0]
                self.assertEqual(branch.state.stamina, 0)
                self.assertIn(
                    expected_cost,
                    [event.amount for event in branch.events if event.event_type == "COST_STAMINA"],
                )

                unaffordable = replace(affordable, stamina=max(0, expected_cost - 1))
                legal = kernel.legal_actions(
                    unaffordable,
                    plan_type=PlanType.ANOMALY,
                    decision_kind=DecisionKind.CARD,
                )
                self.assertNotIn(card.instance_id, {action.card_instance_id for action in legal})

    def test_new_half_cost_duration_survives_first_end_turn_then_decrements(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        card = CardRef("half-cost-source", "448")
        deck = DeckBelief(
            all_cards=(card,),
            hand=(card,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=3,
            deck=deck,
            stance=Stance.NEUTRAL,
            stamina=3,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            hand_limit=0,
            half_cost_turns=0,
            half_cost_fresh=False,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(played.half_cost_turns, 2)
        self.assertTrue(played.half_cost_fresh)

        first_end = kernel.transition(
            played,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(first_end.half_cost_turns, 2)
        self.assertFalse(first_end.half_cost_fresh)
        second_end = kernel.transition(
            first_end,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(second_end.half_cost_turns, 1)

    def test_hand_score_repeat_modification_repeats_future_score_effect(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("repeat-source", "440")
        target = CardRef("repeat-target", "384")
        deck = DeckBelief(
            all_cards=(source, target),
            hand=(source, target),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            full_power=3,
            cumulative_full_power=0,
            passion=0,
            parameter_gain_bonus_pct=0.0,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
        )
        modified = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(modified.deck.hand[0].score_repeat_bonus, 1)
        result = kernel.transition(
            modified,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(result.state.score, 30)
        self.assertEqual(sum(event.event_type == "SCORE" for event in result.events), 2)

    def test_card_modifiers_are_part_of_draw_and_action_equivalence(self) -> None:
        source = CardRef("modifier-draw-source", "an-draw")
        plain = CardRef("modifier-plain", "an-small")
        grown = CardRef("modifier-grown", "an-small", score_bonus=5)
        deck = DeckBelief(
            all_cards=(source, plain, grown),
            hand=(source,),
            draw_pile=(plain, grown),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(self.suite.cases[0].request.state, deck=deck, actions_remaining=1)
        branches = self.kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )
        self.assertEqual(len(branches), 2)
        self.assertEqual({branch.probability for branch in branches}, {0.5})

        hand_state = replace(
            state,
            deck=DeckBelief(
                all_cards=(plain, grown),
                hand=(plain, grown),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            ),
        )
        actions = (
            GameAction(ActionKind.PLAY_CARD, card_instance_id=plain.instance_id),
            GameAction(ActionKind.PLAY_CARD, card_instance_id=grown.instance_id),
        )
        self.assertEqual(len(RobustHighScorePlanner._merge_equivalent_actions(hand_state, actions)), 2)

    def test_real_anomaly_conditional_effects_are_state_dependent(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)
        card = CardRef("conditional-card", "386")
        deck = DeckBelief(
            all_cards=(card,),
            hand=(card,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        base = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.NEUTRAL,
            full_power=0,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
        )
        action = GameAction(kind=ActionKind.PLAY_CARD, card_instance_id=card.instance_id)
        neutral = kernel.transition(base, action, plan_type=self.suite.cases[0].request.plan_type)[0]
        strength = kernel.transition(
            replace(base, stance=Stance.AGGRESSIVE_1),
            action,
            plan_type=self.suite.cases[0].request.plan_type,
        )[0]
        self.assertEqual(neutral.state.full_power, 0)
        self.assertEqual(strength.state.full_power, 3)
        self.assertIn("CONDITION_SKIPPED", {event.event_type for event in neutral.events})

    def test_immediate_conditions_use_pre_card_snapshot(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)

        def play(card_id: str, *, stance: Stance, full_power: int):
            card = CardRef(f"pre-card-{card_id}", card_id)
            deck = DeckBelief(
                all_cards=(card,),
                hand=(card,),
                draw_pile=(),
                knowledge=DeckKnowledge.KNOWN_COMPOSITION,
            )
            state = replace(
                self.suite.cases[0].request.state,
                deck=deck,
                stance=stance,
                full_power=full_power,
                cumulative_full_power=full_power,
                stamina=30,
                max_stamina=30,
                genki=0,
                actions_remaining=1,
            )
            action = GameAction(kind=ActionKind.PLAY_CARD, card_instance_id=card.instance_id)
            return kernel.transition(state, action, plan_type=self.suite.cases[0].request.plan_type)[0].state

        changed_from_strength = play("396", stance=Stance.AGGRESSIVE_1, full_power=0)
        self.assertEqual(changed_from_strength.stance, Stance.CONSERVE_1)
        self.assertEqual(changed_from_strength.full_power, 3)
        threshold_not_met = play("394", stance=Stance.NEUTRAL, full_power=0)
        self.assertEqual(threshold_not_met.stance, Stance.NEUTRAL)
        self.assertEqual(threshold_not_met.full_power, 2)
        threshold_met = play("394", stance=Stance.NEUTRAL, full_power=1)
        self.assertEqual(threshold_met.stance, Stance.CONSERVE_1)

    def test_real_anomaly_card_condition_controls_legality(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)
        card = CardRef("full-power-card", "406")
        deck = DeckBelief(
            all_cards=(card,),
            hand=(card,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        base = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.NEUTRAL,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
        )
        decision_kind = self.suite.cases[0].request.decision_kind
        plan_type = self.suite.cases[0].request.plan_type
        neutral = kernel.legal_actions(base, plan_type=plan_type, decision_kind=decision_kind)
        full_power = kernel.legal_actions(
            replace(base, stance=Stance.FULL_POWER),
            plan_type=plan_type,
            decision_kind=decision_kind,
        )
        self.assertNotIn(card.instance_id, {action.card_instance_id for action in neutral})
        self.assertIn(card.instance_id, {action.card_instance_id for action in full_power})

    def test_cumulative_full_power_is_not_spent_with_current_charge(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)
        card = CardRef("cumulative-card", "604")
        deck = DeckBelief(
            all_cards=(card,),
            hand=(card,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            score=0,
            score_multiplier=1.0,
            stance=Stance.NEUTRAL,
            full_power=1,
            cumulative_full_power=5,
            passion=0,
            parameter_gain_bonus_pct=0.0,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
        )
        action = GameAction(kind=ActionKind.PLAY_CARD, card_instance_id=card.instance_id)
        result = kernel.transition(state, action, plan_type=self.suite.cases[0].request.plan_type)[0].state
        self.assertEqual(result.full_power, 3)
        self.assertEqual(result.cumulative_full_power, 7)
        self.assertEqual(result.score, 7)

    def test_full_power_stance_enters_and_exits_at_turn_boundaries(self) -> None:
        base = self.suite.cases[0].request.state
        charged = replace(
            base,
            turns_remaining=3,
            stance=Stance.AGGRESSIVE_1,
            full_power=10,
            cumulative_full_power=10,
        )
        entered = self.kernel.transition(
            charged,
            GameAction(kind=ActionKind.END_TURN),
            plan_type=self.suite.cases[0].request.plan_type,
        )[0]
        self.assertEqual(entered.state.stance, Stance.FULL_POWER)
        self.assertEqual(entered.state.full_power, 0)
        self.assertIn("ENTER_FULL_POWER", {event.event_type for event in entered.events})
        exited = self.kernel.transition(
            entered.state,
            GameAction(kind=ActionKind.END_TURN),
            plan_type=self.suite.cases[0].request.plan_type,
        )[0]
        self.assertEqual(exited.state.stance, Stance.NEUTRAL)
        self.assertIn("EXIT_FULL_POWER", {event.event_type for event in exited.events})

        terminal = self.kernel.transition(
            replace(charged, turns_remaining=1),
            GameAction(kind=ActionKind.END_TURN),
            plan_type=self.suite.cases[0].request.plan_type,
        )[0]
        self.assertEqual(terminal.state.stance, Stance.AGGRESSIVE_1)
        self.assertEqual(terminal.state.full_power, 10)
        self.assertEqual(terminal.state.actions_remaining, 0)
        self.assertNotIn("ENTER_FULL_POWER", {event.event_type for event in terminal.events})

    def test_anomaly_stance_applies_score_and_cost_multipliers(self) -> None:
        definition = CardDefinition(
            card_id="stance-multiplier-card",
            name="Stance multiplier probe",
            plan=PlanType.ANOMALY,
            stamina_cost=4,
            effects=(EffectSpec(EffectKind.SCORE, amount=10),),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        card = CardRef("stance-multiplier-instance", definition.card_id)
        expectations = {
            Stance.NEUTRAL: (4, 10),
            Stance.CONSERVE_1: (2, 5),
            Stance.CONSERVE_2: (1, 3),
            Stance.AGGRESSIVE_1: (8, 20),
            Stance.AGGRESSIVE_2: (8, 25),
            Stance.LEISURE: (0, 0),
            Stance.FULL_POWER: (4, 30),
        }
        for stance, (expected_cost, expected_score) in expectations.items():
            with self.subTest(stance=stance.value):
                state = replace(
                    self.suite.cases[0].request.state,
                    score=0,
                    score_multiplier=1.0,
                    stamina=20,
                    max_stamina=20,
                    genki=0,
                    actions_remaining=1,
                    stance=stance,
                    passion=0,
                    parameter_gain_bonus_pct=0.0,
                    deck=DeckBelief(all_cards=(card,), hand=(card,), draw_pile=()),
                )
                result = kernel.transition(
                    state,
                    GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id),
                    plan_type=PlanType.ANOMALY,
                )[0].state
                self.assertEqual(result.stamina, 20 - expected_cost)
                self.assertEqual(result.score, expected_score)

    def test_leisure_blocks_preservation_and_counts_as_preservation_condition(self) -> None:
        preservation_condition = EffectCondition(
            ConditionKind.STANCE_FAMILY,
            value="preservation",
        )
        definition = CardDefinition(
            card_id="leisure-transition-card",
            name="Leisure transition probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(EffectKind.STANCE, stance=Stance.LEISURE),
                EffectSpec(EffectKind.STANCE, stance=Stance.CONSERVE_1),
                EffectSpec(
                    EffectKind.EXTRA_ACTION,
                    amount=1,
                    condition=preservation_condition,
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        card = CardRef("leisure-transition-instance", definition.card_id)
        state = replace(
            self.suite.cases[0].request.state,
            score=0,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
            stance=Stance.CONSERVE_1,
            passion=0,
            leisure_times=0,
            stance_changed_by_direct_effect_times=0,
            deck=DeckBelief(all_cards=(card,), hand=(card,), draw_pile=()),
        )
        result = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(result.state.stance, Stance.LEISURE)
        self.assertEqual(result.state.leisure_times, 1)
        self.assertEqual(result.state.stance_changed_by_direct_effect_times, 1)
        self.assertEqual(result.state.actions_remaining, 1)
        self.assertEqual(result.state.passion, 0)
        self.assertEqual(result.state.genki, 0)
        self.assertIn("STANCE_BLOCKED", {event.event_type for event in result.events})

    def test_leisure_to_full_power_applies_start_turn_defaults(self) -> None:
        definition = CardDefinition(
            card_id="leisure-full-power-card",
            name="Leisure full-power probe",
            plan=PlanType.ANOMALY,
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        drawn = CardRef("leisure-full-power-drawn", definition.card_id)
        held = CardRef("leisure-full-power-held", definition.card_id)
        state = replace(
            self.suite.cases[0].request.state,
            turns_remaining=2,
            score=0,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=0,
            hand_limit=1,
            stance=Stance.LEISURE,
            full_power=10,
            cumulative_full_power=10,
            leisure_times=1,
            full_power_times=0,
            deck=DeckBelief(
                all_cards=(drawn, held),
                hand=(),
                draw_pile=(drawn,),
                held=(held,),
                knowledge=DeckKnowledge.KNOWN_ORDER,
            ),
        )
        result = kernel.transition(
            state,
            GameAction(ActionKind.END_TURN),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertEqual(result.state.stance, Stance.FULL_POWER)
        self.assertEqual(result.state.full_power, 0)
        self.assertEqual(result.state.full_power_times, 1)
        self.assertEqual(result.state.genki, 5)
        self.assertEqual(result.state.actions_remaining, 3)
        self.assertEqual(
            tuple(card.instance_id for card in result.state.deck.hand),
            (drawn.instance_id, held.instance_id),
        )
        self.assertEqual(result.state.deck.held, ())
        self.assertTrue(all(card.score_bonus == 10 for card in result.state.deck.all_cards))
        self.assertIn("FULL_POWER_READY", {event.event_type for event in result.events})

    def test_hold_self_moves_played_card_without_breaking_conservation(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)
        card = CardRef("hold-self", "372")
        deck = DeckBelief(
            all_cards=(card,),
            hand=(card,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.CONSERVE_1,
            stamina=20,
            max_stamina=20,
            genki=0,
            actions_remaining=1,
        )
        action = GameAction(kind=ActionKind.PLAY_CARD, card_instance_id=card.instance_id)
        result = kernel.transition(state, action, plan_type=self.suite.cases[0].request.plan_type)[0].state
        self.assertEqual(result.deck.held, (card,))
        self.assertEqual(result.deck.discard, ())
        self.assertTrue(result.deck.conservation_valid)

    def test_hold_selected_uses_explicit_secondary_decision(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)
        hold = CardRef("hold-source", "416")
        target = CardRef("hold-target", "384")
        deck = DeckBelief(
            all_cards=(hold, target),
            hand=(hold, target),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
        )
        played = kernel.transition(
            state,
            GameAction(kind=ActionKind.PLAY_CARD, card_instance_id=hold.instance_id),
            plan_type=self.suite.cases[0].request.plan_type,
        )[0].state
        self.assertIsNotNone(played.pending_choice)
        self.assertEqual(
            kernel.legal_actions(
                played,
                plan_type=self.suite.cases[0].request.plan_type,
                decision_kind=self.suite.cases[0].request.decision_kind,
            ),
            (),
        )
        planner = RobustHighScorePlanner(kernel, PlannerConfig(max_decision_depth=8, timeout_ms=2000.0))
        request = replace(
            self.suite.cases[0].request,
            state=played,
            decision_kind=DecisionKind.SECONDARY,
            card_slots=((target.instance_id, 0),),
        )
        response = planner.decide(request)
        self.assertEqual(response.outcome, DecisionOutcome.DECIDE)
        self.assertEqual(response.action, GameAction(kind=ActionKind.HOLD_CARD, card_instance_id=target.instance_id, source_slot=0))
        assert response.action is not None
        resolved = kernel.transition(
            played,
            replace(response.action, source_slot=None),
            plan_type=request.plan_type,
        )[0].state
        self.assertIsNone(resolved.pending_choice)
        self.assertEqual(resolved.deck.held, (target,))

        no_target_deck = DeckBelief(
            all_cards=(hold,),
            hand=(hold,),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        no_target = replace(state, deck=no_target_deck)
        legal = kernel.legal_actions(
            no_target,
            plan_type=request.plan_type,
            decision_kind=self.suite.cases[0].request.decision_kind,
        )
        self.assertNotIn(hold.instance_id, {action.card_instance_id for action in legal})

    def test_secondary_continuation_runs_only_after_selection(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("continuation-source", "528")
        target = CardRef("continuation-target", "370")
        next_draw = CardRef("continuation-draw", "374")
        deck = DeckBelief(
            all_cards=(source, target, next_draw),
            hand=(source,),
            draw_pile=(target, next_draw),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=self.suite.cases[0].request.plan_type,
        )[0].state
        self.assertIsNotNone(played.pending_choice)
        assert played.pending_choice is not None
        self.assertEqual(played.pending_choice.continuation_index, 1)
        self.assertEqual(played.actions_remaining, 0)
        self.assertEqual(played.deck.hand, ())
        self.assertEqual(played.deck.draw_pile, (target, next_draw))

        resolved = kernel.transition(
            played,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=target.instance_id),
            plan_type=self.suite.cases[0].request.plan_type,
        )
        self.assertEqual(len(resolved), 1)
        branch = resolved[0]
        self.assertIsNone(branch.state.pending_choice)
        self.assertEqual(branch.state.deck.held, (target,))
        self.assertEqual(branch.state.deck.hand, (next_draw,))
        self.assertEqual(branch.state.actions_remaining, 1)
        self.assertTrue(branch.state.deck.conservation_valid)
        self.assertEqual(
            [event.event_type for event in branch.events],
            ["RESOLVE_SECONDARY", "DRAW", "EXTRA_ACTION"],
        )

    def test_secondary_continuation_keeps_pre_card_condition_snapshot(self) -> None:
        definition = CardDefinition(
            card_id="conditional-continuation",
            name="Conditional continuation probe",
            plan=PlanType.ANOMALY,
            effects=(
                EffectSpec(EffectKind.HOLD_SELECTED, target_zones=(CardZone.HAND,)),
                EffectSpec(EffectKind.STANCE, stance=Stance.CONSERVE_1),
                EffectSpec(
                    EffectKind.FULL_POWER,
                    amount=3,
                    condition=EffectCondition(ConditionKind.STANCE_FAMILY, value="strength"),
                ),
            ),
        )
        kernel = PythonNiaProKernel({definition.card_id: definition})
        source = CardRef("conditional-source", definition.card_id)
        target = CardRef("conditional-target", definition.card_id)
        deck = DeckBelief(all_cards=(source, target), hand=(source, target), draw_pile=())
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stance=Stance.AGGRESSIVE_1,
            full_power=0,
            cumulative_full_power=0,
            actions_remaining=1,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert played.pending_choice is not None
        self.assertEqual(played.pending_choice.condition_stance, Stance.AGGRESSIVE_1)
        resolved = kernel.transition(
            played,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=target.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertEqual(resolved.stance, Stance.CONSERVE_1)
        self.assertEqual(resolved.full_power, 3)

    def test_required_two_card_hold_resolves_as_two_explicit_decisions(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("required-hold-source", "527")
        first = CardRef("required-hold-first", "370")
        second = CardRef("required-hold-second", "374")
        deck = DeckBelief(
            all_cards=(source, first, second),
            hand=(source,),
            draw_pile=(first,),
            discard=(second,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=4,
            max_stamina=30,
            genki=0,
            full_power=0,
            cumulative_full_power=0,
            actions_remaining=1,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert played.pending_choice is not None
        self.assertEqual(played.pending_choice.selections_remaining, 2)
        self.assertFalse(played.pending_choice.selection_optional)
        legal = kernel.legal_actions(
            played,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.SECONDARY,
        )
        self.assertNotIn(GameAction(ActionKind.FINISH_SELECTION), legal)

        partial = kernel.transition(
            played,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=first.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        assert partial.state.pending_choice is not None
        self.assertEqual(partial.state.pending_choice.selections_remaining, 1)
        self.assertEqual(partial.state.pending_choice.candidate_instance_ids, (second.instance_id,))
        self.assertEqual(partial.state.deck.held, (first,))
        self.assertEqual(partial.events[0].event_type, "RESOLVE_SECONDARY_PARTIAL")

        resolved = kernel.transition(
            partial.state,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=second.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNone(resolved.pending_choice)
        self.assertEqual(resolved.deck.held, (first, second))
        self.assertEqual(resolved.full_power, 10)
        self.assertTrue(resolved.deck.conservation_valid)

        insufficient_deck = DeckBelief(
            all_cards=(source, first),
            hand=(source,),
            draw_pile=(first,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        insufficient = replace(state, deck=insufficient_deck)
        legal = kernel.legal_actions(
            insufficient,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.CARD,
        )
        self.assertNotIn(source.instance_id, {action.card_instance_id for action in legal})

    def test_optional_two_card_hold_can_finish_early_and_evicts_oldest_at_limit(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("optional-hold-source", "754")
        old = CardRef("optional-hold-old", "370")
        first = CardRef("optional-hold-first", "374")
        second = CardRef("optional-hold-second", "376")
        deck = DeckBelief(
            all_cards=(source, old, first, second),
            hand=(source,),
            draw_pile=(first,),
            discard=(second,),
            held=(old,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            stamina=20,
            max_stamina=30,
            genki=0,
            full_power=2,
            cumulative_full_power=2,
            actions_remaining=1,
        )
        played = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        assert played.pending_choice is not None
        self.assertEqual(played.pending_choice.selections_remaining, 2)
        self.assertTrue(played.pending_choice.selection_optional)
        legal = kernel.legal_actions(
            played,
            plan_type=PlanType.ANOMALY,
            decision_kind=DecisionKind.SECONDARY,
        )
        self.assertIn(GameAction(ActionKind.FINISH_SELECTION), legal)

        partial = kernel.transition(
            played,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=first.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        early = kernel.transition(
            partial,
            GameAction(ActionKind.FINISH_SELECTION),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIsNone(early.pending_choice)
        self.assertEqual(
            tuple(card.instance_id for card in early.deck.held),
            (old.instance_id, first.instance_id),
        )
        early_cards = {card.instance_id: card for card in early.deck.all_cards}
        self.assertEqual(early_cards[old.instance_id].score_repeat_bonus, 1)
        self.assertEqual(early_cards[first.instance_id].stamina_cost_delta, -2)
        self.assertEqual(early_cards[second.instance_id].score_repeat_bonus, 0)
        self.assertEqual(len(early.scheduled_effects), 1)

        full = kernel.transition(
            partial,
            GameAction(ActionKind.HOLD_CARD, card_instance_id=second.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertIsNone(full.state.pending_choice)
        self.assertEqual(
            tuple(card.instance_id for card in full.state.deck.held),
            (first.instance_id, second.instance_id),
        )
        self.assertIn(old.instance_id, {card.instance_id for card in full.state.deck.discard})
        self.assertIn("HOLD_LIMIT_EVICT", {event.event_type for event in full.events})
        full_cards = {card.instance_id: card for card in full.state.deck.all_cards}
        self.assertEqual(full_cards[old.instance_id].score_repeat_bonus, 0)
        self.assertEqual(full_cards[first.instance_id].score_repeat_bonus, 1)
        self.assertEqual(full_cards[second.instance_id].score_repeat_bonus, 1)
        self.assertTrue(full.state.deck.conservation_valid)

    def test_optional_hold_without_candidates_continues_as_no_op(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("empty-optional-source", "754")
        held = CardRef("empty-optional-held", "370")
        deck = DeckBelief(
            all_cards=(source, held),
            hand=(source,),
            draw_pile=(),
            held=(held,),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        state = replace(
            self.suite.cases[0].request.state,
            deck=deck,
            full_power=2,
            cumulative_full_power=2,
            actions_remaining=1,
        )
        branch = kernel.transition(
            state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0]
        self.assertIsNone(branch.state.pending_choice)
        self.assertIn("OPTIONAL_HOLD_EMPTY", {event.event_type for event in branch.events})
        held_after = next(card for card in branch.state.deck.held if card.instance_id == held.instance_id)
        self.assertEqual(held_after.score_repeat_bonus, 1)
        self.assertEqual(held_after.stamina_cost_delta, -2)
        self.assertEqual(len(branch.state.scheduled_effects), 1)

    def test_add_card_to_deck_uses_uniform_positions_and_stable_instance_ids(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=PlanType.ANOMALY)
        kernel = PythonNiaProKernel(catalog.cards)
        source = CardRef("overdrive-source", "702")
        first = CardRef("overdrive-first", "370")
        second = CardRef("overdrive-second", "374")
        ordered_deck = DeckBelief(
            all_cards=(source, first, second),
            hand=(source,),
            draw_pile=(first, second),
            knowledge=DeckKnowledge.KNOWN_ORDER,
        )
        ordered_state = replace(
            self.suite.cases[0].request.state,
            deck=ordered_deck,
            stamina=30,
            max_stamina=30,
            genki=0,
            actions_remaining=1,
            created_card_count=0,
        )
        branches = kernel.transition(
            ordered_state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )
        self.assertEqual(len(branches), 3)
        self.assertTrue(all(abs(branch.probability - 1 / 3) < 1e-12 for branch in branches))
        self.assertEqual(
            {
                tuple(card.instance_id for card in branch.state.deck.draw_pile)
                for branch in branches
            },
            {
                ("task275-created-0", first.instance_id, second.instance_id),
                (first.instance_id, "task275-created-0", second.instance_id),
                (first.instance_id, second.instance_id, "task275-created-0"),
            },
        )
        for branch in branches:
            self.assertEqual(branch.state.created_card_count, 1)
            self.assertTrue(branch.state.deck.conservation_valid)
            created = next(
                card for card in branch.state.deck.all_cards if card.instance_id == "task275-created-0"
            )
            self.assertEqual(created.card_id, "23")
            self.assertIn("ADD_CARD_TO_DECK", {event.event_type for event in branch.events})

        second_source = CardRef("overdrive-source-2", "702")
        composition_deck = DeckBelief(
            all_cards=(source, second_source),
            hand=(source, second_source),
            draw_pile=(),
            knowledge=DeckKnowledge.KNOWN_COMPOSITION,
        )
        composition_state = replace(
            ordered_state,
            deck=composition_deck,
            actions_remaining=1,
        )
        first_play = kernel.transition(
            composition_state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=source.instance_id),
            plan_type=PlanType.ANOMALY,
        )
        self.assertEqual(len(first_play), 1)
        second_play = kernel.transition(
            first_play[0].state,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=second_source.instance_id),
            plan_type=PlanType.ANOMALY,
        )
        self.assertEqual(len(second_play), 1)
        final_state = second_play[0].state
        self.assertEqual(final_state.created_card_count, 2)
        self.assertEqual(
            {card.instance_id for card in final_state.deck.draw_pile},
            {"task275-created-0", "task275-created-1"},
        )
        self.assertTrue(final_state.deck.conservation_valid)

        generated = next(card for card in final_state.deck.draw_pile if card.instance_id == "task275-created-0")
        remaining_draw = tuple(card for card in final_state.deck.draw_pile if card.instance_id != generated.instance_id)
        playable_deck = final_state.deck.replace_zones(
            hand=(generated,),
            draw_pile=remaining_draw,
        )
        playable = replace(final_state, deck=playable_deck, actions_remaining=1)
        removed = kernel.transition(
            playable,
            GameAction(ActionKind.PLAY_CARD, card_instance_id=generated.instance_id),
            plan_type=PlanType.ANOMALY,
        )[0].state
        self.assertIn(generated, removed.deck.removed)
        self.assertTrue(removed.deck.conservation_valid)

    def test_dynamic_upgrade_uses_real_upgraded_business_id(self) -> None:
        catalog = compile_skill_card_catalog(SKILL_CARDS, plan_type=self.suite.cases[0].request.plan_type)
        supported_pairs = sorted(catalog.upgrade_map.items())
        self.assertGreaterEqual(len(supported_pairs), 5)
        base_ids = tuple(base for base, _upgraded in supported_pairs[:5])
        generator = NiaProDynamicDeckGenerator(
            catalog.cards,
            plan_type=self.suite.cases[0].request.plan_type,
            base_card_ids=base_ids,
            acquisition_pool=base_ids,
            upgrade_map=catalog.upgrade_map,
            minimum_deck_size=5,
            mutation_count_range=(20, 20),
        )
        samples = generator.generate(seed=5000, episode_count=10)
        upgrade_events = [
            event
            for sample in samples
            for event in sample.mutation_events
            if event.kind.value == "UPGRADE"
        ]
        self.assertTrue(upgrade_events)
        for event in upgrade_events:
            self.assertEqual(catalog.upgrade_map[event.previous_card_id], event.card_id)


if __name__ == "__main__":
    unittest.main()
