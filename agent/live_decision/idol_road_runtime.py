"""Tower rule state, bound to Maa's visible hand, choices and resources."""
from collections import Counter
from dataclasses import replace, dataclass

from .high_score.kernel import RuleEvent
from .high_score.planner import PlannerConfig, RobustHighScorePlanner
from .high_score.contracts import (
    CardRef,
    ActionKind,
    DeckBelief,
    GameAction,
    DecisionKind,
    DeckKnowledge,
    StrategyState,
    DecisionRequest,
    CapabilityReport,
)
from .idol_road_card_rules import TowerCardState, TowerCardKernel
from .high_score.hif_environment import HifStep

ATTRIBUTES = ('vocal', 'dance', 'visual')
_READY = CapabilityReport(True, True, True, True, True)


@dataclass(frozen=True)
class TowerFrame:
    frame_id: str
    turns: int
    attribute: str
    score: int
    stamina: int
    genki: int
    hand: tuple[str, ...]
    drinks: tuple[tuple[int, str], ...]
    playable: tuple[int, ...]
    available_drinks: tuple[int, ...] | None = None
    end_available: bool = True
    skip_recovery: int | None = None
    statuses: tuple[tuple[str, int], ...] = ()


def _observed_fields(frame):
    return {field: value for field, value in frame.statuses
            if field in ('good_impression_turns', 'motivation', 'concentration', 'good_condition_turns')}


class TowerChoicePolicy:
    """Use the common rules for secondary actions UINet cannot emit."""

    def __init__(self, kernel, **_options):
        self.planner = RobustHighScorePlanner(kernel, PlannerConfig(
            max_decision_depth=2, timeout_ms=float('inf'), require_terminal_leaf=False))

    def choose(self, request, legal):
        if len(legal) == 1:
            return legal[0]
        selected = self.planner.decide(request).action
        matches = [action for action in legal if selected is not None and
                   replace(action, source_slot=None) == replace(selected, source_slot=None)]
        if len(matches) != 1:
            raise ValueError('idol_road_secondary_planner_no_bound_choice')
        return matches[0]

    @staticmethod
    def committed(_before, _action, _after):
        pass


def bind_tower_slots(cards, visible):
    """Assign observed ID/upgrade per physical occurrence within one zone."""
    remaining = list(cards)
    result = []
    for slot, card_id in enumerate(visible):
        matches = [card for card in remaining if card.card_id == card_id]
        if not matches or len({card.semantic_key for card in matches}) != 1:
            raise ValueError(f'idol_road_unresolved_card_slot:{slot}:{card_id}')
        card = matches[0]
        remaining.remove(card)
        result.append((card.instance_id, slot))
    return tuple(result)


def tower_schedule(attribute, turns, multipliers, *, timeline=None, current_turn=1):
    """Supply a planning forecast while retaining the observed current attribute."""
    trend = sorted(ATTRIBUTES, key=lambda name: multipliers[ATTRIBUTES.index(name)], reverse=True)
    future = ([trend[0]] * max(0, turns - 3) + trend[::-1])[-turns:]
    if timeline:
        future = [timeline[turn].attribute if turn in timeline and timeline[turn].attribute else forecast
                  for turn, forecast in enumerate(future, current_turn)]
    return (attribute, *future[1:])


def _turn_fields(attributes, multipliers):
    values = tuple(multipliers[ATTRIBUTES.index(attr)] / 1000 for attr in attributes)
    return dict(turn_attributes=attributes, score_multiplier=values[0], future_score_multipliers=values[1:])


def tower_opening(kernel, loadout, frame, deck_ids, *, status=None, item_uses=None, timeline=None, native=False,
                  partial_deck=False):
    """Build the first request from the visible formation and opening HUD."""
    info = loadout.overview
    if (info.turns != frame.turns or info.max_stamina is None or info.multiplier_permils is None
            or any(cid not in kernel.cards for cid in deck_ids)):
        raise ValueError('idol_road_opening_formation_and_hud_incomplete')
    refs = tuple(CardRef(f'formation:{i}', cid, kernel.cards[cid].base_card_id not in (None, cid))
                 for i, cid in enumerate(deck_ids))
    slots = bind_tower_slots(refs, frame.hand)
    by_id = {card.instance_id: card for card in refs}
    hand = tuple(by_id[iid] for iid, _ in slots)
    schedule = tower_schedule(frame.attribute, frame.turns, info.multiplier_permils, timeline=timeline)
    item_uses = item_uses or {}
    state_type = TowerCardState if isinstance(kernel, TowerCardKernel) else StrategyState
    state = state_type(turn=1, turns_remaining=frame.turns, score=frame.score, stamina=frame.stamina,
        max_stamina=info.max_stamina, genki=frame.genki, actions_remaining=1, hand_limit=5,
        deck=DeckBelief(refs, hand, tuple(card for card in refs if card not in hand),
                        knowledge=DeckKnowledge.UNKNOWN if partial_deck else DeckKnowledge.KNOWN_COMPOSITION),
        drinks=tuple(identifier for _, identifier in frame.drinks), items=tuple(item_uses),
        **_turn_fields(schedule, info.multiplier_permils), **{**(status or {}), **_observed_fields(frame)})
    if native:
        return DecisionRequest(frame.frame_id, loadout.idol.plan, DecisionKind.LIVE, state,
            _READY, ruleset_id=kernel.ruleset_id, card_slots=slots, drink_slots=frame.drinks)
    state = kernel.initialize_stage_entry(state, plan_type=loadout.idol.plan)
    if item_uses:
        scheduled = []
        for ref in state.scheduled_effects:
            item = ref.source_id.removeprefix('item:')
            if ref.source_id.startswith('item:') and ref.remaining_uses is not None:
                uses = item_uses[item]
                if type(uses) is not int or not 0 <= uses <= ref.remaining_uses:
                    raise ValueError('idol_road_item_use_count_unread:' + item)
                ref = replace(ref, remaining_uses=uses)
            scheduled.append(ref)
        state = replace(state, scheduled_effects=tuple(scheduled))
    return DecisionRequest(frame.frame_id, loadout.idol.plan, DecisionKind.LIVE, state,
        _READY, ruleset_id=kernel.ruleset_id, card_slots=slots, drink_slots=frame.drinks)


def tower_legal(kernel, request, frame):
    actions = kernel.legal_actions(request.state, plan_type=request.plan_type, decision_kind=request.decision_kind)
    slots = dict(request.card_slots)
    result = []
    for action in actions:
        if action.kind is ActionKind.USE_DRINK:
            result.extend(replace(action, source_slot=slot) for slot, cid in request.drink_slots
                          if cid == action.drink_id and (frame.available_drinks is None
                                                       or slot in frame.available_drinks))
        elif action.kind is ActionKind.PLAY_CARD:
            slot = slots[action.card_instance_id]
            if slot in frame.playable:
                result.append(replace(action, source_slot=slot))
        elif action.kind is not ActionKind.END_TURN or frame.end_available:
            result.append(action)
    return tuple(result)


def tower_branches(kernel, before, action, multipliers, *, observed=None, skip_recovery=0, timeline=None):
    """Project rule consequences, including the game's automatic end of turn."""
    def with_next_attribute(state):
        if state.turns_remaining < 2:
            return state
        turn = state.turn + 1
        if observed is not None and observed.turns <= state.turns_remaining:
            attribute = observed.attribute
        elif timeline and turn in timeline and timeline[turn].attribute:
            attribute = timeline[turn].attribute
        else:
            return state
        updated = tower_schedule(attribute, state.turns_remaining - 1, multipliers, timeline=timeline, current_turn=turn)
        return replace(state, **_turn_fields((state.turn_attributes[0], *updated), multipliers))

    initial = with_next_attribute(before.state) if action.kind is ActionKind.END_TURN else before.state
    if action.kind is ActionKind.END_TURN and skip_recovery:
        initial = replace(initial, stamina=min(initial.max_stamina, initial.stamina + skip_recovery))
    result = []
    for branch in kernel.transition(initial, action, plan_type=before.plan_type):
        state = branch.state
        if (action.kind is ActionKind.END_TURN or state.terminal or state.actions_remaining
                or state.pending_choice is not None or state.resolution_stack):
            result.append(branch)
            continue
        for end in kernel.transition(with_next_attribute(state), GameAction(ActionKind.END_TURN),
                                     plan_type=before.plan_type):
            result.append(replace(end, probability=branch.probability * end.probability,
                                  events=branch.events + end.events))
    return tuple(result)


def tower_live_step(before, branches, frame):
    """Match an actual post-action hand and resources, keeping displayed score."""
    matches = {}
    for branch in branches:
        state = branch.state
        if (state.terminal or state.pending_choice is not None
                or not state.turn_attributes or state.turn_attributes[0] != frame.attribute
                or (state.turns_remaining, state.stamina, state.genki) != (frame.turns, frame.stamina, frame.genki)
                or Counter(card.card_id for card in state.deck.hand) != Counter(frame.hand)
                or Counter(state.drinks) != Counter(cid for _, cid in frame.drinks)):
            continue
        slots = bind_tower_slots(state.deck.hand, frame.hand)
        observed = _observed_fields(frame)
        corrections = tuple(RuleEvent('TOWER_OBSERVED_VALUE', 'hud', value - getattr(state, field), field)
                            for field, value in observed.items() if value != getattr(state, field))
        if isinstance(state, TowerCardState) and observed.get('good_impression_turns') == 0:
            observed['review_fresh'] = False
        if observed.get('good_condition_turns') == 0:
            observed['good_condition_fresh'] = False
        state = replace(state, score=frame.score, **observed)
        after = replace(before, snapshot_id=frame.frame_id, state=state, decision_kind=DecisionKind.LIVE,
            card_slots=slots, drink_slots=frame.drinks)
        events = branch.events + corrections
        matches[(after, branch.events)] = HifStep(after, frame.score - before.state.score, False, events)
    return _merge_hidden_draw_positions(tuple(matches.values()))


def _merge_hidden_draw_positions(steps):
    if len(steps) <= 1:
        return steps[0] if steps else None
    decks = tuple(step.request.state.deck for step in steps)
    if any(deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION for deck in decks):
        return None
    first = decks[0]
    prefix = []
    for index in range(min(deck.known_top_count for deck in decks)):
        card = first.draw_pile[index]
        if any(deck.draw_pile[index] != card for deck in decks[1:]):
            break
        prefix.append(card)

    def normalize_event(event):
        if event.event_type == 'ADD_CARD_TO_DECK' and event.detail:
            return replace(event, detail=event.detail.split(':position:', 1)[0])
        return event

    normalized = []
    for step, deck in zip(steps, decks, strict=True):
        unknown = tuple(sorted(deck.draw_pile[len(prefix):], key=lambda card: card.instance_id))
        shared = replace(deck, draw_pile=(*prefix, *unknown), known_top_count=len(prefix))
        request = replace(step.request, state=replace(step.request.state, deck=shared))
        normalized.append(replace(step, request=request, events=tuple(map(normalize_event, step.events))))
    if any(candidate != normalized[0] for candidate in normalized[1:]):
        return None
    # Keep a known generated-card position only when every possible branch put it there.
    events = []
    for index, observed in enumerate(normalized[0].events):
        first_event = steps[0].events[index]
        if all(step.events[index] == first_event for step in steps[1:]):
            observed = first_event
        events.append(observed)
    return replace(normalized[0], events=tuple(events))


def tower_observed_extra_action(kernel, before, action, frame):
    """The game may leave a playable hand after the forecast spent its final action."""
    if (action.kind is not ActionKind.PLAY_CARD or before.state.turns_remaining != frame.turns
            or not frame.playable):
        return None
    candidates = []
    for branch in kernel.transition(before.state, action, plan_type=before.plan_type):
        state = branch.state
        if state.actions_remaining or state.pending_choice is not None or state.resolution_stack:
            continue
        observed = replace(state, actions_remaining=1)
        event = RuleEvent('TOWER_OBSERVED_VALUE', 'hud', 1, 'actions_remaining')
        candidates.append(replace(branch, state=observed, events=branch.events + (event,)))
    return tower_live_step(before, candidates, frame)


def tower_choice_step(before, branches, frame_id, cards, *, drink_slots):
    """Bind a multi-step action to the displayed selection grid."""
    matches = {}
    for branch in branches:
        state = branch.state
        pending = state.pending_choice
        if pending is None or Counter(state.drinks) != Counter(cid for _, cid in drink_slots):
            continue
        by_id = {card.instance_id: card for card in state.deck.all_cards}
        choices = tuple(by_id[iid] for iid in pending.candidate_instance_ids)
        if Counter(card.card_id for card in choices) != Counter(cards):
            continue
        slots = bind_tower_slots(choices, cards)
        after = replace(before, snapshot_id=frame_id, state=state, decision_kind=DecisionKind.SECONDARY,
            card_slots=slots, drink_slots=drink_slots)
        matches[(after, branch.events)] = HifStep(after, state.score - before.state.score, False, branch.events)
    return next(iter(matches.values())) if len(matches) == 1 else None


def tower_terminal_step(before, branches, frame_id, score, drink_slots):
    """Ranking is the terminal proof and supplies the official reward."""
    matches = {}
    for branch in branches:
        if branch.state.terminal:
            after = replace(before, snapshot_id=frame_id, state=replace(branch.state, score=score),
                decision_kind=DecisionKind.LIVE, card_slots=(), drink_slots=drink_slots)
            matches[(after, branch.events)] = (HifStep(after, score - before.state.score, True, branch.events),
                                               branch.state.score)
    return next(iter(matches.values())) if len(matches) == 1 else None
