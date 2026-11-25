"""Hiro's battle simulator behind the Maa receipt and UINet decision interfaces."""

import json
from collections import Counter
from dataclasses import replace, dataclass

from .exam_loop import ExamResult
from .drink_icons import EVENT_GREEN_JUICE_ID
from .idol_road_battle import _wire_card
from .idol_road_runtime import ATTRIBUTES, _turn_fields, _observed_fields, bind_tower_slots
from .high_score.contracts import CardRef, ActionKind, DeckBelief, GameAction, DecisionKind, DeckKnowledge
from .high_score.tower_gnrpa import _deck_entry, load_native_tower
from .high_score.hif_environment import HifStep
from .high_score.gkms_master_forms import PLAN_CODES, drink_id
from .high_score.gkms_continuous_policy import verify_model_output


def native_v2(uiobs, snapshot, exam, legal, *, customize_ids=None):
    """Use Hiro's native encoder for the shared fields and preserve V2 rule records."""
    observation = uiobs.from_engine(snapshot, exam, legal, customize_ids=customize_ids)
    observation['v'] = 'uiobs.v2'
    observation.update(timers=[], costs=[], pcount=[])
    enchants = []
    for status in snapshot['status']['effects']:
        trig = status.get('trig')
        if trig:
            trigger = trig['trigger']
            phases = trigger['phaseTypes']
            if phases in ([21], [42]):
                target, = trigger['phaseValues']
                remaining = target - trig['turn_count']
                if remaining >= 0:
                    observation['timers'].extend([remaining, effect['id'], int(phases == [42])]
                                                 for effect in trig['effects'])
            else:
                enchants.extend(uiobs.enchants({'status': {'effects': [status]}}))
        cls = status['cls']
        if cls == 'SearchPlayCardStaminaConsumptionChangeStatusEffect':
            extra = status['extra']
            observation['costs'].append([extra.get('_searchId') or None, status['value'], extra['_count']])
        elif cls == 'PlayCountBuffStatusEffect':
            extra = status['extra']
            observation['pcount'].append([extra['_searchId'], status['value'], extra['_count'], uiobs._turns(status)])
    observation['ench'] = enchants
    return observation


class NativeTowerRules:
    """Prepare a fixed native master and a fresh scenario without assuming an unseen hand."""

    def __init__(self, kernel, stage, *, items):
        rust, tables = load_native_tower()
        if not hasattr(rust.Exam, 'begin_live_choices'):
            raise ValueError('idol_road_native_choice_runtime_missing')
        from hiro.bench_bridge import uiobs
        from gakumas_exam.setup import new_exam
        from gakumas_exam.native import to_native
        from hiro.search.loadout import parse_deck_entry
        from hiro.search.exam_env import _master

        self.master = _master()
        self.uiobs = uiobs
        self.new_exam = new_exam
        self.to_native = to_native
        self.parse_deck_entry = parse_deck_entry
        self.stage = stage
        self.kernel = kernel
        self.cards = {}
        for cid, card in kernel.cards.items():
            try:
                form = _wire_card(kernel, cid, card.name.endswith('+'), native=True)
            except ValueError:
                continue
            self.cards.setdefault(form, []).append(cid)
        official_items = tuple(tables.master().rows('ProduceItem'))
        self.items = []
        for key, name in items.items():
            if any(row['id'] == key for row in official_items):
                self.items.append((key, key))
                continue
            candidates = {row['id'] for row in official_items if row['name'] == name
                          and not row['id'].startswith('pitem_tower_')}
            if len(candidates) != 1:
                raise ValueError(f'idol_road_native_memory_item_unmapped:{key}:{name}')
            self.items.append((key, candidates.pop()))
        self.drink_forms = {cid: form for cid, definition in kernel.drinks.items()
                            if (form := drink_id(definition, native=True)) is not None}
        if any(row['id'] == EVENT_GREEN_JUICE_ID for row in tables.master().rows('ProduceDrink')):
            self.drink_forms[EVENT_GREEN_JUICE_ID] = EVENT_GREEN_JUICE_ID

    def open(self, opening, loadout, model, observed_types, *, item_uses, statuses=()):
        from gakumas_exam.cards import ExamCard

        opening.validate()
        initial = opening.state
        deck = []
        for ref in initial.deck.all_cards:
            identifier, up, custom = self.parse_deck_entry(_deck_entry(ref, self.kernel, native=True))
            card = ExamCard(self.master, identifier, up, guid=ref.instance_id)
            card.custom_override = [{'id': cid, 'count': count} for cid, count in custom.items()] or None
            deck.append(card)
        attributes = [ATTRIBUTES.index(attr) + 1 for attr in initial.turn_attributes]
        exam = self.new_exam(None, self.master, 0, deck=deck, max_stamina=initial.max_stamina,
            stamina=initial.max_stamina, plan_type=PLAN_CODES[opening.plan_type.value],
            turn_types=attributes, bonus_permil=dict(enumerate(loadout.overview.multiplier_permils, 1)),
            items=[wire for _, wire in self.items], gimmicks=[self.stage.identity] if self.stage else (),
            drinks=[self.drink_forms[cid] for cid in initial.drinks], exam_type=getattr(loadout, 'exam_type', 1))
        for local_id, wire_id in self.items:
            scheduled = [effect for effect in exam.status.effects
                         if getattr(effect, 'source', {}).get('item') == wire_id]
            uses = item_uses[local_id]
            if uses is not None and len(scheduled) == 1 and scheduled[0].limit_count >= 0:
                scheduled[0].limit_count = uses
        native = self.to_native(exam)
        native.advance()
        # Replace only the simulated opening deal. The inventory and its physical instances
        # came from formation, and the unseen suffix is still only a composition.
        hand = [card.instance_id for card in initial.deck.hand]
        pool = native.zone('hand') + native.zone('deck')
        for guid in hand:
            pool.remove(guid)
        native.set_zone('hand', hand)
        native.set_zone('deck', pool)
        _read_resources(native, initial.score, initial.stamina, initial.genki, dict(statuses))
        return NativeTowerSession(self, native, model, loadout, observed_types)


_OBSERVED_STATUS = {
    'concentration': ('LessonBuffStatusEffect', False),
    'good_condition_turns': ('ParameterBuffStatusEffect', True),
}
_REVIEW_MULTIPLIER = 'ReviewMultipleStatusEffect'


def _read_resources(rx, score, stamina, genki, statuses, effects=None):
    rx.score, rx.stamina, rx.block = score, stamina, genki
    if 'good_impression_turns' in statuses and rx.review != statuses['good_impression_turns']:
        rx.review = statuses['good_impression_turns']
    if 'motivation' in statuses and rx.aggressive != statuses['motivation']:
        rx.aggressive = statuses['motivation']
    for field, (rule, by_turn) in _OBSERVED_STATUS.items():
        if field in statuses:
            rx.override_status(rule, by_turn, statuses[field])
    if 'review_score_percent' in statuses:
        if effects is None:
            effects = json.loads(rx.export_json())['status']['effects']
        layers = [entry for entry in effects if entry['cls'] == _REVIEW_MULTIPLIER]
        # The HUD exposes the sum, not the expiry of each layer. Keep the
        # existing duration and source only when the value has one owner.
        if len(layers) == 1:
            observed = statuses['review_score_percent'] * 10
            if layers[0]['value'] != observed:
                rx.override_status(_REVIEW_MULTIPLIER, False, observed)


@dataclass(frozen=True)
class NativeChoice:
    effect: str
    effect_type: int
    destination: int
    minimum: int
    maximum: int
    cards: tuple[CardRef, ...]
    source_zones: tuple[str | None, ...]


@dataclass(frozen=True)
class ChoiceReceipt:
    effect: str
    slots: tuple[tuple[int, str], ...]
    frame_id: str
    job_id: int


class NativeTowerSession:
    """Only a matching host receipt may replace the committed Rust position."""

    def __init__(self, rules, native, model, loadout, observed_types):
        self.rules, self.native, self.model = rules, native, model
        self.search = None
        self.observation_version = 2
        self.unread_formation_cards = 0
        self.multipliers = loadout.overview.multiplier_permils
        self.exam = {'produce': 'tower', 'step': None, 'plan': 0, 'crit': [0, 0, 0],
                     'idol': loadout.idol.idol_card_id}
        self.observed_types = observed_types
        self.candidate = None
        self.actions = {}
        self.choice_receipts = []

    def record_choice(self, choice, receipt):
        instance_ids = {card.instance_id for card in choice.cards}
        slots, selected = zip(*receipt.slots) if receipt.slots else ((), ())
        if (receipt.effect != choice.effect or not receipt.frame_id or receipt.job_id <= 0
                or any(old.job_id == receipt.job_id for old in self.choice_receipts)
                or len(set(slots)) != len(slots) or min(slots, default=0) < 0
                or len(set(selected)) != len(selected) or not instance_ids.issuperset(selected)
                or not choice.minimum <= len(selected) <= choice.maximum):
            raise ValueError('idol_road_native_invalid_choice_receipt')
        self.choice_receipts.append(receipt)

    def pending_choice(self, before, action):
        trial = self._fork(before, action, None)
        return trial if isinstance(trial, NativeChoice) else None

    def overlapping_hand_forms(self, before, action, frame):
        trial = self._fork(before, action, frame)
        if (isinstance(trial, NativeChoice) or trial.ended or trial.remain_turn != frame.turns
                or len(trial.zone('hand')) != len(frame.hand)):
            return ()
        refs = self._cards(trial, before.state, frame.hand)
        return {refs[guid].card_id for zone in ('hand', 'deck', 'grave', 'hold') for guid in trial.zone(zone)}

    def drawable_card_forms(self):
        """Forms in visible or drawable zones, including a reshuffled discard pile."""
        drawable = {guid for zone in ('hand', 'deck', 'hold', 'grave') for guid in self.native.zone(zone)}
        return {form for guid, card_id, upgrade, _ in self.native.cards_info() if guid in drawable
                for form in self.rules.cards.get((card_id, upgrade), ())}

    def _observation(self, legal):
        uiobs = self.rules.uiobs

        def customize_ids(identifier, grade):
            return self.rules.master.card(identifier, grade).get('produceCardCustomizeIds') or ()

        snapshot = json.loads(self.native.export_json())
        if self.observation_version == 1:
            result = uiobs.from_engine(snapshot, self.exam, legal, customize_ids=customize_ids)
        else:
            result = native_v2(uiobs, snapshot, self.exam, legal, customize_ids=customize_ids)
        if self.exam.get('lesson'):
            result['exam'].update(types=[], permil=[0, 0, 0])
        elif self.exam['produce'] == 'tower':
            result['exam']['types'] = list(self.observed_types)
        return result

    def legal_actions(self, request, frame):
        self.actions = {}
        by_instance = dict(request.card_slots)
        order = self.native.zone('hand')
        for token in self.native.legal_actions():
            kind = token[0]
            if kind == 'play':
                instance = order[token[1]]
                slot = by_instance[instance]
                if slot in frame.playable:
                    self.actions[tuple(token)] = GameAction(ActionKind.PLAY_CARD, instance, source_slot=slot)
            elif kind == 'drink':
                slot, cid = request.drink_slots[token[1]]
                if frame.available_drinks is None or slot in frame.available_drinks:
                    self.actions[tuple(token)] = GameAction(ActionKind.USE_DRINK, drink_id=cid, source_slot=slot)
            elif kind == 'end' and frame.end_available:
                self.actions[('end',)] = GameAction(ActionKind.END_TURN)
        return tuple(self.actions.values())

    def choose(self, request, legal):
        tokens = tuple(token for token, action in self.actions.items() if action in legal)
        if len(tokens) == 1:
            return self.actions[tokens[0]]
        observation = self._observation(tokens)
        if self.search is not None:
            token = self.search(self.native, observation)
            if token not in tokens:
                raise ValueError('idol_road_native_search_action_not_visible')
            return self.actions[token]
        token = verify_model_output(self.model.decide(observation), observation)
        return self.actions[tuple(token)]

    def sync_item_uses(self, uses):
        """Use bound HUD charges for limited items, including field-granted uses."""
        if not uses:
            return {}
        from gakumas_exam.native import master_path

        visible = {wire: (key, uses[key]) for key, wire in self.rules.items if key in uses}
        state = json.loads(self.native.export_json())
        triggers = [entry['trig'] for entry in state['status']['effects'] if entry.get('trig')]
        changes = {}
        for wire, (key, observed) in visible.items():
            effects = [trigger for trigger in triggers if trigger.get('source', {}).get('item') == wire
                       and trigger['limit_count'] >= 0]
            if len(effects) != 1 or effects[0]['limit_count'] == observed:
                continue
            changes[key] = observed - effects[0]['limit_count']
            effects[0]['limit_count'] = observed
        if changes:
            self.native = type(self.native)(master_path(self.rules.master), json.dumps(state))
        return changes

    def _cards(self, rx, before, visible=()):
        previous = {card.instance_id: card for card in before.deck.all_cards}
        return {guid: self._card_ref(guid, identifier, up, previous, visible)
                for guid, identifier, up, _ in rx.cards_info()}

    def _card_ref(self, guid, identifier, up, previous, visible=()):
        options = self.rules.cards.get((identifier, up), ())
        old = previous.get(guid)
        if old is not None and old.card_id in options:
            return old
        candidates = [cid for cid in options if cid in visible] or options
        if len(candidates) != 1:
            raise ValueError(f'idol_road_native_card_identity_unmapped:{identifier}:{up}')
        return CardRef(guid, candidates[0], bool(up))

    def _observed_draw_prefix(self, rx, before, frame, *, played=None):
        refs = self._cards(rx, before.state, frame.hand)
        hand = Counter(refs[guid].card_id for guid in rx.zone('hand') if guid != played)
        needs = Counter(frame.hand) - hand
        deck = rx.zone('deck')
        front = []
        for cid in frame.hand:
            selected = next((guid for guid in deck if refs[guid].card_id == cid), None)
            if needs[cid] and selected is not None:
                needs[cid] -= 1
                deck.remove(selected)
                front.append(selected)
        if front:
            rx.set_zone('deck', front + deck)

    def _fork(self, before, action, frame, *, observed_draws=False, extra_action=False, observed_turn_end=False):
        rx = self.native.clone()
        if extra_action:
            rx.remaining_actions += 1
        rx.begin_live_choices([(receipt.effect, [guid for _, guid in receipt.slots])
                               for receipt in self.choice_receipts])

        def pending():
            choice = rx.pending_choice()
            if choice is None:
                return None
            *rule, candidates = choice
            previous = {card.instance_id: card for card in before.state.deck.all_cards}
            cards = tuple(self._card_ref(guid, wire, up, previous) for guid, wire, up, _ in candidates)
            return NativeChoice(*rule, cards, tuple(zone for _, _, _, zone in candidates))

        if observed_draws:
            self._observed_draw_prefix(rx, before, frame, played=action.card_instance_id if action else None)
        if action is None:
            rx.end_observed_turn()
        else:
            token, = [token for token, bound in self.actions.items() if bound == action]
            rx.step(token)
        if (menu := pending()) is not None:
            return menu
        if observed_turn_end:
            rx.end_observed_turn()
            if (menu := pending()) is not None:
                return menu
        if frame is not None:
            state = json.loads(rx.export_json())
            if state['pending_start']:
                if observed_draws:
                    self._observed_draw_prefix(rx, before, frame)
                types = rx.turn_types
                index = rx.current_turn
                attribute = ATTRIBUTES.index(frame.attribute) + 1
                types += [types[-1]] * max(0, index + 1 - len(types))
                types[index] = attribute
                rx.turn_types = types
        rx.advance()
        if (menu := pending()) is not None:
            return menu
        rx.finish_live_choices()
        if rx.unimplemented > self.native.unimplemented:
            raise ValueError('idol_road_native_effect_not_implemented')
        return rx

    def observe_frame(self, before, action, frame, *, confirmed_card=False):
        """Prepare a successor on a clone, retaining original until DecisionBox commits."""
        predicted = self._fork(before, action, frame)
        if isinstance(predicted, NativeChoice):
            return None
        step = self._match(before, action, frame, predicted)
        if step is None:
            # The next visible hand can constrain a draw. It never establishes the
            # unseen suffix or mutates the previously committed engine.
            predicted = self._fork(before, action, frame, observed_draws=True)
            if isinstance(predicted, NativeChoice):
                return None
            step = self._match(before, action, frame, predicted)
        if step is None and self._align_observed_draw(before, frame, predicted):
            step = self._match(before, action, frame, predicted)
        if (step is None and confirmed_card and action.kind is ActionKind.PLAY_CARD
                and frame.turns == before.state.turns_remaining - 1
                and predicted.current_turn == self.native.current_turn
                and predicted.remain_turn == before.state.turns_remaining
                and action.card_instance_id not in predicted.zone('hand')):
            # Like Hiro's observed-turn alignment, advance the simulation to
            # the visible round after the confirmed card. Use an automatic
            # turn end so the model does not add Skip recovery or an input.
            advanced = self._fork(before, action, frame, observed_draws=True, observed_turn_end=True)
            if not isinstance(advanced, NativeChoice):
                self._align_observed_draw(before, frame, advanced)
                step = self._match(before, action, frame, advanced)
                if step is None and not advanced.ended and advanced.remain_turn == frame.turns:
                    learned = self._align_partial_formation(before, frame, advanced, confirmed_draw=True)
                    if learned:
                        step = self._match(before, action, frame, advanced, learned=learned)
                if step is not None:
                    self.observed_delta['observed_turn_end'] = 1
        if step is None and confirmed_card and self._paid_same_turn_play(before, action, frame, predicted):
            if self._align_observed_draw(before, frame, predicted, rebind_survivors=True):
                step = self._match(before, action, frame, predicted)
                if step is not None:
                    self.observed_delta['hand_rebound'] = 1
        confirmed_draw = (confirmed_card and action.kind is ActionKind.PLAY_CARD
                          and action.card_instance_id not in predicted.zone('hand'))
        if (step is None and (self.unread_formation_cards != 0 or confirmed_draw)
                and not predicted.ended and predicted.remain_turn == frame.turns):
            learned = self._align_partial_formation(before, frame, predicted, confirmed_draw=confirmed_draw)
            if learned:
                step = self._match(before, action, frame, predicted, learned=learned)
        if (step is None and action.kind is ActionKind.PLAY_CARD and (frame.playable or frame.end_available)
                and frame.turns == before.state.turns_remaining
                and (predicted.ended or predicted.current_turn > self.native.current_turn)):
            # The next action can be a SKIP even if the remaining cards are
            # disabled. Keep the observed turn until that action is resolved.
            held = self._fork(before, action, frame, observed_draws=True, extra_action=True)
            if not isinstance(held, NativeChoice) and held.current_turn == self.native.current_turn:
                step = self._match(before, action, frame, held)
                if step is not None:
                    self.observed_delta['actions_remaining'] = 1
        return step

    def _paid_same_turn_play(self, before, action, frame, predicted):
        if action.kind is not ActionKind.PLAY_CARD or predicted.current_turn != self.native.current_turn:
            return False
        old = before.state
        played = next(card for card in old.deck.hand if card.instance_id == action.card_instance_id)
        resources = frame.stamina, frame.genki
        paid = resources != (old.stamina, old.genki)
        scored = frame.score != old.score and frame.score == predicted.score
        return (resources == (predicted.stamina, predicted.block) and (paid or scored)
                and frame.hand.count(played.card_id) < sum(card.card_id == played.card_id for card in old.deck.hand))

    def _align_partial_formation(self, before, frame, predicted, *, confirmed_draw=False):
        """As in Hiro, register an unmapped card only when it has appeared."""
        refs = self._cards(predicted, before.state, frame.hand)
        predicted_hand = predicted.zone('hand')
        pool = predicted_hand + predicted.zone('deck')
        missing = Counter(frame.hand) - Counter(refs[guid].card_id for guid in pool)
        limit = self.unread_formation_cards
        if not self.rules.kernel.cards.keys() >= missing.keys():
            return 0
        if limit is not None and missing.total() > limit:
            # Hiro registers visible cards missing from its tracked pools.
            # Keep the confirmed play and surviving slots while correcting an unseen draw.
            known = {card.card_id for card in refs.values()}
            if not confirmed_draw or known & missing.keys() or len(frame.hand) != len(predicted_hand):
                return 0
            if predicted.current_turn == self.native.current_turn:
                survivors = Counter(card.card_id for card in before.state.deck.hand
                                    if card.instance_id in predicted_hand)
                if survivors - Counter(frame.hand):
                    return 0
        learned = 0
        hand = []
        for card_id in frame.hand:
            observed = next((guid for guid in pool if refs[guid].card_id == card_id), None)
            if observed is None:
                card = self.rules.kernel.cards[card_id]
                upgraded = card.base_card_id not in (None, card_id)
                wire_id, grade = _wire_card(self.rules.kernel, card_id, upgraded, native=True)
                observed = predicted.new_card(wire_id, grade)
                learned += 1
            else:
                pool.remove(observed)
            hand.append(observed)
        if learned:
            predicted.set_zone('hand', hand)
            predicted.set_zone('deck', pool)
        return learned

    def _align_observed_draw(self, before, frame, predicted, *, rebind_survivors=False):
        """Bind the visible hand from tracked pools after the action resolves."""
        hand, deck = predicted.zone('hand'), predicted.zone('deck')
        if predicted.ended or predicted.remain_turn != frame.turns:
            return False
        shortfall = len(hand) - len(frame.hand)
        if shortfall:
            if (predicted.current_turn != self.native.current_turn + 1
                    or not frame.end_available or not frame.hand):
                return False
            if shortfall > 0:
                # The first frame of a deal may show a partial hand. The previous
                # draw penalty explains when the stable opening can be shorter.
                effects = json.loads(self.native.export_json())['status']['effects']
                if not any(effect['cls'] == 'StartTurnCardDrawDownStatusEffect'
                           and effect['turn'] == 1 and effect['value'] == shortfall for effect in effects):
                    return False
        # Keep cards surviving a same-turn draw. At turn start the old hand
        # has already been discarded and can reappear in the new draw.
        old_hand = ({card.instance_id for card in before.state.deck.hand}
                    if predicted.current_turn == self.native.current_turn else set())
        refs = self._cards(predicted, before.state, frame.hand)
        missing = Counter(frame.hand)
        kept, replaced = [], []
        for guid in sorted(hand, key=lambda iid: iid not in old_hand):
            cid = refs[guid].card_id
            if missing[cid] > 0:
                missing[cid] -= 1
                kept.append(guid)
            elif guid in old_hand and not rebind_survivors:
                return False
            else:
                replaced.append(guid)
        grave = predicted.zone('grave')
        older_discards = set(self.native.zone('grave'))
        for cid in missing.elements():
            guid = next((guid for guid in deck if refs[guid].card_id == cid), None)
            if guid is not None:
                deck.remove(guid)
            else:
                guid = next((guid for guid in grave if guid in older_discards and refs[guid].card_id == cid), None)
                if guid is None:
                    return False
                if shortfall:
                    grave.remove(guid)
                else:
                    grave[grave.index(guid)] = replaced.pop()
            kept.append(guid)
        predicted.set_zone('hand', kept)
        predicted.set_zone('deck', deck + replaced)
        predicted.set_zone('grave', grave)
        return True

    def observed_position(self, before, frame):
        """Rebuild a visible position reached without committing the pending action."""
        if frame.drinks != before.drink_slots:
            return None
        if frame.turns == before.state.turns_remaining:
            rx = self.native.clone()
            refs = self._cards(rx, before.state, frame.hand)
            pool = rx.zone('hand') + rx.zone('deck')
            hand = []
            for cid in frame.hand:
                guid = next((guid for guid in pool if refs[guid].card_id == cid), None)
                if guid is None:
                    return None
                pool.remove(guid)
                hand.append(guid)
            rx.set_zone('hand', hand)
            rx.set_zone('deck', pool)
        elif frame.turns == before.state.turns_remaining - 1:
            rx = self._fork(before, None, frame, observed_draws=True)
            if isinstance(rx, NativeChoice):
                return None
            self._align_observed_draw(before, frame, rx)
        else:
            return None
        step = self._match(before, None, frame, rx)
        return step.request if step is not None else None

    def reobserve(self, observed):
        if self.candidate is None or self.candidate[1] is not None or self.candidate[2].request != observed:
            raise ValueError('idol_road_native_observation_mismatch')
        self.native = self.candidate[3]
        self.candidate = None

    def _match(self, before, action, frame, predicted, *, learned=0):
        if predicted.ended or predicted.remain_turn != frame.turns:
            return None
        refs = self._cards(predicted, before.state, frame.hand)
        hand = tuple(refs[guid] for guid in predicted.zone('hand'))
        if Counter(card.card_id for card in hand) != Counter(frame.hand):
            return None
        slots = bind_tower_slots(hand, frame.hand)
        order = tuple(guid for guid, _ in slots)
        predicted.set_zone('hand', list(order))
        expected_drinks = tuple(pair for pair in before.drink_slots
                                if action is None or action.kind is not ActionKind.USE_DRINK
                                or pair[0] != action.source_slot)
        if frame.drinks != expected_drinks or predicted.drinks != [self.rules.drink_forms[c] for _, c in frame.drinks]:
            return None
        state = before.state
        attrs = tuple(ATTRIBUTES[i - 1] for i in predicted.turn_types[predicted.current_turn - 1:])
        if attrs and len(attrs) < predicted.remain_turn:
            attrs += (attrs[-1],) * (predicted.remain_turn - len(attrs))
        if not attrs or attrs[0] != frame.attribute:
            return None
        statuses = _observed_fields(frame)
        native_status = json.loads(predicted.export_json())['status']
        effects = native_status['effects']
        model_values = {'score': predicted.score, 'stamina': predicted.stamina, 'genki': predicted.block,
                        'good_impression_turns': predicted.review, 'motivation': predicted.aggressive,
                        'concentration': predicted.lesson_buff,
                        'good_condition_turns': next((entry['turn'] for entry in effects
                            if entry['cls'] == 'ParameterBuffStatusEffect'), 0)}
        self.observed_delta = {key: value - model_values[key] for key, value in (
            ('score', frame.score), ('stamina', frame.stamina), ('genki', frame.genki), *statuses.items())
            if value != model_values[key]}
        _read_resources(predicted, frame.score, frame.stamina, frame.genki, dict(frame.statuses), effects)
        unread = self.unread_formation_cards
        unread = unread - learned if unread is not None and unread >= learned else None
        deck = DeckBelief(tuple(refs.values()), tuple(refs[guid] for guid in order),
            *(tuple(refs[guid] for guid in predicted.zone(zone)) for zone in ('deck', 'grave', 'lost', 'hold')),
            knowledge=DeckKnowledge.UNKNOWN if unread != 0 else DeckKnowledge.KNOWN_COMPOSITION)
        after_state = replace(state, deck=deck, turn=predicted.current_turn, turns_remaining=frame.turns,
            stamina=frame.stamina, genki=frame.genki, score=frame.score, drinks=tuple(c for _, c in frame.drinks),
            actions_remaining=max(0, self.rules.uiobs._extra_actions(native_status) + predicted.remaining_actions),
            good_impression_turns=predicted.review, motivation=predicted.aggressive,
            concentration=statuses.get('concentration', model_values['concentration']),
            good_condition_turns=statuses.get('good_condition_turns', model_values['good_condition_turns']),
            **_turn_fields(attrs[:frame.turns], self.multipliers))
        after = replace(before, snapshot_id=frame.frame_id, state=after_state, card_slots=slots, drink_slots=frame.drinks,
                        decision_kind=DecisionKind.LIVE)
        step = HifStep(after, frame.score - state.score, False, ())
        self.candidate = (before.state, action, step, predicted, unread)
        return step

    def observe_transition(self, before, action, step):
        if self.candidate is None or self.candidate[:3] != (before, action, step):
            raise ValueError('idol_road_native_receipt_mismatch')
        self.native = self.candidate[3]
        self.unread_formation_cards = self.candidate[4]
        self.candidate = None
        self.choice_receipts.clear()

    def observe_result(self, before, action, result):
        """Ranking is an authoritative receipt even if the modeled final score differs."""
        try:
            fork = self._fork(before, action, None)
        except (RuntimeError, ValueError):
            return ExamResult(**result), None
        if isinstance(fork, NativeChoice):
            return ExamResult(**result), None
        if fork.ended:
            self.native = fork
        self.choice_receipts.clear()
        return ExamResult(**result), fork.score if fork.ended else None
