"""Hiro's battle simulator behind the Maa receipt and UINet decision interfaces."""

import json
from collections import Counter
from dataclasses import replace, dataclass

from .exam_loop import ExamResult
from .drink_icons import EVENT_GREEN_JUICE_ID
from .hiro_runtime import HIRO_REVISION, hiro_module
from .idol_road_battle import _wire_card
from .idol_road_runtime import _READY, ATTRIBUTES, _turn_fields, _observed_fields, bind_tower_slots
from .high_score.contracts import CardRef, PlanType, ActionKind, BattleStep, DeckBelief, GameAction, DecisionKind, DeckKnowledge
from .high_score.tower_gnrpa import _deck_entry, load_native_tower
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

    def open(self, opening, loadout, model, observed_types, *, item_uses, statuses=(), status_slots=()):
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
        _read_resources(native, initial.score, initial.stamina, initial.genki, dict(statuses), status_slots=status_slots)
        return NativeTowerSession(self, native, model, loadout, observed_types)

    def restore(self, raw, frame, loadout, model, observed_types):
        from gakumas_exam.native import master_path

        from .high_score.hif_mcts_cache import request_from_public

        if raw['hiro'] != HIRO_REVISION:
            raise ValueError('idol_road_saved_rules_changed')
        request = request_from_public(raw['public'], self.kernel)
        state = request.state
        if (request.decision_kind is not DecisionKind.LIVE or state.terminal
                or (state.turns_remaining, state.stamina, state.genki, state.score) !=
                (frame.turns, frame.stamina, frame.genki, frame.score)
                or request.drink_slots != frame.drinks):
            raise ValueError('idol_road_saved_position_does_not_match_hud')
        slots = bind_tower_slots(state.deck.hand, frame.hand)
        if tuple(tuple(row) for row in raw['card_slots']) != slots:
            raise ValueError('idol_road_saved_hand_does_not_match_hud')
        rust, _ = load_native_tower()
        native = rust.Exam(master_path(self.master), raw['native'])
        if (native.ended or native.remain_turn != frame.turns or native.current_turn != state.turn
                or (native.stamina, native.block, native.score) != (frame.stamina, frame.genki, frame.score)
                or native.zone('hand') != [guid for guid, _ in slots]
                or native.drinks != [self.drink_forms[cid] for _, cid in frame.drinks]):
            raise ValueError('idol_road_saved_native_position_invalid')
        request = replace(request, snapshot_id=frame.frame_id, card_slots=slots, capabilities=_READY)
        request.validate()
        observed_types.extend(raw['observed_types'])
        return request, NativeTowerSession(self, native, model, loadout, observed_types)


_OBSERVED_STATUS = {
    'concentration': ('LessonBuffStatusEffect', False),
    'good_condition_turns': ('ParameterBuffStatusEffect', True),
    'full_power': ('FullPowerPointStatusEffect', False),
}
_REVIEW_MULTIPLIER = 'ReviewMultipleStatusEffect'
_PRIMARY_NATIVE_KINDS = frozenset(('lesson_buff', 'aggressive', 'review', 'parameter_buff', 'full_power_point'))


def _hiro_effect_kinds():
    mapping = hiro_module('bench_bridge.screen').EFFECT_KIND
    # The packed status icon is the stage effect. The engine maps it to the
    # same 熱意 class that Hiro exports as enthusiastic.
    return mapping | {'ProduceExamEffectType_ExamGimmickEnthusiastic':
                      mapping['ProduceExamEffectType_ExamEnthusiastic']}


def _read_status_slots(rx, slots, effects):
    mapping = _hiro_effect_kinds()
    uiobs = hiro_module('bench_bridge.uiobs')
    kinds = uiobs.VALUE_KIND | uiobs.TURN_AMOUNT_KIND | uiobs.FLAG_KIND | uiobs.LAYER_KIND
    visible = [(row, mapping[row.effect_type]) for row in slots if row.effect_type in mapping]
    counts = Counter(kind for _, (kind, _) in visible)
    active_kinds = {kinds.get(entry['cls']) for entry in effects}
    for row, (kind, how) in visible:
        if row.faint or counts[kind] != 1 or kind in _PRIMARY_NATIVE_KINDS:
            continue
        matched = [entry for entry in effects if kinds.get(entry['cls']) == kind and not entry.get('trig')]
        if len(matched) != 1:
            continue  # The panel can aggregate several native effects into one slot.
        if row.alternatives:
            icons = hiro_module('vision.status')._icon_index()[0]
            alternatives = {mapping.get(icons.get(key, {}).get('type'), (None, None))[0]
                            for key in row.alternatives}
            if active_kinds.intersection(alternatives - {kind, None}):
                continue
        active, = matched
        value = row.value
        if value is not None and ((how == 'pct' and row.unit == '%') or
                                  (how == 'value' and row.unit in (None, '回'))):
            native_value = value * 10 if how == 'pct' else value
            if active['value'] != native_value:
                rx.override_status(active['cls'], False, native_value)
                if native_value == 0:
                    continue
        duration = row.turns
        if duration is None and how in ('flag', 'turns') and row.unit in ('ターン', 'turns'):
            duration = value
        if duration is not None and active.get('limited') and uiobs._turns(active) != duration:
            native_turn = duration - (0 if active.get('passing') else 1)
            if native_turn > 0 or duration == 0:
                rx.override_status(active['cls'], True, native_turn if duration else 0)


def _read_resources(rx, score, stamina, genki, statuses, effects=None, *, status_slots=()):
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
    if status_slots:
        if effects is None:
            effects = json.loads(rx.export_json())['status']['effects']
        _read_status_slots(rx, status_slots, effects)


def _native_observed_statuses(rx, effects):
    return {'good_impression_turns': rx.review, 'motivation': rx.aggressive,
            'concentration': rx.lesson_buff,
            'good_condition_turns': next((entry['turn'] for entry in effects
                if entry['cls'] == 'ParameterBuffStatusEffect'), 0),
            'full_power': next((entry['value'] for entry in effects
                if entry['cls'] == 'FullPowerPointStatusEffect'), 0)}


def _visible_status_readouts(slots, kinds):
    rows = [(row, kinds[row.effect_type]) for row in slots if row.effect_type in kinds]
    counts = Counter(kind for _, (kind, _) in rows)
    readouts = {}
    for row, (kind, how) in rows:
        if kind in _PRIMARY_NATIVE_KINDS or counts[kind] != 1 or row.faint or row.alternatives:
            continue
        if row.value is not None and how == 'pct' and row.unit == '%':
            readouts[kind, 'value'] = row.value / 100
        elif row.value is not None and how == 'value' and row.unit in (None, '回'):
            readouts[kind, 'value'] = row.value
        duration = row.turns
        if duration is None and how == 'flag' and row.unit in ('ターン', 'turns'):
            duration = row.value
        if duration is not None:
            readouts[kind, 'turns'] = duration
    return readouts


def _native_status_readouts(uiobs, snapshot):
    rows = uiobs.statuses(snapshot)
    counts = Counter(kind for kind, _, _ in rows)
    return {(kind, field): amount for kind, value, turns in rows if counts[kind] == 1
            for field, amount in (('value', value), ('turns', turns)) if amount >= 0}


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

    def committed_position(self, request):
        from .high_score.hif_mcts_cache import public_request

        if self.candidate is not None or request.decision_kind is not DecisionKind.LIVE or self.choice_receipts:
            return None
        return dict(hiro=HIRO_REVISION, native=self.native.export_json(), public=public_request(request),
                    card_slots=request.card_slots, observed_types=list(self.observed_types),
                    unknown_deck=self.unread_formation_cards)

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
        trial = self._fork(before, action, frame, observed_draws=frame.turns == before.state.turns_remaining)
        if (isinstance(trial, NativeChoice) or trial.ended or trial.remain_turn != frame.turns
                or len(trial.zone('hand')) != len(frame.hand)):
            return ()
        refs = self._cards(trial, before.state, frame.hand)
        older_discards = set(self.native.zone('grave'))
        return Counter(refs[guid].card_id for guid in (
            trial.zone('hand') + trial.zone('deck') +
            [guid for guid in trial.zone('grave') if guid in older_discards]))

    def idle_hand_forms(self, before, frame):
        if frame.turns == before.state.turns_remaining:
            deck = before.state.deck
            return Counter(card.card_id for card in (*deck.hand, *deck.draw_pile))
        if frame.turns == before.state.turns_remaining - 1:
            return self.overlapping_hand_forms(before, None, frame)
        return ()

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

    def sync_item_uses(self, uses, *, status_slots=()):
        """Reconcile bound charges for items and uniquely identified card effects."""
        slots = [row for row in status_slots if row.kind == 'enchant' and row.card_id]
        if not uses and not slots:
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
        observed_cards = Counter(row.card_id for row in slots)
        for row in slots:
            if observed_cards[row.card_id] != 1:
                continue
            matched = [entry for entry in state['status']['effects']
                       if ((entry.get('trig') or {}).get('source') or {}).get('card') == row.card_id]
            if len(matched) != 1:
                continue
            active, = matched
            trigger = active['trig']
            if row.unit == '回' and row.value is not None and trigger['limit_count'] >= 0:
                if trigger['limit_count'] != row.value:
                    changes[row.card_id] = row.value - trigger['limit_count']
                    trigger['limit_count'] = row.value
            duration = row.turns if row.turns is not None else row.value if row.unit == 'ターン' else None
            if duration is not None and active.get('limited'):
                remaining = self.rules.uiobs._turns(active)
                if duration != remaining:
                    changes[f'{row.card_id}:turns'] = duration - remaining
                    active['turn'] = duration - (0 if active.get('passing') else 1)
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
        cid, = candidates
        if old is not None and self._base_card_id(old.card_id) == self._base_card_id(cid):
            return replace(old, card_id=cid, upgraded=bool(up))
        return CardRef(guid, cid, bool(up))

    def _base_card_id(self, cid):
        card = self.rules.kernel.cards.get(cid)
        return getattr(card, 'base_card_id', None) or cid

    def _observed_draw_prefix(self, rx, before, frame, *, played=None):
        refs = self._cards(rx, before.state, frame.hand)
        survivors = Counter(refs[guid].card_id for guid in rx.zone('hand') if guid != played)
        exact = survivors & Counter(frame.hand)
        # As in Hiro, keep the existing instance when its visible form changes.
        # Reserve exact survivors first, leaving only unmatched slots for draws.
        card_key = self._base_card_id if frame.turns == before.state.turns_remaining else lambda cid: cid
        families = Counter(card_key(cid) for cid in (survivors - exact).elements())
        needed = []
        for slot, cid in enumerate(frame.hand):
            family = card_key(cid)
            if exact[cid]:
                exact[cid] -= 1
            elif families[family]:
                families[family] -= 1
            else:
                needed.append((slot, cid))
        deck = rx.zone('deck')
        bound = {}
        for slot, cid in needed:
            if (selected := next((guid for guid in deck if refs[guid].card_id == cid), None)) is not None:
                deck.remove(selected)
                bound[slot] = selected
        for slot, cid in needed:
            if slot in bound:
                continue
            family = card_key(cid)
            candidates = [guid for guid in deck if card_key(refs[guid].card_id) == family]
            if len(candidates) == 1:
                selected, = candidates
                deck.remove(selected)
                bound[slot] = selected
        if bound:
            # This only constrains the trial draw. The actual effect must still
            # yield the observed forms before the same physical cards are committed.
            rx.set_zone('deck', [bound[slot] for slot in sorted(bound)] + deck)

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

    def observe_frame(self, before, action, frame, *, confirmed_card=False, previous_frame=None):
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
        if step is None and confirmed_card:
            step = self._match_confirmed_same_turn(before, action, frame, predicted, previous_frame)
        confirmed_play = (confirmed_card and action.kind is ActionKind.PLAY_CARD
                          and action.card_instance_id not in predicted.zone('hand'))
        # SKIP has no SELECT confirmation. Its dispatched action and a fresh
        # playable board one turn later provide the draw receipt instead.
        observed_turn_start = (action is not None and action.kind is ActionKind.END_TURN
                               and previous_frame is not None and previous_frame.frame_id == before.snapshot_id
                               and frame.frame_id != previous_frame.frame_id and frame.end_available
                               and frame.turns == previous_frame.turns - 1 == before.state.turns_remaining - 1
                               and predicted.current_turn == self.native.current_turn + 1)
        confirmed_draw = confirmed_play or observed_turn_start
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
                if step is None and confirmed_card:
                    step = self._match_confirmed_same_turn(before, action, frame, held, previous_frame)
                if step is not None:
                    self.observed_delta['actions_remaining'] = 1
        return step

    def _match_confirmed_same_turn(self, before, action, frame, predicted, previous_frame):
        if (not self._confirmed_same_turn_play(before, action, frame, predicted, previous_frame)
                or not self._align_observed_draw(before, frame, predicted, rebind_survivors=True)):
            return None
        step = self._match(before, action, frame, predicted)
        if step is not None:
            self.observed_delta['hand_rebound'] = 1
        return step

    def _confirmed_same_turn_play(self, before, action, frame, predicted, previous_frame):
        if action.kind is not ActionKind.PLAY_CARD or predicted.current_turn != self.native.current_turn:
            return False
        old = before.state
        played = next(card for card in old.deck.hand if card.instance_id == action.card_instance_id)
        genki_matches = frame.genki == predicted.block
        resources = frame.stamina, frame.genki
        if (frame.stamina != predicted.stamina
                or frame.hand.count(played.card_id) >= sum(card.card_id == played.card_id for card in old.deck.hand)):
            return False
        paid = resources != (old.stamina, old.genki)
        scored = frame.score != old.score and frame.score == predicted.score
        if genki_matches and (paid or scored):
            return True
        if (frame.turns != old.turns_remaining or action.card_instance_id in predicted.zone('hand')
                or previous_frame is None or previous_frame.frame_id != before.snapshot_id
                or previous_frame.frame_id == frame.frame_id or previous_frame.attribute != frame.attribute
                or frame.score != predicted.score):
            return False
        # Some cards pay in status turns or add stacks without changing HP,
        # block or score. Verify these against HUD reads on both sides of the play.
        before_status = _observed_fields(previous_frame)
        after_status = _observed_fields(frame)
        common = before_status.keys() & after_status.keys()
        snapshot = json.loads(predicted.export_json())
        effects = snapshot['status']['effects']
        # A reduced Genki gain can round up in the game where Hiro rounds it down.
        # In that case the gain alone cannot confirm the play. Use the status evidence below.
        if not genki_matches and not (
                frame.genki == predicted.block + 1 and predicted.block > old.genki
                and any(effect['cls'] == 'BlockAddDownStatusEffect' for effect in effects)
                and any(row.effect_type == 'ProduceExamEffectType_ExamBlockAddDown' and not row.faint
                        for row in frame.status_slots)):
            return False
        expected = _native_observed_statuses(predicted, effects)
        comparisons = [(before_status[key], after_status[key], getattr(old, key), expected[key]) for key in common]
        if previous_frame.status_slots and frame.status_slots:
            kinds = _hiro_effect_kinds()
            before_slots = _visible_status_readouts(previous_frame.status_slots, kinds)
            after_slots = _visible_status_readouts(frame.status_slots, kinds)
            if (paired := before_slots.keys() & after_slots.keys()):
                old_model = _native_status_readouts(self.rules.uiobs, json.loads(self.native.export_json()))
                new_model = _native_status_readouts(self.rules.uiobs, snapshot)
                comparisons.extend((before_slots[key], after_slots[key], old_model[key], new_model[key])
                                   for key in paired & old_model.keys() & new_model.keys())
        return (all((observed_before, observed_after) == (model_before, model_after)
                    for observed_before, observed_after, model_before, model_after in comparisons)
                and any(observed_before != observed_after for observed_before, observed_after, _, _ in comparisons))

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
                        **_native_observed_statuses(predicted, effects)}
        self.observed_delta = {key: value - model_values[key] for key, value in (
            ('score', frame.score), ('stamina', frame.stamina), ('genki', frame.genki), *statuses.items())
            if value != model_values[key]}
        _read_resources(predicted, frame.score, frame.stamina, frame.genki, dict(frame.statuses), effects,
                        status_slots=frame.status_slots)
        unread = self.unread_formation_cards
        unread = unread - learned if unread is not None and unread >= learned else None
        deck = DeckBelief(tuple(refs.values()), tuple(refs[guid] for guid in order),
            *(tuple(refs[guid] for guid in predicted.zone(zone)) for zone in ('deck', 'grave', 'lost', 'hold')),
            knowledge=DeckKnowledge.UNKNOWN if unread != 0 else DeckKnowledge.KNOWN_COMPOSITION)
        passion = state.passion
        if before.plan_type is PlanType.ANOMALY:
            model_statuses = self.rules.uiobs.statuses(json.loads(predicted.export_json()))
            passion = sum(value for kind, value, _ in model_statuses if kind == 'enthusiastic')
        after_state = replace(state, deck=deck, turn=predicted.current_turn, turns_remaining=frame.turns,
            stamina=frame.stamina, genki=frame.genki, score=frame.score, drinks=tuple(c for _, c in frame.drinks),
            actions_remaining=max(0, self.rules.uiobs._extra_actions(native_status) + predicted.remaining_actions),
            good_impression_turns=predicted.review, motivation=predicted.aggressive,
            concentration=statuses.get('concentration', model_values['concentration']),
            good_condition_turns=statuses.get('good_condition_turns', model_values['good_condition_turns']),
            full_power=statuses.get('full_power', model_values['full_power']), passion=passion,
            **_turn_fields(attrs[:frame.turns], self.multipliers))
        after = replace(before, snapshot_id=frame.frame_id, state=after_state, card_slots=slots, drink_slots=frame.drinks,
                        decision_kind=DecisionKind.LIVE)
        step = BattleStep(after, frame.score - state.score, False, ())
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
