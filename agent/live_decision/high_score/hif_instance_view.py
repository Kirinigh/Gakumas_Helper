"""Private search coordinates for unreferenced, interchangeable card copies.

Business states, transition inputs and physical slots never use these labels.
Referenced copies and ordered information retain their original identities.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, replace, is_dataclass

from .kernel import PythonNiaProKernel
from .contracts import DeckKnowledge

VERSION = 'hif-unreferenced-instance-view-v1'
_ZONES = ('hand', 'draw_pile', 'discard', 'removed', 'held')


def _strings(value):
    if isinstance(value, str):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _strings(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _strings(item)


def instance_view(state):
    """Return a canonical search state and actual-to-search ID bijection.

    The whole non-deck state remains exact, including exhausted schedules.
    A referenced member pins its entire semantic group. Pending resolution,
    support facts and known draw order conservatively disable this first slice.
    """
    if (state.hand_support or state.pending_choice or state.pending_after_card_ids
            or state.resolution_stack or state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION
            or state.deck.known_top_count):
        return state, {}
    referenced = {text for field in fields(state) if field.name != 'deck'
                  for text in _strings(getattr(state, field.name))}
    groups = defaultdict(list)
    for card in state.deck.all_cards:
        groups[card.semantic_key].append(card)
    location = {card.instance_id: index for index, zone in enumerate(_ZONES)
                for card in getattr(state.deck, zone)}
    mapping = {}
    for cards in groups.values():
        ids = {card.instance_id for card in cards}
        if len(ids) < 2 or ids & referenced:
            continue
        actual = sorted(ids, key=lambda identity: (location[identity], identity))
        mapping.update(zip(actual, sorted(ids), strict=True))
    mapping = {actual: canonical for actual, canonical in mapping.items() if actual != canonical}
    if not mapping:
        return state, {}
    renamed = {card.instance_id: replace(card, instance_id=mapping.get(card.instance_id, card.instance_id))
               for card in state.deck.all_cards}
    deck = replace(state.deck, **{zone: tuple(renamed[card.instance_id]
                    for card in getattr(state.deck, zone)) for zone in ('all_cards', *_ZONES)})
    return PythonNiaProKernel._canonical_probability_state(replace(state, deck=deck)), mapping


def action_view(key, mapping):
    """Relabel only a card action's entity; physical source slots stay external."""
    return (key[0], mapping.get(key[1], key[1]), *key[2:])
