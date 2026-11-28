"""Bind GKH reward identities to Hiro's physical HIF card choices."""

import json
from pathlib import Path
from functools import cache

from utils.runtime_paths import resolve_packaged_runtime_path

from agent.live_decision.hiro_runtime import hiro_module

_PLANS = {
    'ProducePlanType_Plan1': 'sense',
    'ProducePlanType_Plan2': 'logic',
    'ProducePlanType_Plan3': 'anomaly',
}


def _contains_letter(box, letter):
    x, y, width, height = box
    center = letter.center
    return x <= center.x < x + width and y <= center.y < y + height


def hif_reward_options(skin):
    """Use the existing GKH offer reader only inside HIF cultivation."""
    from card_selection.model import ModelIntegrityError
    from live_decision.catalog import SkillCardFeatureCatalog

    from .outing import outing_offer_boxes
    from .activity_production import read_card_offers
    from .nonbattle_inventory import load_flat_inventory_readers

    select = hiro_module('vision.card_select')
    cards = hiro_module('vision.cards')
    master = hiro_module('master')

    @cache
    def resources():
        rows = master.rows('SELECT card.planType FROM IdolCardSkin AS skin JOIN IdolCard AS card '
                           'ON card.id = skin.idolCardId WHERE skin.id = ?', skin)
        plan = _PLANS.get(rows[0]['planType']) if len(rows) == 1 else None
        if plan is None:
            return None
        data = resolve_packaged_runtime_path(Path('assets/data'), Path('data'))
        classification = json.loads((data / 'skill_card_classification.json').read_bytes())['cards']
        lookup = {str(row['business_id']): row for row in classification}
        if len(lookup) != len(classification):
            return None
        reader, _ = load_flat_inventory_readers()
        return reader, SkillCardFeatureCatalog.load(data / 'skill_cards.csv'), lookup, plan

    def select_card_options(image):
        image = select.normalize(image)
        letters = select.card_letters(image)
        unknown = [select.CardOption(letter, None) for letter in letters]
        if len(letters) != 3:
            return unknown
        try:
            boxes = outing_offer_boxes(image)
            if not all(_contains_letter(box, letter) for box, letter in zip(boxes, letters, strict=True)):
                return unknown
            loaded = resources()
            if loaded is None:
                return unknown
            reader, catalog, lookup, plan = loaded
            offers = read_card_offers(reader, [image], catalog, plan=plan)
        except (OSError, ValueError, KeyError, ModelIntegrityError):
            return unknown

        if tuple((row['index'], tuple(row['box'])) for row in offers) != tuple(enumerate(boxes)):
            return unknown

        for index, (letter, offer) in enumerate(zip(letters, offers, strict=True)):
            entry = lookup.get(offer.get('card_id'))
            upgraded = offer.get('upgraded')
            if (entry is None or not entry.get('internal_id') or type(upgraded) is not bool
                    or entry.get('upgraded') is not upgraded or entry.get('plan') not in (plan, 'free')):
                continue
            card = master.card(entry['internal_id'], int(upgraded))
            if card is None:
                continue
            # Hiro's extractor also checks the red '+' pixels. Do not let that
            # path promote a candidate whose verified GKH form is ordinary.
            if not upgraded and cards.dialog_card_upgraded(image, letter)[0]:
                continue
            unknown[index] = select.CardOption(letter, card)
        return unknown

    return select_card_options
