"""Miyu's move-list input flow for a pending Hiro selection."""

from collections import defaultdict

from .hif_choice_panel import CHOICE_BOXES, hif_choice_grid, _choice_selected_title_rows
from .idol_road_native import ChoiceReceipt
from .hif_choice_scroll import low_move_layout, scan_move_viewports
from .high_score.contracts import ActionKind


def _center(box):
    x, y, width, height = box
    return x + width // 2, y + height // 2


def execute_native_choice(battle, image, menu, kind, *, confirm):
    """Bind the visible slots and return the job that confirmed their selection."""
    by_id = defaultdict(list)
    for card, zone in zip(menu.cards, menu.source_zones, strict=True):
        by_id[card.card_id].append((card, zone))
    equivalent = {cid: [ref.instance_id for ref, _ in candidates]
                  for cid, candidates in by_id.items()
                  if len({(ref.semantic_key, zone) for ref, zone in candidates}) == 1}
    move = kind is ActionKind.MOVE_CARD
    low = move and low_move_layout(image)
    count = len(menu.cards)
    wanted = min(menu.maximum, count)
    scrolls = []

    def journal(_stage, payload):
        scrolls[:] = payload['choice_scrolls']

    scroll = dict(capture=battle.capture, identify_frame=lambda *_: battle.frame_id, journal=journal,
                  client_size=battle.client_size)
    views = (scan_move_viewports(battle.context, image, count, **scroll) if low
             else ((image, None, enumerate(CHOICE_BOXES[:count])),))
    targets = []
    for frame, offset, cells in views:
        for slot, box in cells:
            try:
                hif_choice_grid(frame, expected_count=count, target_slot=slot, scroll_offset=offset)
            except ValueError:
                continue
            card_id = battle.choice_card_id(frame, box, slot, equivalent)
            guids = equivalent.get(card_id)
            if guids:
                targets.append((slot, card_id, guids.pop(0)))
            if len(targets) == wanted:
                break
        if len(targets) == wanted:
            break
    if len(targets) < menu.minimum:
        raise ValueError('idol_road_native_choice_targets_unbound')
    if not targets:
        frame_id = battle.frame_id
        job_id = battle.click((220, 1160), purpose='cancel-move-choice')
        return ChoiceReceipt(menu.effect, (), frame_id, job_id)

    picked = []
    for slot, card_id, guid in targets:
        current = battle.capture()
        offset = None
        if low:
            current, offset, _ = next(scan_move_viewports(battle.context, current, count, **scroll,
                                                          previous=scrolls, target_slot=slot))
        box = hif_choice_grid(current, expected_count=count, target_slot=slot, scroll_offset=offset)[slot]
        if battle.choice_card_id(current, box, slot, equivalent) != card_id:
            raise ValueError('idol_road_native_choice_slot_changed')
        battle.click(_center(box), purpose='preview-move-choice' if menu.maximum > 1 else 'select-move-choice')

        def title_matches(frame):
            rows = _choice_selected_title_rows(battle.context, frame)
            title = ''.join(''.join(row.text.split()) for row in rows).replace('顏', '顔')
            expected = ''.join(battle.kernel.cards[card_id].name.split()).replace('顏', '顔')
            return box if title == expected else None

        target = battle.poll(title_matches, 'move-choice-title')
        if menu.maximum > 1:
            battle.click(_center(target), purpose='select-move-choice')
        picked.append((slot, guid))

    button = battle.poll(confirm, 'move-choice-confirm')
    frame_id = battle.frame_id
    job_id = battle.click(_center(button), purpose='confirm-move-choice')
    return ChoiceReceipt(menu.effect, tuple(picked), frame_id, job_id)
