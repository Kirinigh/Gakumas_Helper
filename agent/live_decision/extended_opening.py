"""Resolve an initial four/five-card row from complete inventory and UI titles.

Overlapped titles never establish upgrade identity. Only unresolved physical
slots are selected for a full-title inspection, then returned without play.
"""
from collections import Counter
from dataclasses import asdict

from .opening_observer import valid_opening_frame_ids
from .hif_hand_inspection import TITLE_VISIT_VERSION, read_upstream_opening_slot

UPSTREAM_VERSION = 'hif-extended-opening-identity-v3'


def verify_unsubmitted_opening(context, event, fresh_core):
    """An initial observation stays an observation; fresh core supplies no old input permit."""
    import json
    from dataclasses import asdict

    bundle = context.gkh_hif_opening_state
    run = context.gkh_hif_live_context
    same = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    if (getattr(context, 'gkh_hif_committed_state', None) is not None
            or event.get('event') != 'live_opening_state' or event.get('accepted') is not True
            or event.get('strategy_state_bound') is not True
            or (event.get('run_id'), event.get('revision')) != (run.run_id, run.revision)
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or bundle.state.turn != 1 or bundle.state.score != 0
            or bundle.state.deck.discard or bundle.state.deck.removed
            or not same(event.get('state'), asdict(bundle.state))
            or any(not same(fresh_core.get(k), event.get('core', {}).get(k)) for k in ('fields', 'families'))
            or not valid_opening_frame_ids(fresh_core.get('frame_ids'))
            or set(fresh_core['frame_ids']) & set(event['core'].get('frame_ids', ()))):
        raise ValueError('unsubmitted opening fresh continuity failed')


def assignments(families, inventory_rows, known):
    counts = Counter(row['card_id'] for row in inventory_rows if row['zone'] == '手札')
    if sum(counts.values()) != len(families) or len(families) not in (4, 5):
        raise ValueError('extended_opening_inventory_size_invalid')
    output = []

    def visit(prefix):
        slot = len(prefix)
        if slot == len(families):
            output.append(tuple(prefix))
            return
        for cid in sorted(set(families[slot])):
            if counts[cid] and (slot not in known or known[slot] == cid):
                counts[cid] -= 1
                visit([*prefix, cid])
                counts[cid] += 1
    visit([])
    if not output:
        raise ValueError('extended_opening_title_inventory_conflict')
    return output


def _permanent_title_id(inspection, family, catalog, run):
    from .hif_single_plus import normalize_hif_title
    from .hif_hand_inspection import verify_upstream_inspection

    name = verify_upstream_inspection(inspection,run,catalog)
    cid = inspection['detail']['card_id']
    if cid not in family:
        raise ValueError('extended_opening_full_title_not_verified')
    # Extra temporary pluses describe this play. Keep the observed title,
    # but let the permanent inventory and another exact slot decide its ID.
    return cid if name == normalize_hif_title(catalog.entries[cid].name) else None


def verify_extended_opening_identity(proof, *, run, core, inventory_rows, catalog):
    if (not isinstance(proof, dict) or proof.get('version') != UPSTREAM_VERSION
            or (proof.get('run_id'), proof.get('revision')) != (run.run_id, run.revision)
            or proof.get('business_submit_count') != 0
            or proof.get('core') != {key: _json(core[key]) for key in ('fields', 'families', 'frame_ids')}):
        raise ValueError('extended_opening_identity_epoch_or_core_changed')
    families = core['families']
    origin = proof.get('initial_core', {})
    if (any(origin.get(key) != _json(core[key]) for key in ('fields', 'families'))
            or not valid_opening_frame_ids(origin.get('frame_ids'))):
        raise ValueError('extended_opening_initial_checkpoint_changed')
    known = {}
    inspected = set(known)
    for item in proof.get('inspections', ()):
        slot = item.get('slot')
        if type(slot) is not int or not 0 <= slot < len(families) or slot in inspected:
            raise ValueError('extended_opening_duplicate_or_invalid_slot')
        inspected.add(slot)
        if item.get('selection_accepted') is not True:
            raise ValueError('extended_opening_inspection_not_returned')
        if item.get('version') != TITLE_VISIT_VERSION or item.get('slot') not in item.get('physical_slots', ()):
            raise ValueError('extended_opening_upstream_selection_unconfirmed')
        cid = _permanent_title_id(item, families[slot], catalog, run)
        if cid is not None:
            known[slot] = cid
    values = assignments(families, inventory_rows, known)
    if len(values) != 1 or list(values[0]) != proof.get('card_ids'):
        raise ValueError('extended_opening_physical_identity_still_ambiguous')
    return values[0]


def _json(value):
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def initial_bound_hand_ids(context, families, catalog):
    """Reuse the just-verified opening titles before its first business action."""
    bundle = context.gkh_hif_opening_state
    proof = context.gkh_hif_extended_opening_identity
    state = bundle.state
    if (getattr(context, 'gkh_hif_committed_state', None) is not None
            or getattr(context, 'gkh_hif_committed_event', None) is not None
            or state.turn != 1 or state.score != 0 or state.deck.discard or state.deck.removed
            or (bundle.run_id,bundle.revision) != (context.gkh_hif_live_context.run_id,
                                                  context.gkh_hif_live_context.revision)
            or tuple(tuple(row) for row in proof.get('core',{}).get('families',())) != tuple(families)):
        raise ValueError('extended_opening_initial_hand_binding_changed')
    ids = tuple(bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id for ref in state.deck.hand)
    verified = verify_extended_opening_identity(proof,run=context.gkh_hif_live_context,
        core=proof['core'],inventory_rows=[dict(zone='手札',card_id=cid) for cid in ids],catalog=catalog)
    if verified != ids:
        raise ValueError('extended_opening_initial_hand_order_changed')
    return ids


def observe_extended_opening(module, context, params, frames, core, inventory_rows):
    """Reuse the existing Maa selection, full-title reader and verified cancel."""
    from .inventory import capture_initial_live_checkpoint

    run = context.gkh_hif_live_context
    catalog = module.SkillCardFeatureCatalog.load(module.CATALOG_PATH)
    recognizer = module.EmbeddingCardRecognizer.load(module.MODEL_ROOT)
    baseline = {key: core[key] for key in ('fields', 'families')}
    known, inspections = {}, []
    proof = dict(version=UPSTREAM_VERSION, run_id=run.run_id, revision=run.revision,
        business_submit_count=0, input_click_count=0, inspections=inspections,
        initial_core={key: _json(core[key]) for key in ('fields','families','frame_ids')})
    context.gkh_hif_extended_opening_identity = proof

    def checkpoint(group):
        # Selection reset owns a separately loaded recognizer. Rebind its
        # gallery before returning to this reader; feature tensors are not
        # interchangeable across recognizer instances.
        context.gkh_hif_card_art = module.FrozenMaskedArt.load(recognizer, module.MODEL_ROOT / 'hif')
        value = module._inventory_live_checkpoint(context, group, recognizer,
            identity_threshold=params['debug_identity_threshold'], identity_min_margin=params['debug_identity_min_margin'],
            **({'single_frame': True} if len(group) == 1 else {}))
        if any(value[key] != baseline[key] for key in baseline):
            raise ValueError('extended_opening_inspection_changed_hand_or_resources')
        return value

    for _ in range(len(core['families'])):
        possible = assignments(core['families'], inventory_rows, known)
        if len(possible) == 1:
            break
        visited = {item['slot'] for item in inspections}
        slot = next((i for i in range(len(core['families'])) if i not in visited
            and len({value[i] for value in possible}) > 1),None)
        if slot is None:
            break
        record = dict(slot=slot, selection_accepted=False)
        inspections.append(record)
        record.update(hand_count=len(core['families']),candidate_families=_json(core['families']),
            input_click_count=0,business_submit_count=0,observed_live_context=asdict(run),accepted=False)
        return_frames = []
        try:
            read_upstream_opening_slot(context,record,module,catalog,returned_frames=return_frames)
        finally:
            proof['input_click_count'] += record['input_click_count']
        record.update(selection_accepted=True,accepted=True)
        cid = _permanent_title_id(record,core['families'][slot],catalog,run)
        if cid is not None:
            known[slot] = cid
    possible = assignments(core['families'], inventory_rows, known)
    if len(possible) != 1:
        raise ValueError('extended_opening_inspection_budget_exhausted')
    if inspections:
        # Reuse the hand frame already captured after closing the detail.
        def capture_return():
            if return_frames:
                group = (return_frames[-1],)
                return_frames.clear()
                return group
            return (context.tasker.controller.post_screencap().wait().get(),)
        frames,core = capture_initial_live_checkpoint(capture_return,checkpoint,proof)
    proof.update(card_ids=list(possible[0]), core={key: _json(core[key]) for key in ('fields','families','frame_ids')})
    ids = verify_extended_opening_identity(proof,run=run,core=core,inventory_rows=inventory_rows,catalog=catalog)
    titles = tuple(tuple(catalog.entries[cid].name for cid in ids) for _ in frames)
    return frames, core, titles, proof
