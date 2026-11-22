"""Compose native HIF components within one uninterrupted observation batch."""

import math

from .upgrade import normalize_upgrade_title
from .selection import LiveSelectionError

OPENING_MODES = (
    ('cards', 'inventory_card_state_observe'),
    ('status', 'status_observe'),
    ('items', 'item_status_full'),
    ('drinks', 'drink_inventory_observe'),
    ('turns', 'turn_information_observe'),
)


def valid_opening_frame_ids(ids):
    return (isinstance(ids, (tuple, list)) and bool(ids)
            and all(isinstance(value, str) and value for value in ids)
            and len(ids) == len(set(ids)))


def direct_item_observation(event):
    """The HUD carries one shared endpoint because it dispatched no input."""
    frame_id = event.get('item_hud_frame_id')
    observation = event.get('identity_count_observation') or {}
    rows = observation.get('items') or ()
    return (event.get('phase') == 'item_hud_inventory'
            and event.get('accepted') is True and event.get('complete_list') is True
            and event.get('returned_unchanged') is True
            and event.get('input_click_count') == 0 and not event.get('inputs')
            and isinstance(frame_id, str) and bool(frame_id)
            and all(tuple(group or ()) == (frame_id,) for group in (
                event.get('before_frame_ids'), event.get('after_frame_ids'), observation.get('after_frame_ids')))
            and bool(rows) and event.get('item_slot_sources') == ['hud'] * len(rows))


def requested_opening_modes(required, observed):
    """Look up only the observation modes still requested by the caller."""
    modes = dict(OPENING_MODES)
    return ((key, modes[key]) for key in required if key not in observed)


def partial_hud_drinks(payload, run):
    """Return read-only observations for unknown or unsupported bottle identities."""
    reports = payload.get('status_events') or ()
    if not reports:
        return None
    event = reports[-1]
    if (payload.get('accepted') is not False or payload.get('executed') is not False
            or payload.get('execution_submit_count') != 0 or payload.get('total_input_click_count') != 0
            or event.get('phase') != 'complete_drink_inventory' or event.get('observation_source') != 'hud_bottles'
            or event.get('returned_unchanged') is not True or event.get('full_drink_state_ready') is not False
            or not (event.get('unknown_slots') or event.get('unsupported_slots')) or not event.get('slots')
            or (event.get('run_id'), event.get('revision')) != (run.run_id, run.revision)):
        return None
    return event


def opening_component_params(params, mode):
    if mode not in {value for _,value in OPENING_MODES}:
        raise ValueError('invalid opening component mode')
    names = {'provider','preset','expected_plan','allowed_unknown_fields',
             'provider_timeout_ms','max_snapshot_age_ms','minimum_identity_confidence',
             'debug_identity_threshold','debug_identity_min_margin'}
    return dict({name:value for name,value in params.items() if name in names},live_decision_mode=mode)


def opening_component_retryable(payload, key):
    """Resume a zero-input drink observation inside the current batch only."""
    if (key!='drinks' or payload.get('accepted') is not False
            or payload.get('executed') is not False
            or type(payload.get('total_input_click_count')) is not int
            or payload['total_input_click_count']!=0
            or type(payload.get('execution_submit_count')) is not int
            or payload['execution_submit_count']!=0):
        return False
    events = payload.get('status_events',())
    return bool(events) and events[-1].get('reason') in {
        'drink occupancy has unresolved or unstable slots',
        'drink controls changed during initial observation',
        'drink controls have no three-frame consensus',
    }


def opening_component(payload, key, run):
    """Bind only a successful native component with its own return proof."""
    events = payload.get('inventory_events' if key == 'cards' else 'status_events', ())
    if payload.get('accepted') is not True or not events or events[-1].get('accepted') is not True:
        if key == 'cards':
            raise LiveSelectionError('HIF opening component failed: '+key, events=events)
        raise ValueError('HIF opening component failed: '+key)
    if key == 'cards':
        transactions = [event for event in events if 'live_state_unchanged' in event]
        returned = bool(transactions) and all(event['live_state_unchanged'] is True for event in transactions)
    else:
        returned = events[-1].get('returned_unchanged') is True
    if not returned:
        raise ValueError('HIF opening component did not return unchanged: '+key)
    event = dict(events[-1])
    if key == 'items' and event.get('phase') == 'item_hud_inventory':
        observation = event.get('identity_count_observation') or {}
        if (not direct_item_observation(event) or payload.get('total_input_click_count') != 0
                or (observation.get('run_id'), observation.get('revision')) != (run.run_id, run.revision)):
            raise ValueError('HIF item HUD opening observation is incomplete or unbound')
    for name,value in (('run_id',run.run_id),('revision',run.revision)):
        if name in event and event[name] != value:
            raise ValueError('HIF opening native component epoch mismatch')
        event[name] = value
    event['returned_unchanged'] = True
    return event


def _opening_title_row(row):
    return (len(row.box)==4 and 1090 <= row.box[1] <= 1120
        and (14 <= row.box[3] <= 32
             or 32 < row.box[3] <= 36 and getattr(row,'complete_title_plate',False) is True))


def opening_hand_titles(rows, *, hand_count=3, excluded_slot=None):
    """Read one exact title per visible hand column; never strip upgrade marks."""
    if hand_count not in (1, 2, 3):
        raise ValueError('opening title layout requires two or three hand cards')
    titles = []
    columns = ((240,480),) if hand_count==1 else ((140,360),(360,590)) if hand_count == 2 else ((0,240),(240,480),(480,720))
    for slot,(left,right) in enumerate(columns):
        if slot == excluded_slot:
            titles.append(None)
            continue
        candidates = [row for row in rows if _opening_title_row(row)
            and left <= row.box[0]+row.box[2]/2 < right]
        if len(candidates) != 1:
            raise ValueError('HIF opening hand title is missing or ambiguous')
        row = candidates[0]
        if (type(row.score) not in (int,float) or not math.isfinite(row.score)
                or not .95 <= row.score <= 1 or not row.text):
            raise ValueError('HIF opening hand title confidence is unresolved')
        titles.append(row.text)
    return tuple(titles)


def opening_title_card_ids(title_frames, families, catalog):
    """Resolve full native titles against independently recognized families."""
    titles = tuple(tuple(normalize_upgrade_title(title) for title in
                         opening_hand_titles(rows,hand_count=len(families))) for rows in title_frames)
    if not titles or any(group != titles[0] for group in titles[1:]):
        raise ValueError('HIF full hand titles do not agree across observed frames')
    identities = []
    for title,family in zip(titles[0],families,strict=True):
        matches = [cid for cid in family if cid in catalog.entries
                   and catalog.entries[cid].name == title]
        if len(matches) != 1:
            raise ValueError('HIF full title does not uniquely match its visual family')
        identities.append(matches[0])
    return tuple(identities)


def _verify_selected_short_hand_detail(title_frames, families, *, selected_slot, selected_detail, frame_ids):
    if (len(families) not in (1,2,3) or type(selected_slot) is not int
            or not 0 <= selected_slot < len(families) or len(title_frames) != len(frame_ids)
            or not valid_opening_frame_ids(frame_ids)
            or selected_detail.get('accepted') is not True
            or selected_detail.get('card_id') not in families[selected_slot]):
        raise ValueError('selected short hand lacks complete detail identity')
    observations = selected_detail.get('observations',())
    if (tuple(row.get('frame_id') for row in observations) != tuple(frame_ids)
            or any(row.get('card_id') != selected_detail['card_id'] for row in observations)):
        raise ValueError('selected short hand detail frames do not match')


def selected_short_hand_title_card_ids(title_frames, families, catalog, *,
                                        selected_slot, selected_detail, frame_ids):
    """Combine the selected popup title with complete neighbouring footers."""
    _verify_selected_short_hand_detail(title_frames, families, selected_slot=selected_slot,
        selected_detail=selected_detail, frame_ids=frame_ids)
    frames = tuple(opening_hand_titles(rows,hand_count=len(families),
                   excluded_slot=selected_slot) for rows in title_frames)
    identities = []
    for slot,family in enumerate(families):
        if slot == selected_slot:
            identities.append(selected_detail['card_id'])
            continue
        titles = tuple(normalize_upgrade_title(rows[slot]) for rows in frames)
        if len(set(titles)) != 1:
            raise ValueError('HIF full hand titles do not agree across three frames')
        matches = [cid for cid in family if cid in catalog.entries
                   and catalog.entries[cid].name == titles[0]]
        if len(matches) != 1:
            raise ValueError('HIF full title does not uniquely match its visual family')
        identities.append(matches[0])
    return tuple(identities)


def verify_recovered_short_hand_titles(bundle, card_ids):
    """A selected recovery may not rewrite its already verified permanent hand."""
    from .high_score.live_policy import BoundHifOpeningPolicy

    if len(card_ids) != len(bundle.state.deck.hand):
        raise ValueError('recovered short hand count differs from the complete hand')
    policy = BoundHifOpeningPolicy(bundle,action_order=())
    for card_id,ref in zip(card_ids,bundle.state.deck.hand,strict=True):
        definition = bundle.kernel.cards[ref.card_id]
        if card_id == (definition.canonical_card_id or ref.card_id):
            continue
        if (getattr(bundle.state, 'hand_support', ())
                and card_id == policy._hand_support_visible_id(ref)):
            continue
        receipts = [r for r in bundle.opening_skill_support
                    if r.get('card_instance_id') == ref.instance_id]
        if len(receipts) != 1 or card_id != policy._opening_support_visible_id(ref):
            raise ValueError('complete selected-recovery title contradicts the verified hand')


def read_opening_identity_titles(context, image):
    """Read complete plates; a tight-box fallback must preserve exact text."""
    from types import SimpleNamespace

    from .layout import calibrated_live_client_size
    from .inventory import _inventory_ocr
    from .hif_visual import _hif_unselected_title_plates

    rows = _inventory_ocr(context, image, 'LiveOpeningIdentityTitles')
    plates = _hif_unselected_title_plates(image)
    native_complete_plate = calibrated_live_client_size(
        getattr(context, 'gkh_hif_client_size', None)) == (480, 853)
    gap = None
    if len(plates) == 2 and all(180 <= p[2] <= 190 and 30 <= p[3] <= 34 for p in plates):
        left, right = sorted(plates)
        if left[0]+left[2] < right[0]:
            gap = (left[0]+left[2], max(left[1], right[1]),
                   right[0], min(left[1]+left[3], right[1]+right[3]))
    result = []
    for row in rows:
        x,y,w,h = row.box
        # A native centered singleton has one independent complete title plate.
        # Whole-page OCR can extend beyond it or become 33-36px tall. Read that
        # plate directly, preserving the raw box, and require identical text;
        # the caller still checks all three frames and the visual card family.
        if native_complete_plate and len(plates) == 1:
            px, py, pw, ph = plates[0]
            if (265 <= px <= 269 and 180 <= pw <= 190 and 30 <= ph <= 34
                    and 1090 <= y <= 1120 and 14 <= h <= 36 and 40 <= w <= 190
                    and px <= x+w/2 < px+pw and .95 <= row.score <= 1):
                reread = _inventory_ocr(context, image[py:py+ph,px:px+pw].copy(),
                    'LiveOpeningIdentityTitlePlate', only_rec=True)
                if (len(reread) == 1 and .95 <= reread[0].score <= 1
                        and normalize_upgrade_title(reread[0].text) == normalize_upgrade_title(row.text)):
                    bounds = _inventory_ocr(context, image[py:py+ph,px:px+pw].copy(),
                        'HifSupportCompleteTitleBounds', only_rec=False)
                    observed_bounds = None
                    if (len(bounds) == 1 and bounds[0].text == reread[0].text
                            and .95 <= bounds[0].score <= 1):
                        bx, by, bw, bh = bounds[0].box
                        if 0 <= bx < bx+bw <= pw and 0 <= by < by+bh <= ph:
                            observed_bounds = dict(text=bounds[0].text, score=bounds[0].score,
                                box=[px+bx, py+by, bw, bh], roi=list(plates[0]),
                                acquisition_mode='native_complete_title_detection')
                    result.append(SimpleNamespace(text=reread[0].text, score=reread[0].score,
                        box=plates[0], complete_title_plate=True,
                        acquisition_mode='native_complete_title_plate',
                        **(dict(source_plate_observation=observed_bounds) if observed_bounds else {}),
                        source_page_observation=dict(text=row.text, score=row.score, box=row.box)))
                    continue
        # Two independently complete plates bound the inter-card background.
        # Exclude only whole rows in that gap; keep plate/edge ambiguities and
        # all text/confidence evidence inside the actual cards unchanged.
        if (gap is not None and _opening_title_row(row) and w > 0 and h > 0
                and gap[0] <= x and x+w <= gap[2]
                and gap[1] <= y and y+h <= gap[3]):
            continue
        if native_complete_plate and _opening_title_row(row):
            complete = [p for p in plates if 180 <= p[2] <= 190 and 30 <= p[3] <= 34
                and p[0]-2 <= x and x+w <= p[0]+p[2]+2
                and p[1]-4 <= y and y+h <= p[1]+p[3]+2]
            if len(complete) == 1:
                # Choose the independently bounded full plate before reading
                # its value. Native 480 text detection can turn 一 into '-'
                # with high confidence; never pick the more convenient score.
                px, py, pw, ph = complete[0]
                reread = _inventory_ocr(context, image[py:py+ph,px:px+pw].copy(),
                    'LiveOpeningIdentityTitlePlate', only_rec=True)
                if len(reread) != 1:
                    raise ValueError('native complete title plate is missing or ambiguous')
                result.append(SimpleNamespace(text=reread[0].text, score=reread[0].score,
                    box=complete[0], complete_title_plate=True,
                    acquisition_mode='native_complete_title_plate',
                    source_page_observation=dict(text=row.text, score=row.score, box=row.box)))
                continue
        # 1750's detector omitted one leading glyph although its suffix was
        # confident. Only a uniquely visible complete plate and a strongly
        # right-shifted box permit a full-plate read; no character is guessed.
        prefix_plates = [p for p in plates if 180 <= p[2] <= 190
            and 14 <= h <= 32 and 40 <= w <= 190
            and type(row.score) in (int,float) and math.isfinite(row.score)
            and .95 <= row.score <= 1
            and 18 <= x-p[0] and 0 <= p[0]+p[2]-x-w <= 6
            and p[1]-4 <= y and y+h <= p[1]+p[3]+2]
        if len(prefix_plates) == 1:
            px,py,pw,ph = prefix_plates[0]
            reread = _inventory_ocr(context,image[py:py+ph,px:px+pw].copy(),
                                    'LiveOpeningIdentityTitlePlate',only_rec=True)
            if (len(reread) == 1 and type(reread[0].score) in (int,float)
                    and math.isfinite(reread[0].score) and .95 <= reread[0].score <= 1):
                full = normalize_upgrade_title(reread[0].text)
                suffix = normalize_upgrade_title(row.text)
                if len(full) == len(suffix)+1 and full.endswith(suffix):
                    result.append(SimpleNamespace(text=reread[0].text,score=reread[0].score,
                        box=prefix_plates[0],complete_title_plate=True))
                    continue
        if (1090 <= y <= 1120 and 14 <= h <= 32 and 40 <= w <= 190
                and type(row.score) in (int,float) and math.isfinite(row.score)
                and 0 <= row.score < .95 and x >= 2 and x+w+2 <= 720):
            complete = [p for p in plates if 180 <= p[2] <= 190
                and p[0]-2 <= x and x+w <= p[0]+p[2]+2
                and p[1]-4 <= y and y+h <= p[1]+p[3]+2]
            # 1753's weak full-page first glyph differs from the real glyph.
            # A unique complete plate supplies independent text, still subject
            # to the caller's three-frame full-title and visual-family gates.
            box = complete[0] if len(complete)==1 else (x-2,y-2,w+4,h+4)
            px,py,pw,ph = box
            reread = _inventory_ocr(context,image[py:py+ph,px:px+pw].copy(),
                'LiveOpeningIdentityTitlePlate' if len(complete)==1 else 'LiveOpeningIdentityTitleTight',
                only_rec=True)
            if (len(reread)==1 and type(reread[0].score) in (int,float)
                    and math.isfinite(reread[0].score) and .95 <= reread[0].score <= 1
                    and (len(complete)==1 or
                         normalize_upgrade_title(reread[0].text)==normalize_upgrade_title(row.text))):
                row=SimpleNamespace(text=reread[0].text,score=reread[0].score,
                    box=box if len(complete)==1 else row.box,complete_title_plate=len(complete)==1)
        elif (1090 <= y <= 1120 and 32 < h <= 36 and 40 <= w <= 190
                and type(row.score) in (int,float) and 0 <= row.score <= 1
                and x >= 2 and x+w+2 <= 720):
            # 1106's complete title is 35px high. Require the independently
            # visible full plate and an exact crop reread; keep the raw box.
            matches=[plate for plate in plates
                if plate[0]-2 <= x and x+w <= plate[0]+plate[2]+2
                and plate[1]-4 <= y and y+h <= plate[1]+plate[3]+2]
            if len(matches)==1:
                if row.score < .95:
                    # Geometry proves a full weak row, not accepted text.
                    # Keep its confidence so only the existing complete-hand
                    # identity continuation may use it; fresh title reading
                    # still rejects it at the unchanged .95 threshold.
                    result.append(SimpleNamespace(text=row.text, score=row.score,
                        box=row.box, complete_title_plate=True))
                    continue
                reread=_inventory_ocr(context,image[y-2:y+h+2,x-2:x+w+2].copy(),
                    'LiveOpeningIdentityTitleTight',only_rec=True)
                if (len(reread)==1 and .95 <= reread[0].score <= 1
                        and normalize_upgrade_title(reread[0].text)==normalize_upgrade_title(row.text)):
                    row=SimpleNamespace(text=row.text,score=min(row.score,reread[0].score),
                        box=row.box,complete_title_plate=True)
        result.append(row)
    return tuple(result)


def retained_short_hand_titles(title_frames, card_ids, catalog, upgrades, *, excluded_slot=None):
    """Use already verified identities; weak OCR never becomes accepted text."""
    if not title_frames or len(card_ids) not in (1,2,3):
        raise ValueError('retained short-hand title evidence is incomplete')
    columns = (((240,480),) if len(card_ids)==1 else
        ((140,360),(360,590)) if len(card_ids)==2 else ((0,240),(240,480),(480,720)))
    observations = []
    for rows in title_frames:
        current = []
        for slot, ((left,right), card_id) in enumerate(zip(columns,card_ids,strict=True)):
            if slot == excluded_slot:
                current.append(dict(slot=slot, identity_source='fresh_selected_detail'))
                continue
            matching = [r for r in rows if _opening_title_row(r)
                and left <= r.box[0]+r.box[2]/2 < right]
            if len(matching)!=1:
                raise ValueError('retained hand title is missing or ambiguous')
            row = matching[0]
            if (not row.text or type(row.score) not in (int,float)
                    or not math.isfinite(row.score) or not 0 <= row.score <= 1):
                raise ValueError('retained hand title observation is invalid')
            entry = catalog.entries[card_id]
            original_title = normalize_upgrade_title(row.text)
            # The same complete retained title can use OCR's historical Han
            # glyphs (1915); 顏/顔 already occurs in the HIF choice reader.
            # Keep raw text and compare the entire name, including its +.
            title = original_title.translate(str.maketrans({'橫':'横','顏':'顔'}))
            if row.score >= .95 and title != entry.name:
                raise ValueError('confident visible title contradicts complete hand')
            upgrade = upgrades[slot]
            complete_fresh_title = (upgrade.accepted is True and upgrade.upgraded is None
                and upgrade.reason == 'live_upgrade_not_required'
                and upgrade.plus_votes == upgrade.normal_votes == 0
                and row.score >= .95 and title == entry.name)
            if not complete_fresh_title and (upgrade.accepted is not True or upgrade.upgraded is not entry.upgraded
                    or upgrade.plus_votes != (len(title_frames) if entry.upgraded else 0)
                    or upgrade.normal_votes != (0 if entry.upgraded else len(title_frames))):
                raise ValueError('retained short hand requires unanimous visible upgrade marks')
            current.append(dict(slot=slot,text=row.text,confidence=row.score,
                text_accepted=row.score>=.95,identity_source='retained_complete_hand',
                upgrade_source='complete_fresh_title' if complete_fresh_title else 'visible_upgrade_marks',
                **({'normalized_title':title,'title_normalization':'hif_complete_han_glyph_variants'}
                   if original_title != title else {})))
        observations.append(current)
    return dict(card_ids=tuple(card_ids),observations=observations)


def retained_selected_short_hand_titles(title_frames, families, card_ids, catalog, upgrades, *,
                                         selected_slot, selected_detail, frame_ids):
    """Retain verified neighbours; the selected card still requires fresh detail."""
    _verify_selected_short_hand_detail(title_frames, families, selected_slot=selected_slot,
        selected_detail=selected_detail, frame_ids=frame_ids)
    if (len(card_ids) != len(families) or selected_detail['card_id'] != card_ids[selected_slot]
            or any(card_id not in family for card_id,family in zip(card_ids,families,strict=True))):
        raise ValueError('selected detail or current families contradict retained hand')
    proof = retained_short_hand_titles(title_frames,card_ids,catalog,upgrades,excluded_slot=selected_slot)
    return dict(proof,selected_slot=selected_slot,selected_detail=selected_detail,frame_ids=tuple(frame_ids))


def isolated_preview_pixel_retry(transaction, baseline, threshold, *, selected_slot=None):
    """Reobserve non-glyph red noise in one region or selected frame, never accept it."""
    if (transaction.get('accepted') is not False or transaction.get('selected_slot') != selected_slot
            or type(threshold) not in (int,float) or not .85 <= threshold <= 1
            or baseline.get('accepted') is not True):
        return False
    # Compact-window profiles may use .85 for ordinary recognition. A retry
    # still requires the existing .95 observed confidence; it never accepts
    # this rejected batch or weakens the subsequent input checks.
    threshold = max(.95, threshold)
    if selected_slot is not None:
        preview = transaction.get('card_effect_preview') or {}
        costs = transaction.get('observations', {}).get('effective_cost', ())
        # Ordinary zero-cost cards use the legacy transaction without a
        # modeled preview. Their complete observed fee and unchanged resource
        # baseline can request the same retry, but cannot resolve noisy deltas.
        zero_cost = (transaction.get('card_effect_preview', False) is None
            and type(transaction.get('catalog_base_cost')) is int
            and transaction['catalog_base_cost'] == 0
            and transaction.get('catalog_stamina_recovery') == 0
            and all(type(transaction.get('expected_'+name+'_delta')) is int
                and transaction['expected_'+name+'_delta'] == 0
                and transaction.get('preview_'+name) == baseline.get(name)
                for name in ('stamina','shield')))
        cost = 0 if zero_cost else preview.get('effective_cost')
        if (type(selected_slot) is not int or not zero_cost and (
                preview.get('selected_slot') != selected_slot
                or any(preview.get('preview_'+name) != baseline.get(name) for name in ('stamina','shield')))
                or type(cost) is not int or len(costs) != 3
                or any(row.get('accepted') is not True or type(row.get('value')) is not int
                       or row['value'] != cost
                       or type(row.get('confidence')) not in (int,float)
                       or not threshold <= row['confidence'] <= 1 for row in costs)):
            return False
    observations = transaction.get('observations',{})
    absolutes = observations.get('absolutes',{})
    confidence = absolutes.get('confidence')
    if (absolutes.get('accepted') is not True or type(confidence) not in (int,float)
            or not threshold <= confidence <= 1):
        return False
    noisy_fields = set()
    noisy_frames = set()
    for name in ('stamina','shield'):
        if type(baseline.get(name)) is not int or absolutes.get(name) != baseline[name]:
            return False
        rows = observations.get(name+'_delta',{}).get('observations',())
        if len(rows) != 3:
            return False
        for index, row in enumerate(rows):
            evidence = row.get('red_evidence',{})
            pixels = evidence.get('raw_red_pixel_count')
            if (row.get('accepted') is not False or type(pixels) is not int or pixels < 0
                    or evidence.get('component_count') != 0 or evidence.get('pixel_count') != 0
                    or evidence.get('red_present') is not bool(pixels)):
                return False
            # Background colour may persist across several frames in one
            # region. This only admits the existing one fresh-batch retry;
            # no pixel is ignored and no missing delta is filled with zero.
            if pixels:
                noisy_fields.add(name)
                noisy_frames.add(index)
    #1208: one animation frame can cross both resource regions. The two
    # other frames must be clean, and every absolute/fee/model check above
    # remains mandatory. This only requests the existing one full-batch
    # retry; the rejected batch cannot supply zero deltas or input authority.
    return len(noisy_fields) == 1 or (selected_slot is not None and len(noisy_frames) == 1)
def selected_baseline_noise_reobserve_allowed(event, params):
    """Request one new selected baseline batch; never resolve noisy pixels to zero."""
    frames = event.get('frame_ids', ())
    selected = event.get('select_consensus') or {}
    votes = selected.get('observations', ())
    transaction = event.get('resource_transaction') or {}
    preview = transaction.get('card_effect_preview') or {}
    slot = preview.get('selected_slot')
    detail = event.get('detail_identity') or {}
    titles = detail.get('observations', ())
    selections = event.get('select_evidence', ())
    thresholds = [params.get(name) for name in ('debug_resource_absolute_min_confidence',
        'debug_displayed_effective_cost_min_confidence', 'debug_preview_delta_min_confidence')]
    if (any(type(value) not in (int,float) or not .85 <= value <= 1 for value in thresholds)
            or len(frames) != 3 or any(not isinstance(value,str) or not value for value in frames)
            or len(set(frames)) != 3 or event.get('geometry_stable') is not True
            or type(slot) is not int or slot < 0 or selected.get('accepted') is not True
            or selected.get('selected_slot') != slot or not isinstance(votes,(tuple,list))
            or len(votes) != 3 or any(not isinstance(vote,(tuple,list)) or tuple(vote) != (slot,) for vote in votes)
            or len(selections) != 3 or any(row.get('geometry_available') is not True
                or row.get('geometry_ambiguous') is not False or row.get('ocr_slots') != [slot]
                or row.get('geometry_raised_slots') != [slot] or row.get('resolved_slots') != [slot]
                for row in selections)
            or detail.get('accepted') is not True or detail.get('card_id') != preview.get('card_id')
            or len(titles) != 3 or [row.get('frame_id') for row in titles] != list(frames)
            or any(row.get('card_id') != detail.get('card_id') or row.get('roi') is None for row in titles)
            or transaction.get('accepted') is not False or transaction.get('selected_slot') != slot
            or transaction.get('frame_ids') != list(frames)
            or preview.get('source') != 'native_hif_card_effect_preview'):
        return False
    # The supported compact profile configures .85, but retry qualification
    # still needs the original .95 observed evidence. Never accept this batch.
    retry_thresholds = [max(.95, value) for value in thresholds]
    return (_selected_resource_noise_reobserve_allowed(transaction, retry_thresholds)
            or _selected_whole_shield_reobserve_allowed(transaction, thresholds))


def _selected_resource_noise_reobserve_allowed(transaction, thresholds):
    """Share conservation proof only; each caller validates its own frame/identity envelope."""
    preview = transaction.get('card_effect_preview') or {}
    if (transaction.get('accepted') is not False
            or preview.get('source') != 'native_hif_card_effect_preview'
            or len(thresholds) != 3
            or any(type(value) not in (int,float) or not .95 <= value <= 1 for value in thresholds)):
        return False
    absolute_threshold, cost_threshold, delta_threshold = thresholds
    observed = transaction.get('observations', {})
    absolutes, costs = observed.get('absolutes', {}), observed.get('effective_cost', ())
    absolute_rows = absolutes.get('observations', ())
    displayed = preview.get('displayed_card_cost', preview.get('effective_cost'))
    def confident(row, threshold):
        return (row.get('accepted') is True and type(row.get('confidence')) in (int,float)
            and threshold <= row['confidence'] <= 1)
    if (not confident(absolutes, absolute_threshold) or len(absolute_rows) != 3
            or type(displayed) is not int or displayed < 0 or len(costs) != 3
            or any(not confident(row,cost_threshold) or type(row.get('value')) is not int
                or row['value'] != displayed for row in costs)
            or any(transaction.get('conservation', {}).get(name) is not True
                   for name in ('declared_cost','modeled_card_preview'))):
        return False
    noisy_fields = 0
    for name in ('stamina','shield'):
        baseline, value = preview.get('baseline_'+name), preview.get('preview_'+name)
        if (type(baseline) is not int or type(value) is not int or baseline < 0 or value < 0
                or absolutes.get(name) != value or transaction.get('preview_'+name) != value
                or any(not confident(row.get(name,{}),absolute_threshold)
                    or row[name].get('value') != value for row in absolute_rows)):
            return False
        delta = observed.get(name+'_delta', {})
        rows = delta.get('observations', ())
        expected = value-baseline
        if len(rows) != 3 or transaction.get('expected_'+name+'_delta') != expected:
            return False
        if delta.get('accepted') is True:
            clean_zero = (expected == 0 and delta.get('reason') == 'resource_delta_explicit_clean_zero'
                and all(_explicit_clean_zero_observation(row) for row in rows))
            if (not confident(delta,delta_threshold) or delta.get('value') != expected
                    or transaction.get('conservation', {}).get(name) is not True
                    or not clean_zero and any(not confident(row,delta_threshold)
                        or row.get('value') != expected for row in rows)):
                return False
            continue
        if (expected != 0 or delta.get('accepted') is not False or delta.get('value') is not None
                or transaction.get('conservation', {}).get(name) is not False):
            return False
        noisy = 0
        for row in rows:
            red = row.get('red_evidence', {})
            pixels = red.get('raw_red_pixel_count')
            if (row.get('accepted') is not False or row.get('value') is not None
                    or red.get('accepted') is not False or type(pixels) is not int or pixels < 0
                    or red.get('red_present') is not bool(pixels) or red.get('bottom_guard_red_pixels') != 0
                    or red.get('component_count') != 0 or red.get('pixel_count') != 0
                    or red.get('components') != [] or red.get('box') is not None):
                return False
            reason = 'resource_delta_red_evidence_invalid' if pixels else 'resource_delta_red_evidence_absent'
            if row.get('reason') != reason or red.get('reason') != reason:
                return False
            noisy += bool(pixels)
        if noisy != 1:
            return False
        noisy_fields += 1
    return noisy_fields == 1


def _explicit_clean_zero_observation(row):
    red = row.get('red_evidence', {})
    return (row.get('accepted') is False and row.get('value') is None
        and row.get('reason') == 'resource_delta_red_evidence_absent'
        and red.get('reason') == 'resource_delta_red_evidence_absent'
        and red.get('accepted') is False and red.get('red_present') is False
        and all(red.get(key) == 0 for key in ('raw_red_pixel_count',
            'bottom_guard_red_pixels','component_count','pixel_count'))
        and red.get('components') == [] and red.get('box') is None)


def selected_refresh_noise_reobserve_allowed(event, params, slot, hand_count, *,
                                             phase='authoritative_refresh_b', baseline=None):
    """Qualify a rejected batch; its caller owns the unchanged retry budget."""
    from .hif_skill_support import _verify_selection

    frames = event.get('frame_ids', ())
    transaction = event.get('resource_transaction') or {}
    preview = transaction.get('card_effect_preview') or {}
    identities = event.get('identity_revalidation', ())
    boxes = event.get('candidate_boxes', ())
    if (phase not in ('authoritative_refresh_a', 'authoritative_refresh_b')
            or event.get('phase') != phase
            or phase == 'authoritative_refresh_a' and event.get('input_may_have_been_dispatched') is not False
            or event.get('reason') != 'authoritative_live_resource_transaction_unresolved'
            or event.get('accepted') is not False or event.get('click_dispatched') is not False
            or event.get('selected_identity_authorized') is not True
            or type(slot) is not int or type(hand_count) is not int or not 0 <= slot < hand_count
            or event.get('expected_selected_slot') != slot
            or transaction.get('selected_slot') != slot
            or (preview.get('selected_slot') != slot
                and not ((phase == 'authoritative_refresh_b' or baseline is not None)
                         and transaction.get('card_effect_preview', False) is None))
            or transaction.get('frame_ids') != list(frames)
            or len(boxes) != 3 or any(len(row) != hand_count for row in boxes)
            or any(len(box) != 4 or any(type(value) is not int for value in box)
                   or min(box[2:]) <= 0 for row in boxes for box in row)
            or len(identities) != hand_count
            or [row.get('slot') for row in identities] != list(range(hand_count))
            or any(row.get('accepted') is not True for row in identities)):
        return False
    try:
        _verify_selection(frames, slot, event)
    except (ValueError, TypeError, KeyError):
        return False
    thresholds = [params.get(name) for name in ('debug_resource_absolute_min_confidence',
        'debug_displayed_effective_cost_min_confidence', 'debug_preview_delta_min_confidence')]
    if any(type(value) not in (int,float) or not .85 <= value <= 1 for value in thresholds):
        return False
    # Match the selected-baseline retry contract for the supported compact
    # configuration; observed evidence remains at least .95 in A and B too.
    retry_thresholds = [max(.95, value) for value in thresholds]
    return (_selected_resource_noise_reobserve_allowed(transaction, retry_thresholds)
            or _selected_whole_shield_reobserve_allowed(transaction, thresholds)
            or baseline is not None
            and all(type(value) in (int,float) and .95 <= value <= 1 for value in thresholds)
            and isolated_preview_pixel_retry(transaction,baseline,max(thresholds),selected_slot=slot)
            or phase == 'authoritative_refresh_b'
            and _ordinary_zero_cost_whole_shield_reobserve_allowed(transaction, thresholds))


def _ordinary_zero_cost_whole_shield_reobserve_allowed(transaction, thresholds):
    """1656: qualify fresh B from observed zero cost; preserve unknown resources."""
    observed = transaction.get('observations', {})
    absolutes = observed.get('absolutes', {})
    rows = absolutes.get('observations', ())
    if (len(thresholds) != 3
            or any(type(value) not in (int, float) or not .95 <= value <= 1 for value in thresholds)
            or transaction.get('card_effect_preview', False) is not None
            or transaction.get('accepted') is not False
            or transaction.get('reason') != 'selected_resource_transaction_unresolved_or_nonconserving'
            or transaction.get('catalog_stamina_recovery') != 0
            or transaction.get('conservation') != dict(
                effective_cost=False, shield=False, shield_first_split=False, stamina=False)
            or absolutes.get('accepted') is not False
            or absolutes.get('reason') != 'live_resource_absolutes_unresolved' or len(rows) != 3):
        return False
    absolute_threshold, cost_threshold, _ = thresholds
    costs = observed.get('effective_cost', ())
    if (len(costs) != 3 or any(row.get('accepted') is not True
            or type(row.get('value')) is not int or row['value'] != 0
            or type(row.get('confidence')) not in (int, float)
            or not cost_threshold <= row['confidence'] <= 1 for row in costs)):
        return False
    weak = 0
    for name in ('stamina', 'shield'):
        values = []
        for row in rows:
            field = row.get(name, {})
            value, confidence = field.get('value'), field.get('confidence')
            if (field.get('accepted') is not True
                    or type(confidence) not in (int, float)
                    or not absolute_threshold <= confidence <= 1):
                if name != 'shield':
                    return False
                value = _unresolved_shield_retry_value(field)
                weak += 1
            if type(value) is not int or value < 0:
                return False
            values.append(value)
        if len(set(values)) != 1:
            return False
        delta = observed.get(name+'_delta', {})
        deltas = delta.get('observations', ())
        expected = transaction.get('expected_'+name+'_delta')
        if (type(expected) is not int or expected != 0
                or delta.get('accepted') is not False or delta.get('value') is not None
                or delta.get('reason') != 'resource_delta_absolute_unresolved'
                or len(deltas) != 3
                or not all(_explicit_clean_zero_observation(row) for row in deltas)):
            return False
    return weak == 1


def _selected_whole_shield_reobserve_allowed(transaction, thresholds):
    """One agreeing weak whole shield may request fresh frames, never a value."""
    preview = transaction.get('card_effect_preview') or {}
    observed = transaction.get('observations', {})
    absolutes = observed.get('absolutes', {})
    rows = absolutes.get('observations', ())
    if (len(thresholds) != 3
            or any(type(value) not in (int,float) or not .85 <= value <= 1 for value in thresholds)
            or transaction.get('accepted') is not False
            or transaction.get('reason') != 'selected_resource_transaction_unresolved_or_nonconserving'
            or preview.get('source') != 'native_hif_card_effect_preview'
            or absolutes.get('accepted') is not False
            or absolutes.get('reason') != 'live_resource_absolutes_unresolved' or len(rows) != 3
            or transaction.get('conservation') != dict(
                declared_cost=True, modeled_card_preview=False, shield=False, stamina=False)):
        return False
    absolute_threshold, _, delta_threshold = [max(.95,value) for value in thresholds]
    def confident(row, value, threshold):
        return (row.get('accepted') is True and type(row.get('value')) is int
            and row['value'] == value and type(row.get('confidence')) in (int,float)
            and threshold <= row['confidence'] <= 1)
    displayed = preview.get('displayed_card_cost', preview.get('effective_cost'))
    costs = observed.get('effective_cost', ())
    if (type(displayed) is not int or displayed < 0 or len(costs) != 3
            # Payment is already independently specified by the model. As in
            # the normal transaction, OCR corroborates the same number; it
            # does not impose another confidence gate on a read-only retry.
            or any(not confident(row, displayed, 0) for row in costs)):
        return False
    weak = 0
    for name in ('stamina','shield'):
        baseline, value = preview.get('baseline_'+name), preview.get('preview_'+name)
        if (type(baseline) is not int or type(value) is not int or min(baseline,value) < 0
                or transaction.get('expected_'+name+'_delta') != value-baseline):
            return False
        for row in rows:
            field = row.get(name, {})
            if confident(field, value, absolute_threshold):
                continue
            fragmented = (field.get('accepted') is False and field.get('value') is None
                and field.get('confidence') == 0
                and field.get('reason') == 'hif_shifted_shield_detection_unresolved'
                and (field.get('acquisition_mode') == 'hif_resampled_shield_contrast_detection'
                    or field.get('acquisition_mode') == 'direct_calibrated_hif_shield'
                    and field.get('source_acquisition_mode') == 'hif_resampled_shield_contrast_detection')
                and field.get('roi') == [660,177,55,31]
                and type(field.get('observed_result_count')) is int
                and 2 <= field['observed_result_count'] <= 3
                and len(field.get('raw_results', ())) == field['observed_result_count'])
            # A split detection has no numeric value. Two other complete
            # frames and the unchanged model may request one fresh batch;
            # never join fragments or relabel this frame as accepted.
            if name != 'shield' or not (fragmented or _unresolved_shield_retry_value(field) == value):
                return False
            weak += 1
        delta = observed.get(name+'_delta', {})
        deltas = delta.get('observations', ())
        if (delta.get('accepted') is not False or delta.get('value') is not None
                or delta.get('reason') != 'resource_delta_absolute_unresolved' or len(deltas) != 3):
            return False
        for row in deltas:
            if confident(row, value-baseline, delta_threshold):
                continue
            red = row.get('red_evidence', {})
            if (value != baseline or row.get('accepted') is not False or row.get('value') is not None
                    or row.get('reason') != 'resource_delta_red_evidence_absent'
                    or red.get('reason') != 'resource_delta_red_evidence_absent'
                    or red.get('accepted') is not False or red.get('red_present') is not False
                    or any(red.get(key) != 0 for key in ('raw_red_pixel_count',
                        'bottom_guard_red_pixels','component_count','pixel_count'))
                    or red.get('components') != [] or red.get('box') is not None):
                return False
    return weak == 1


def _unresolved_shield_retry_value(field):
    """1590: agreeing weak whole reads may request new frames, never a value."""
    if (field.get('accepted') is not False or field.get('value') is not None
            or field.get('reason') != 'hif_shield_complete_roi_unresolved'):
        return None
    original = field.get('original_observation', {})
    complete = field.get('complete_roi_reread', {})
    contrast = field.get('complete_contrast_reread', {})
    complete_box = complete.get('box')
    # Both complete badge layouts are already produced by _live_field_roi;
    # the expanded layout must keep its entire width in the contrast read.
    if (len(original.get('raw_results', ())) != 1
            or contrast.get('raw_result_count') != 1
            or complete_box not in ([660, 175, 55, 34], [650, 175, 65, 34])
            or contrast.get('box') != [0, 0, complete_box[2], complete_box[3]]):
        return None
    values = []
    for observed in (original, complete, contrast):
        value, confidence = observed.get('value'), observed.get('confidence')
        if (observed.get('accepted') is not True or observed.get('observed_result_count') != 1
                or type(value) is not int or value < 0
                or type(confidence) not in (int, float) or not 0 <= confidence < .95):
            return None
        values.append(value)
    return values[0] if len(set(values)) == 1 else None


def _faint_shield_noise_retry_value(field):
    """1944: a faint disjoint border mark can request a wholly new batch."""
    import re

    if (field.get('accepted') is not False or field.get('value') is not None
            or field.get('reason') != 'hif_shifted_shield_detection_unresolved'
            or field.get('acquisition_mode') != 'hif_resampled_shield_contrast_detection'
            or field.get('roi') != [660,177,55,31]):
        return None
    raw = field.get('raw_results', ())
    if len(raw) != 2:
        return None
    strong = [row for row in raw if isinstance(row.get('text'),str) and re.fullmatch(r'\d{1,3}',row['text'])
        and type(row.get('confidence')) in (int,float) and .95 <= row['confidence'] <= 1]
    if len(strong) != 1:
        return None
    number = strong[0]
    other = next(row for row in raw if row is not number)
    box, noise = number.get('box',()), other.get('box',())
    if (len(box) != 4 or len(noise) != 4 or any(type(v) is not int for v in (*box,*noise))
            or not 20 <= box[3] <= 31 or box[0] < 660 or box[0]+box[2] > 715
            or noise[0] < box[0]+box[2] or noise[0]+noise[2] > 715
            or not 0 < noise[3] <= 13 or noise[1] < 177 or noise[1]+noise[3] > 208
            or type(other.get('confidence')) not in (int,float) or not 0 <= other['confidence'] < .5):
        return None
    return int(number['text'])


def resource_baseline_reobserve_allowed(baseline, threshold, *, selected_slot=None):
    """Qualify unread fields or missing geometry for a fresh whole batch."""
    if type(threshold) not in (int,float) or not .85 <= threshold <= 1:
        return False
    threshold = max(.95, threshold)
    rows=baseline.get('resource_observations',())
    if len(rows)!=3:
        return False
    weak=0
    missing=0
    for name in ('stamina','shield'):
        values=[]
        for row in rows:
            field=row.get(name,{})
            value=field.get('value');confidence=field.get('confidence')
            if field.get('accepted') is not True:
                # Whole-ROI OCR can reject punctuation without producing a
                # numeric value. It cannot authorize input, but need not block
                # a bounded fresh batch when another frame supplies the field.
                if (selected_slot is None and baseline.get('accepted') is False
                        and field.get('accepted') is False and value is None
                        and field.get('reason') == 'live_field_ocr_missing'
                        and field.get('observed_result_count') == 0
                        and field.get('candidate_values') == []):
                    missing += 1
                    weak += 1
                    continue
                # Only the existing unselected retry can consume this weak
                # diagnostic; leave the failed baseline unknown and untouched.
                if (name != 'shield' or selected_slot is not None
                        or baseline.get('accepted') is not False
                        or baseline.get('reason') != 'live_resource_baseline_unresolved'):
                    return False
                value = _unresolved_shield_retry_value(field)
                if value is None:
                    value = _faint_shield_noise_retry_value(field)
                confidence = 0.0
            if (type(value) is not int
                    or type(confidence) not in (int,float) or not 0 <= confidence <= 1):
                return False
            values.append(value)
            weak += confidence < threshold
        if not values or len(set(values))!=1:
            return False
    if baseline.get('geometry_stable') is False:
        # 1104: two complete identical rows, one empty row caused by a
        # transient extra detector box. No old frame authorizes input: the
        # existing caller captures an entirely new batch once and rechecks
        # geometry, SELECT and resources before starting fresh A/B.
        boxes=baseline.get('candidate_boxes',())
        frame_ids=baseline.get('frame_ids',())
        present=[row for row in boxes if row]
        matching = (len(present)==2 and 1 <= len(present[0]) <= 5
                    and len(present[0])==len(present[1]))
        if matching and present[0]!=present[1]:
            from types import SimpleNamespace

            from .selection import same_live_candidate_geometry

            # A retry is not an input authorization. Use the same physical
            # geometry comparison as the live reader, not exact x/y equality.
            matching = (all(len(box) == 4 and all(type(value) is int for value in box)
                and box[2:] in ([190, 250], (190, 250)) for row in present for box in row)
                and same_live_candidate_geometry(*(tuple(SimpleNamespace(box=box) for box in row)
                    for row in present)))
        return (selected_slot is None and weak==0
            and baseline.get('accepted') is False
            and len(boxes)==3 and len(present)==2
            and matching
            and len(frame_ids)==3 and all(isinstance(value,str) and value for value in frame_ids)
            and len(set(frame_ids))==3)
    selection=baseline.get('select_consensus') or {}
    evidence=baseline.get('select_evidence',())
    boxes=baseline.get('candidate_boxes',())
    if (selected_slot is None and weak == 0 and baseline.get('accepted') is False
            and baseline.get('geometry_stable') is True and selection.get('accepted') is False
            and selection.get('reason') == 'multiple_select_labels'
            and len(evidence) == len(boxes) == 3 and 1 <= len(boxes[0]) <= 5
            and all(len(group) == len(boxes[0]) for group in boxes)
            and all(e.get('ocr_slots') == [] and e.get('geometry_available') is True
                and e.get('geometry_raised_slots') == [] and e.get('geometry_slot') is None
                and e.get('used_geometry_fallback') is False
                and (e.get('geometry_ambiguous') is False and e.get('resolved_slots') == []
                    or e.get('geometry_ambiguous') is True
                    and e.get('resolved_slots') == list(range(len(boxes[0])))) for e in evidence)
            and any(e.get('geometry_ambiguous') is False for e in evidence)):
        # Cropped NN boxes can report every slot as possible despite no SELECT
        # label or raised card. Only allow the existing bounded fresh batch;
        # this failed batch still supplies no input authorization.
        return True
    return ((weak==1 or missing > 0 and weak==missing) and baseline.get('geometry_stable') is True
        and selection.get('accepted') is True
        and selection.get('selected_slot')==selected_slot)


def selected_resource_reobserve_allowed(event, params, slot):
    transaction=event.get('resource_transaction',{})
    conservation=transaction.get('conservation',{})
    if (type(slot) is not int or event.get('reason') != 'authoritative_live_resource_absolute_confidence_unapproved'
            or transaction.get('accepted') is not True or not conservation
            or any(value is not True for value in conservation.values())
            or transaction.get('selected_slot') != slot):
        return False
    for field,gate in (('displayed_effective_cost_confidence','debug_displayed_effective_cost_min_confidence'),
                       ('preview_delta_confidence','debug_preview_delta_min_confidence')):
        value,threshold=transaction.get(field),params.get(gate)
        if (type(value) not in (int,float) or type(threshold) not in (int,float)
                or not .95 <= threshold <= value <= 1):
            return False
    return resource_baseline_reobserve_allowed(dict(
        geometry_stable=event.get('selected_identity_authorized'),
        select_consensus=event.get('select_consensus',{}),
        resource_observations=transaction.get('observations',{}).get('absolutes',{}).get('observations',())),
        params.get('debug_resource_absolute_min_confidence'),selected_slot=slot)
