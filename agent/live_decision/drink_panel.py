"""HIF drink inspection and submission through the audited Maa controller."""

import json
import math
import time
from types import SimpleNamespace
from functools import partial

import cv2
import numpy as np

from .layout import calibrated_live_client_size
from .inventory import _inventory_ocr, capture_initial_live_checkpoint, fresh_identity_reobserve_allowed, unresolved_live_resource_evidence
from .drink_icons import EVENT_GREEN_JUICE_ID, EVENT_GREEN_JUICE_NAME
from .status_panel import StatusInspectionError, _inside, _normalize_status_text
from .hif_capture_adapter import captured_at as frame_captured_at

_DRINK_SLOT_REGIONS = tuple((x, 1154, 90, 90) for x in (17, 102, 187, 273))
_SMALL_BOTTLE_RING_CLIENTS = ((359, 638), (360, 640))


def _occupied_drink_templates(client_size):
    small = calibrated_live_client_size(client_size) == (480,853)
    names = [f'produce/hif_live_drink_{i}_480.png' if small else f'produce/drink_{i}.png' for i in (1, 2, 3)]
    if client_size == (359, 638):
        # Same white ring mask sampled at the lower native resolution.
        names[0] = 'produce/hif_live_drink_1_360.png'
    return names


def drink_slot_targets(context, image):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('drink slot resolution is uncalibrated')
    slots = []
    client_size = getattr(context,'gkh_hif_client_size',(720,1280))
    small = calibrated_live_client_size(client_size) == (480,853)
    templates = _occupied_drink_templates(client_size)
    for index, roi in enumerate(_DRINK_SLOT_REGIONS):
        node = f'LiveStatusDrinkSlot{index}'
        hit = context.run_recognition(node, image, pipeline_override={node: dict(
            recognition='TemplateMatch', roi=list(roi),
            template=templates,
            green_mask=True, threshold=.8)})
        found = list(hit.filtered_results) if hit and hit.hit else []
        boxes = []
        for row in found:
            box = tuple(row.box)
            if (len(box) != 4 or any(type(v) is not int for v in box)
                    or min(box[2:]) <= 0 or not _inside({'box': box}, roi)):
                raise ValueError('drink control is outside its slot')
            # Apply the existing calibrated position contract to every match
            # before deciding uniqueness. A different ring template may match
            # a displaced part of the same control, outside this contract.
            if small and (abs(box[0]-(roi[0]+10))>2 or abs(box[1]-1165)>2):
                continue
            boxes.append(box)
        if len(boxes) > 1:
            raise ValueError('drink slot control is ambiguous')
        box = boxes[0] if boxes else None
        # Missing the occupied-slot anchor is UNKNOWN, not EMPTY.
        slots.append(box)
    return tuple(slots)


def drink_controls_consistent(observations):
    """Compare slot presence and size, allowing at most two pixels of jitter.

    Masked ring matching can alternate adjacent pixel peaks on an unchanged
    control. Compare the entire batch's span, not cumulative pairwise drift.
    Every input box still comes from its independently accepted slot match.
    """
    if not observations or any(len(row) != 4 for row in observations):
        return False
    for boxes in zip(*observations, strict=True):
        if any(box is None for box in boxes):
            if not all(box is None for box in boxes):
                return False
            continue
        if any(box[2:] != boxes[0][2:] for box in boxes[1:]):
            return False
        if any(max(box[axis] for box in boxes)-min(box[axis] for box in boxes) > 2 for axis in (0, 1)):
            return False
    return True


def capture_stable_drink_controls(context, capture_frames, checkpoint, event):
    """One outlier allows fresh evidence, never a voted control or new input."""
    baseline = None
    for attempt in range(3):
        frames,state = capture_initial_live_checkpoint(capture_frames,checkpoint,event)
        current = tuple(state[k] for k in ('families','fields'))
        if baseline is not None and current != baseline:
            raise ValueError('Live changed while drink controls were settling')
        baseline = current
        slots = [drink_slot_targets(context,frame) for frame in frames]
        if drink_controls_consistent(slots):
            return frames,state,slots
        event.setdefault('control_rejections',[]).append(dict(frame_ids=state['frame_ids'],slots=slots))
        if attempt==2 or not any(drink_controls_consistent([row,other])
                for index,row in enumerate(slots) for other in slots[index+1:]):
            raise ValueError('drink controls are unresolved')
    raise ValueError('drink control observation budget exhausted')


def _planned_drink_target(context, slot, fresh, frame_identifier, event):
    """Locate the policy's physical bottle on the first readable current frame."""
    roi = _DRINK_SLOT_REGIONS[slot]
    deadline = time.perf_counter() + 4.0
    while time.perf_counter() < deadline and getattr(context.tasker, 'stopping', False) is not True:
        image, captured_at = fresh()
        node = 'ProduceCheckDrinkButton'
        size = getattr(context, 'gkh_hif_client_size', None)
        if size in _SMALL_BOTTLE_RING_CLIENTS:
            upstream = context.get_node_data(node)['recognition']
            params = dict(upstream['param'], template=_occupied_drink_templates(size))
            hit = context.run_recognition(node, image, pipeline_override={node: {
                'recognition': dict(type=upstream['type'], param=params)}})
        else:
            hit = context.run_recognition(node, image)
        rows = hit.filtered_results if hit and hit.hit else ()
        boxes = sorted((tuple(row.box) for row in rows), key=lambda box: box[0])
        event.update(slot=slot, before_frame_ids=[frame_identifier(image)], control_observations=[boxes])
        targets = [box for box in boxes if _inside(
            {'box': (box[0] + box[2] // 2, box[1] + box[3] // 2, 1, 1)}, roi)]
        if len(targets) == 1:
            event['preinput_controls'] = boxes
            return targets[0], captured_at
        time.sleep(.2)
    raise ValueError('requested drink slot has no unique observed occupied control')


def empty_drink_slot_targets(image, template, *, client_size=None, render_profile='standard'):
    """Positive empty-cup evidence, independent of occupied-anchor misses."""
    if getattr(image,'shape',())!=(1280,720,3) or getattr(template,'shape',())!=(45,36):
        raise ValueError('empty drink glyph layout is uncalibrated')
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    reference=template.astype(np.float32)
    compact = (isinstance(client_size, tuple) and len(client_size) == 2
        and all(type(value) is int for value in client_size)
        and (calibrated_live_client_size(client_size) == (480,853)
             or client_size in ((540,960),(541,961),(541,962),(542,962))))
    threshold = .90 if compact else .95
    if (render_profile not in ('standard', 'compact_route_opacity_binomial3')
            or render_profile != 'standard' and calibrated_live_client_size(client_size) != (480,853)):
        raise ValueError('empty drink render profile is uncalibrated')
    route_raster = render_profile == 'compact_route_opacity_binomial3'
    # The normalized 480px client gives adjacent slots different horizontal
    # half-pixel sampling phases (884 slot 2 versus the slot 3 reference).
    # Keep the same full cup; merge bounded phase responses before applying
    # the profile threshold and unchanged unique-target check.
    phases=(reference,*(cv2.warpAffine(reference,np.array([[1,0,dx],[0,1,0]],dtype=np.float32),
        (36,45),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0) for dx in (-.5,.5)))
    matched_phases=tuple(cv2.GaussianBlur(phase,(3,3),0) for phase in phases) if route_raster else phases
    slots=[]
    for x,anchor_x in zip((17,102,187,273),(44,129,214,300),strict=True):
        crop=image[1154:1244,x:x+90]
        # A translucent white stroke obeys the same alpha equation in the
        # minimum RGB channel. It retains contrast on saturated PV lighting
        # and skin where grayscale compression obscures an intact cup (940).
        target=None
        gray_score=None
        contrasts = (('minimum_channel_opacity_binomial3',crop.min(axis=2)),) if route_raster else (
            ('grayscale',cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)), ('minimum_channel',crop.min(axis=2)))
        for basis,contrast in contrasts:
            glyphs=cv2.morphologyEx(contrast,cv2.MORPH_TOPHAT,kernel)
            # The empty cup is translucent white: foreground contrast equals
            # alpha*(255-background). Remove that background dependence (865)
            # while retaining the full glyph and raw contrast check.
            background=contrast.astype(np.float32)-glyphs
            opacity=glyphs.astype(np.float32)/np.maximum(255-background,1)
            measured_opacity=cv2.GaussianBlur(opacity,(3,3),0) if route_raster else opacity
            responses=[cv2.matchTemplate(measured_opacity,phase,cv2.TM_CCOEFF_NORMED) for phase in matched_phases]
            response=np.maximum.reduce(responses)
            _,score,_,(dx,dy)=cv2.minMaxLoc(response)
            if basis=='grayscale':gray_score=float(score)
            mask=(np.isfinite(response)&(response>=threshold)).astype(np.uint8)
            count,_=cv2.connectedComponents(mask)
            if count>2:
                raise ValueError('empty drink glyph has multiple possible targets')
            if count==1 or float(glyphs[dy:dy+45,dx:dx+36].std())<5:
                continue
            if compact and (abs(x+dx-anchor_x)>2 or abs(1154+dy-1177)>2):
                continue
            proof={}
            if route_raster:
                phase_index=int(np.argmax([value[dy,dx] for value in responses]))
                observed=measured_opacity[dy:dy+45,dx:dx+36]
                chosen=matched_phases[phase_index]
                parts=[float(cv2.matchTemplate(observed[top:bottom],chosen[top:bottom],cv2.TM_CCOEFF_NORMED)[0,0])
                    for top,bottom in ((0,15),(15,30),(30,45))]
                if not np.isfinite(parts).all() or min(parts)<.8:
                    continue
                proof=dict(normalization_profile=render_profile, normalization_kernel='binomial_3x3',
                    sampling_phase_x=(0,-.5,.5)[phase_index], component_correlations=parts,
                    raw_confidence=float(cv2.matchTemplate(opacity[dy:dy+45,dx:dx+36],phases[phase_index],cv2.TM_CCOEFF_NORMED)[0,0]))
            target=dict(box=(x+dx,1154+dy,36,45),confidence=float(score),
                        grayscale_confidence=gray_score,contrast_basis=basis,
                        acceptance_threshold=threshold,client_size=client_size if compact else None,**proof)
            break  # Preserve a passed grayscale proof; no pixels are combined.
        slots.append(target)
    return tuple(slots)


def _drink_occupancy_row(occupied, empty, *, allow_unknown=False):
    row = []
    for slot, (filled, cup) in enumerate(zip(occupied, empty, strict=True)):
        if filled is not None and cup is not None:
            if not allow_unknown:
                raise ValueError('drink slot has contradictory occupied and empty evidence')
            row.append(dict(slot=slot, state='UNKNOWN', box=None, conflicting_controls=True))
            continue
        kind = 'OCCUPIED' if filled is not None else 'EMPTY' if cup is not None else 'UNKNOWN'
        row.append(dict(slot=slot, state=kind, box=filled if filled is not None else cup['box'] if cup else None,
                        empty_confidence=cup['confidence'] if cup else None,
                        empty_grayscale_confidence=cup.get('grayscale_confidence') if cup else None,
                        empty_contrast_basis=cup.get('contrast_basis') if cup else None,
                        empty_acceptance_threshold=cup.get('acceptance_threshold') if cup else None,
                        empty_client_size=cup.get('client_size') if cup else None))
    return row


def observe_drink_slot_occupancy(context, *, capture_frames, checkpoint, template, accepted_frames=None,
                                 single_frame=False):
    """Zero-input physical slot observation; does not supply drink identities."""
    started=time.perf_counter()
    event=dict(phase='drink_slot_occupancy',accepted=False,input_click_count=0,
               drinks_consumed=0,full_drink_state_ready=False,strategy_state_bound=False,observations=[])
    frames=()
    baseline=None
    known={}
    seen_frames=set()
    recovery=None
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    client_size=getattr(context,'gkh_hif_client_size',None)

    def empty_targets(frame):
        if (isinstance(client_size,tuple) and len(client_size)==2
                and all(type(value) is int for value in client_size)
                and (calibrated_live_client_size(client_size) == (480,853)
                     or client_size in ((540,960),(541,961),(541,962),(542,962)))):
            return empty_drink_slot_targets(frame,template,client_size=client_size)
        return empty_drink_slot_targets(frame,template)

    def motion(left,right,box):
        x,y,width,height=box
        a=left[y:y+height,x:x+width].astype(np.int16)
        b=right[y:y+height,x:x+width].astype(np.int16)
        return float((np.max(np.abs(a-b),axis=2)>32).mean())

    try:
        if single_frame:
            frames = capture_frames()
            if len(frames) != 1:
                raise ValueError('HUD drink occupancy requires one current frame')
            frame, = frames
            state = checkpoint(frames, single_frame=True)
            if len(state['frame_ids']) != 1 or not state['frame_ids'][0]:
                raise ValueError('HUD drink occupancy lacks one current frame ID')
            row = _drink_occupancy_row(drink_slot_targets(context, frame), empty_targets(frame), allow_unknown=True)
            event.update(accepted=True, occupancy_complete=all(slot['state'] != 'UNKNOWN' for slot in row),
                         slots=row, frame_ids=state['frame_ids'], elapsed_ms=(time.perf_counter()-started)*1000)
            if accepted_frames is not None:
                accepted_frames.append(frame)
            return [event]
        for attempt in range(4):
            if not upstream and time.perf_counter()-started>=15:
                raise ValueError('drink occupancy observation deadline expired')
            complete = getattr(checkpoint, '__dict__', {}).get('capture_complete')
            frames,state=capture_initial_live_checkpoint(capture_frames,checkpoint,event,
                max_attempts=1 if recovery is not None and not callable(complete) else 3)
            if not upstream and time.perf_counter()-started>=15:
                raise ValueError('drink occupancy observation deadline expired')
            current=(state['families'],state['fields'])
            if baseline is None:baseline=current
            elif current!=baseline:raise ValueError('Live changed while drink occupancy was settling')
            ids=state['frame_ids']
            if not frames or len(ids)!=len(frames) or len(set(ids))!=len(ids) or seen_frames.intersection(ids):
                raise ValueError('drink occupancy requires entirely new frames')
            if recovery is not None:
                changes={slot:motion(recovery['frame'],frames[0],box)
                         for slot,box in recovery['boxes'].items()}
                event['readability_recovery']['new_frame_ids']=list(ids)
                event['readability_recovery']['new_motion_fraction']=changes
                if any(value<.25 for value in changes.values()):
                    raise ValueError('drink unreadable glyph background has not changed')
            seen_frames.update(state['frame_ids'])
            readings = [_drink_occupancy_row(drink_slot_targets(context, frame), empty_targets(frame)) for frame in frames]
            for row in readings:
                for slot in row:
                    if slot['state']=='UNKNOWN':continue
                    previous=known.setdefault(slot['slot'],dict(state=slot['state'],boxes=[]))
                    previous['boxes'].append(tuple(slot['box']))
                    if (slot['state']!=previous['state']
                            or any(box[2:]!=previous['boxes'][0][2:] for box in previous['boxes'])
                            or any(max(box[axis] for box in previous['boxes'])-
                                   min(box[axis] for box in previous['boxes'])>2 for axis in (0,1))):
                        raise ValueError('drink occupancy known slot changed across observations')
            stable=all(a['state']==b['state'] and a['state']!='UNKNOWN' and
                       max(abs(x-y) for x,y in zip(a['box'],b['box']))<=2
                       for row in readings for a,b in zip(readings[0],row,strict=True))
            event['observations'].append(dict(frame_ids=state['frame_ids'],readings=readings,accepted=stable))
            if not upstream and time.perf_counter()-started>=15:
                raise ValueError('drink occupancy observation deadline expired')
            if stable:
                event.update(accepted=True,occupancy_complete=True,slots=readings[-1],
                             frame_ids=state['frame_ids'],elapsed_ms=(time.perf_counter()-started)*1000)
                if accepted_frames is not None:
                    accepted_frames.extend(frames)
                return [event]
            if attempt==2 and not upstream:
                unknown={slot['slot'] for row in readings for slot in row if slot['state']=='UNKNOWN'}
                # Motion authorizes one new observation, never an EMPTY fact.
                # Only an already positively seen full cup may enter this
                # recovery; ring misses, static unknowns and conflicts stop.
                if not unknown or any(known.get(slot,{}).get('state')!='EMPTY' for slot in unknown):break
                boxes={slot:known[slot]['boxes'][-1] for slot in unknown}
                changes={slot:[motion(a,b,box) for a,b in zip(frames,frames[1:])]
                         for slot,box in boxes.items()}
                recovery=dict(frame=frames[-1],boxes=boxes)
                event['readability_recovery']=dict(reason='moving_background_over_observed_empty_cup',
                    failed_frame_ids=list(state['frame_ids']),boxes=boxes,motion_fraction=changes,
                    maximum_extra_batches=1,deadline_ms=15000)
                # A changed PV background can still hide cup strokes. Only
                # spend the final batch after all previously seen empty cups
                # are readable together. Probes schedule, never prove slots.
                maximum_probes=max(0,math.ceil((started+15-time.perf_counter())/.08))
                waiting=dict(probe_count=0,maximum_probes=maximum_probes,poll_interval_ms=80,
                             changed=False,readable=False,purpose='recovery_schedule_only')
                event['readability_recovery']['background_wait']=waiting
                empty_boxes={slot:item['boxes'][-1] for slot,item in known.items() if item['state']=='EMPTY'}
                for _ in range(maximum_probes):
                    if time.perf_counter()-started>=15:
                        raise ValueError('drink occupancy observation deadline expired')
                    time.sleep(min(.08,max(0,started+15-time.perf_counter())))
                    if time.perf_counter()-started>=15:
                        raise ValueError('drink occupancy observation deadline expired')
                    probe=context.tasker.controller.post_screencap().wait().get()
                    waiting['probe_count']+=1
                    if time.perf_counter()-started>=15:
                        raise ValueError('drink occupancy observation deadline expired')
                    if not isinstance(probe,np.ndarray) or probe.shape!=frames[-1].shape:
                        raise ValueError('drink background probe layout is uncalibrated')
                    waiting['motion_fraction']={slot:motion(frames[-1],probe,box)
                        for slot,box in boxes.items()}
                    waiting['changed']=all(value>=.25 for value in waiting['motion_fraction'].values())
                    if not waiting['changed']:
                        continue
                    cups=empty_targets(probe)
                    waiting['readable_empty_slots']=[slot for slot,box in empty_boxes.items()
                        if cups[slot] is not None and tuple(cups[slot]['box'][2:])==tuple(box[2:])
                        and all(abs(cups[slot]['box'][axis]-box[axis])<=2 for axis in (0,1))]
                    waiting['readable']=len(waiting['readable_empty_slots'])==len(empty_boxes)
                    if time.perf_counter()-started>=15:
                        raise ValueError('drink occupancy observation deadline expired')
                    if waiting['readable']:
                        break
                if not waiting['changed'] or not waiting['readable']:
                    raise ValueError('drink background observation budget exhausted')
            if attempt==3:break
            time.sleep(.08)
        raise ValueError('drink occupancy has unresolved or unstable slots')
    except Exception as error:
        sink=getattr(context,'gkh_hif_status_failure_sink',None)
        if callable(sink):
            for frame in frames:
                sink(frame)
        event.update(reason=str(error),elapsed_ms=(time.perf_counter()-started)*1000)
        raise StatusInspectionError(str(error),[event]) from error


def _wait_drink_panel(context, fresh, *, opened, timeout, open_probe=None):
    from .hif_upstream_ui import HIF_SKIP_RECOGNITION, upstream_drink_actions

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and getattr(context.tasker, 'stopping', False) is not True:
        frame = fresh()
        image, _ = frame
        if open_probe is not None:
            if open_probe(frame):
                return True
            time.sleep(.2)
            continue
        actions = upstream_drink_actions(context, image)
        matches = opened and len(actions) == 2
        if not opened and not actions:
            skip = 'ProduceRecognitionSkipRound'
            battle = context.run_recognition(skip, image, pipeline_override={skip: HIF_SKIP_RECOGNITION})
            move = context.run_recognition('ProduceRecognitionChooseMoveCards', image)
            matches = bool(battle and battle.hit or move and move.hit)
        if matches:
            if opened:
                time.sleep(.5)
            return True
        time.sleep(.2)
    return False


def _miyu_drink_detail(context, fresh, names, event, frame_identifier, *, allow_raw=False, verify_open=False):
    """Use fb8's heading-relative name ROI; the first catalog match suffices."""
    for attempt in range(2):
        image, _ = fresh()
        if verify_open:
            from .hif_upstream_ui import upstream_drink_actions

            if set(upstream_drink_actions(context, image)) != {'use', 'cancel'}:
                continue
        try:
            rows = _inventory_ocr(context, image, 'ProduceRecognitionScore',
                                  roi=[20,560,680,430], expected=r'[^\n]+')
        except RuntimeError:
            rows = ()
        headings = [r for r in rows if r.box[0] < 130 and 'Pドリンク' in _normalize_status_text(r.text)]
        if headings:
            title_y = min(r.box[1] for r in headings)
            candidates = [r for r in rows if 170 <= r.box[0] <= 540 and 50 <= r.box[1]-title_y <= 110
                          and r.box[3] >= 20]
            if candidates:
                row = min(candidates, key=lambda candidate: candidate.box[1])
                name = _normalize_status_text(row.text)
                if name in names or (allow_raw and attempt == 1 and name):
                    event['page_frame_ids'] = [frame_identifier(image)]
                    return dict(name=name, confidence=float(row.score), name_box=tuple(row.box))
        if attempt == 0:
            time.sleep(.2)
    raise ValueError('drink name is unresolved')


def close_miyu_drink_panel(context, *, fresh, click, frame_identifier, names=(), observed_target_only=False):
    """fb8 _close_drink_detail using the caller's audited capture and click."""
    from .hif_upstream_ui import upstream_drink_actions

    identity = None
    for attempt in range(3):
        if getattr(context.tasker, 'stopping', False) is True:
            return False
        if attempt == 2 and _wait_drink_panel(context, fresh, opened=False, timeout=10):
            return True
        image, _ = fresh()
        if 'cancel' not in upstream_drink_actions(context, image):
            return _wait_drink_panel(context, fresh, opened=False, timeout=3)
        try:
            current = _miyu_drink_detail(context, fresh, names, {}, frame_identifier, allow_raw=True)['name']
        except ValueError:
            current = None
        if attempt and (not identity or current != identity):
            raise ValueError('drink cancel retry no longer shows the same bottle')
        identity = current
        image, captured_at = fresh()
        cancel = upstream_drink_actions(context, image).get('cancel')
        if cancel is not None and getattr(context.tasker, 'stopping', False) is not True:
            click((206,1188,1,1) if attempt and not observed_target_only else cancel, captured_at, 'cancel')
        if _wait_drink_panel(context, fresh, opened=False, timeout=3):
            return True
    raise ValueError('drink detail did not return after cancel')


def read_drink_detail(context, image, names, *, closing=False, current_use=False):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('drink detail resolution is uncalibrated')
    rows = _inventory_ocr(context, image, 'LiveStatusDrinkDescription')

    def unique(roi, predicate, *, white_title=False):
        matches = [r for r in rows if _inside({'box': tuple(r.box)}, roi)
                   and predicate(_normalize_status_text(r.text))]
        if len(matches) != 1 or not math.isfinite(matches[0].score):
            raise ValueError('drink detail text is missing or ambiguous')
        row = matches[0]
        if white_title and .85 <= row.score < .95:
            x,y,w,h = row.box
            crop = image[y:y+h,x:x+w]
            ink = crop.min(axis=2) >= 235
            yy,xx = np.where(ink)
            if not len(xx):
                raise ValueError('drink detail title has no visible white text')
            glyphs = 255-ink[yy.min():yy.max()+1,xx.min():xx.max()+1].astype(np.uint8)*255
            glyphs = cv2.copyMakeBorder(glyphs,6,6,6,6,cv2.BORDER_CONSTANT,value=255)
            reread = _inventory_ocr(context,cv2.cvtColor(glyphs,cv2.COLOR_GRAY2BGR),
                                    'LiveStatusDrinkTitle',only_rec=True)
            if (len(reread)!=1 or not math.isfinite(reread[0].score) or reread[0].score < .95
                    or _normalize_status_text(reread[0].text)!=_normalize_status_text(row.text)):
                raise ValueError('drink detail title reread is unresolved or contradictory')
            row = SimpleNamespace(text=row.text,box=row.box,score=reread[0].score)
        if row.score < .95:
            raise ValueError('drink detail text confidence is insufficient')
        return row

    # The bottom actions stay fixed, while longer descriptions grow upward.
    # Locate the exclusive heading first and bind the name to that heading.
    title = unique((30, 320, 600, 650), lambda text: text == 'Pドリンク詳細', white_title=True)
    use = unique((385, 1100, 290, 120), lambda text: text == '使う')
    cancel = unique((40, 1100, 300, 120), lambda text: text in {'キャンセル', 'やめる', '閉じる', '戻る'})
    if closing:
        return dict(cancel_box=tuple(cancel.box))
    name = unique((150, title.box[1] + 60, 490, 90), lambda text: text in names)
    x,y,w,h = name.box
    yy,xx = np.where(image[y:y+h,x:x+w].min(axis=2) < 120)
    if not len(xx) and not current_use:
        raise ValueError('drink name has no visible text')
    ink_box = ((x+int(xx.min()),y+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
               if len(xx) else None)
    # Text alone remains legible on disabled controls. Require the enabled
    # orange button surface around that independently recognized use label.
    ux,uy,uw,uh=use.box
    use_yy,use_xx=np.where(image[uy:uy+uh,ux:ux+uw].min(axis=2)>=235)
    use_ink_box = None
    if (len(use_xx) and use_xx.min()>0 and use_yy.min()>0
            and use_xx.max()<uw-1 and use_yy.max()<uh-1):
        # 1174's OCR padding alternated by three pixels in height. Compare
        # the actual complete white label, retaining its fresh OCR click box.
        use_ink_box=(ux+int(use_xx.min()),uy+int(use_yy.min()),
                     int(use_xx.max()-use_xx.min()+1),int(use_yy.max()-use_yy.min()+1))
    button_roi=(ux+uw//2-100,uy+uh//2-28,200,56)
    bx,by,bw,bh=button_roi
    hsv=cv2.cvtColor(image[by:by+bh,bx:bx+bw],cv2.COLOR_BGR2HSV)
    orange=(hsv[:,:,0]>=5)&(hsv[:,:,0]<=35)&(hsv[:,:,1]>=150)&(hsv[:,:,2]>=180)
    orange_fraction=float(orange.mean())
    return dict(name=_normalize_status_text(name.text), name_box=tuple(name.box),
                name_ink_box=ink_box,
                cancel_box=tuple(cancel.box), use_box=tuple(use.box), use_ink_box=use_ink_box,
                available=True if orange_fraction>=.85 else None,
                availability_evidence=dict(roi=button_roi,orange_fraction=orange_fraction),
                confidence=min(title.score, name.score, use.score, cancel.score))


def observe_drink_detail_transaction(context, *, capture_frames, checkpoint,
                                     frame_identifier, names, maximum_age_ms=1000, slot_index=None,
                                     restore_open_panel=False, _use_name=None, _expected_core=None,
                                     _on_use_progress=None, _expected_core_validator=None,
                                     _planned_upstream_use=False, _inspection_click=None,
                                     _inspection_controls=None, _inspection_single_frame=False):
    """Inspect and cancel, or submit a bound drink for separate settlement."""
    from utils.maa_input import verify_maa_input_endpoints

    if type(maximum_age_ms) not in (int, float) or not 0 < maximum_age_ms <= 1000:
        raise ValueError('drink detail freshness gate is invalid')
    if not names or len(set(names)) != len(names):
        raise ValueError('drink detail catalog names are incomplete or ambiguous')
    if slot_index is not None and (type(slot_index) is not int or not 0<=slot_index<4):
        raise ValueError('drink observation slot index is invalid')
    if (_inspection_single_frame or _inspection_click is not None or _inspection_controls is not None) and (
            not callable(_inspection_click) or slot_index is None or _inspection_controls is None
            or restore_open_panel or _use_name is not None):
        raise ValueError('scoped drink inspection requires authorized open and cancel for one slot')
    if _use_name is not None and (restore_open_panel or slot_index is None or _expected_core is None):
        raise ValueError('drink use requires an explicit slot and bound complete checkpoint')
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    current_use = _use_name is not None and upstream
    if _planned_upstream_use and not current_use:
        raise ValueError('upstream bottle selection requires a bound strong drink action')
    controller = context.tasker.controller
    event = dict(phase='drink_description_inspection', accepted=False, calibration_only=True,
                 input_click_count=0, inputs=[], drinks_consumed=0, strategy_state_bound=False)
    if upstream:
        event['detail_flow'] = 'miyu'
    started = time.perf_counter()

    def fresh():
        image = controller.post_screencap().wait().get()
        return image, frame_captured_at(controller, image)

    def current_controls():
        image, captured_at = fresh()
        state = checkpoint((image,), single_frame=True)
        if len(state['frame_ids']) != 1 or not state['frame_ids'][0]:
            raise ValueError('scoped drink inspection lacks one current frame ID')
        return state, drink_slot_targets(context, image), captured_at

    def click(box, captured_at, phase):
        if _inspection_click is not None:
            deadline = captured_at + maximum_age_ms / 1000
            if phase not in ('open', 'cancel') or not captured_at <= time.perf_counter() < deadline:
                raise ValueError('scoped drink inspection input is not fresh open/cancel navigation')
            endpoints = verify_maa_input_endpoints(controller.info,
                expected_client_size=getattr(context, 'gkh_hif_client_size', (720, 1280)))
            if time.perf_counter() >= deadline:
                raise ValueError('scoped drink inspection input expired during endpoint verification')
            attempt = dict(phase=phase, box=box, confirmed=False)
            event.setdefault('scoped_navigation_attempts', []).append(attempt)
            if _inspection_click(slot=slot_index, box=box, captured_at=captured_at,
                                 phase=phase, input_deadline=deadline) is not True:
                raise ValueError('authorized drink navigation was not confirmed')
            attempt['confirmed'] = True
            event['input_click_count'] += 1
            event['inputs'].append(dict(phase=phase, box=box, endpoints=endpoints))
            return
        endpoints = verify_maa_input_endpoints(controller.info, expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280)))
        if not upstream and (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
            raise ValueError('drink detail input frame expired')
        event['input_click_count'] += 1
        event['inputs'].append(dict(phase=phase, box=box, endpoints=endpoints))
        if phase == 'use' and _on_use_progress is not None:
            _on_use_progress('dispatch_pending', event)
            if not upstream and (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
                raise ValueError('drink detail input frame expired after recording')
        x,y,w,h = box
        if not controller.post_click(x+w//2,y+h//2).wait().succeeded:
            raise ValueError('Maa drink detail input failed')

    def checked_source():
        _, before, slots = capture_stable_drink_controls(context,capture_frames,checkpoint,event)
        if _inspection_click is not None and not drink_controls_consistent([_inspection_controls, *slots]):
            raise ValueError('drink controls differ from the HUD inspection evidence')
        if _use_name is not None:
            event['bound_before_core'] = {k:before[k] for k in ('families','fields','frame_ids')}
            matches = (_expected_core_validator(before) if _expected_core_validator is not None else
                all(json.dumps(before[k]) == json.dumps(_expected_core[k]) for k in ('families','fields')))
            if matches is not True:
                raise ValueError('Live changed before the bound drink transaction')
        event.update(control_observations=slots, before_frame_ids=before['frame_ids'])
        if not drink_controls_consistent(slots):
            raise ValueError('drink controls are unresolved')
        candidates = [index for index, box in enumerate(slots[0]) if box is not None]
        if not candidates:
            raise ValueError('no observed occupied drink control')
        slot = candidates[0] if slot_index is None else slot_index
        if slot not in candidates:
            raise ValueError('requested drink slot has no observed occupied control')
        event.update(slot=slot, before_frame_ids=before['frame_ids'])
        for attempt in range(3):
            image, captured_at = fresh()
            try:
                current = checkpoint((image,), single_frame=True)
                if any(current[k] != before[k] for k in ('families', 'fields')):
                    raise ValueError('Live changed before drink inspection')
                current_slots = drink_slot_targets(context, image)
                if not drink_controls_consistent([*slots, current_slots]):
                    missing = [i for i,(old,new) in enumerate(zip(slots[0],current_slots)) if old is not None and new is None]
                    if (attempt < 2 and len(missing) == 1
                            and drink_controls_consistent([*slots,tuple(
                                slots[0][i] if i in missing else box for i,box in enumerate(current_slots))])):
                        event.setdefault('preinput_control_reobservations', []).append(current_slots)
                        time.sleep(.08)
                        continue
                    raise ValueError('drink controls changed before input')
                break
            except ValueError as error:
                if attempt < 2 and (str(error) in ('inventory fresh recheck is not an unselected hand',
                                  'HIF selection geometry invalid') or fresh_identity_reobserve_allowed(error, before)):
                    # A single animated frame must not authorize input. Prove
                    # the full unselected hand again before a fresh recheck.
                    _, confirmed = capture_initial_live_checkpoint(capture_frames,checkpoint,event)
                    if any(confirmed[k] != before[k] for k in ('families', 'fields')):
                        raise ValueError('Live changed before drink inspection') from error
                    event.setdefault('preinput_hand_reobservations', []).append(confirmed['frame_ids'])
                    continue
                evidence=unresolved_live_resource_evidence(error)
                if not evidence:raise
                event.setdefault('preinput_resource_observations',[]).append(evidence)
                if attempt==2:raise
                time.sleep(.08)
        if any(current[k] != before[k] for k in ('families', 'fields')):
            raise ValueError('Live changed before drink inspection')
        event['preinput_controls'] = current_slots
        if not drink_controls_consistent([*slots, current_slots]):
            raise ValueError('drink controls changed before input')
        return before, slots, current_slots, slot, captured_at

    try:
        if not restore_open_panel and _retains_round_one_drinks(context):
            raise ValueError('remaining R1 bottles are reserved for the next battle')
        if restore_open_panel:
            event.update(recovery_only=True,phase='drink_description_recovery')
            if upstream:
                if not close_miyu_drink_panel(context, fresh=fresh, click=click,
                                              frame_identifier=frame_identifier, names=names):
                    raise ValueError('drink detail did not return after cancel')
            else:
                image,captured_at=fresh()
                current=read_drink_detail(context,image,names,closing=True)
                click(current['cancel_box'],captured_at,'cancel')
            for attempt in range(6):
                try:
                    frames=capture_frames()
                    after=checkpoint(frames)
                    break
                except ValueError:
                    if attempt==5:raise
                    time.sleep(.08)
            event.update(accepted=True,source_restored=True,
                         after_checkpoint={k:after[k] for k in ('families','fields','frame_ids')},
                         elapsed_ms=(time.perf_counter()-started)*1000)
            return [event]
        if _planned_upstream_use:
            box, captured_at = _planned_drink_target(context, slot_index, fresh, frame_identifier, event)
        elif _inspection_single_frame:
            before, current_slots, captured_at = current_controls()
            if not drink_controls_consistent([_inspection_controls, current_slots]):
                raise ValueError('drink controls differ from the HUD inspection evidence')
            slot, slots = slot_index, [current_slots]
            box = current_slots[slot]
            if box is None:
                raise ValueError('requested drink slot has no observed occupied control')
            event.update(slot=slot, before_frame_ids=before['frame_ids'], control_observations=slots,
                         preinput_controls=current_slots)
        else:
            before, slots, current_slots, slot, captured_at = checked_source()
            box = current_slots[slot]
        click(box, captured_at, 'open')
        failure = None
        try:
            if upstream:
                if current_use:
                    from .exam_hud import battle_dialog_header
                    from .exam_loop import exam_drink_open_probe
                    from .hif_upstream_ui import upstream_drink_actions

                    def retry_open(frame):
                        image, captured_at = frame
                        try:
                            current = checkpoint((image,), single_frame=True)
                            same_scene = (_expected_core_validator(current) if _expected_core_validator is not None else
                                all(json.dumps(current[k]) == json.dumps(_expected_core[k])
                                    for k in ('families', 'fields')))
                        except ValueError:
                            return
                        if same_scene is True:
                            box = drink_slot_targets(context, image)[event['slot']]
                            if box is not None:
                                click(box, captured_at, 'open_retry')

                    probe = exam_drink_open_probe(lambda frame: upstream_drink_actions(context, frame[0]),
                        lambda actions: set(actions) == {'use', 'cancel'}, retry_open,
                        dialog_visible=lambda frame: battle_dialog_header(frame[0]) is not None)
                    if not _wait_drink_panel(context, fresh, opened=True, timeout=8, open_probe=probe):
                        raise ValueError('drink detail did not open')
                elif not _wait_drink_panel(context, fresh, opened=True, timeout=4):
                    if _inspection_click is not None:
                        raise ValueError('scoped drink inspection did not confirm the detail opened')
                    if getattr(context.tasker, 'stopping', False) is True:
                        raise ValueError('drink task stopped')
                    image, captured_at = fresh()
                    box = drink_slot_targets(context, image)[event['slot']]
                    if box is None:
                        raise ValueError('drink bottle no longer visible for open retry')
                    click(box, captured_at, 'open_retry')
                    if not _wait_drink_panel(context, fresh, opened=True, timeout=4):
                        raise ValueError('drink detail did not open')
                event['observation'] = (_miyu_drink_detail(context, fresh, names, event, frame_identifier,
                                                         verify_open=True) if _inspection_single_frame
                    else _miyu_drink_detail(context, fresh, names, event, frame_identifier))
                # The opened panel has both miyu action labels. Actual consumption
                # is established separately from the game response.
                event['observation']['available'] = True
            else:
                for attempt in range(6):
                    frames = capture_frames()
                    try:
                        observed = [read_drink_detail(context, frame, names) for frame in frames]
                        if any(o['name'] != observed[0]['name'] or o.get('available') != observed[0].get('available')
                               or max(abs(a-b) for a,b in zip(o['name_ink_box'], observed[0]['name_ink_box'])) > 2
                               for o in observed[1:]):
                            raise ValueError('drink description differs across supplied frames')
                        break
                    except ValueError:
                        if attempt == 5:
                            raise
                        time.sleep(.08)
                event.update(observation=observed[len(observed)//2], page_frame_ids=[frame_identifier(f) for f in frames])
        except (ValueError, RuntimeError) as error:
            failure = str(error)
            event['read_failure'] = failure
        if (current_use and failure is None and (event['observation']['name'] != _use_name
                or event['observation'].get('available') is not True)):
            failure = event['read_failure'] = 'bound drink identity or availability changed'
        if _use_name is not None and failure is None:
            if event['observation']['name'] != _use_name or event['observation'].get('available') is not True:
                raise ValueError('bound drink identity or availability changed')
            image, captured_at = fresh()
            event['use_preinput_frame_id'] = frame_identifier(image)
            if current_use:
                from .hif_upstream_ui import upstream_drink_actions

                button = upstream_drink_actions(context, image).get('use')
                if button is None:
                    raise ValueError('drink use button is no longer visible')
                detail = dict(use_box=button)
                event['use_control_geometry'] = 'miyu_action_box'
            else:
                detail = read_drink_detail(context, image, names)
                event['use_control_observations'] = observed
                if detail['name'] != _use_name or detail.get('available') is not True:
                    raise ValueError('drink use button lacks fresh consensus')
                use_geometry = ('use_ink_box' if any('use_ink_box' in row
                    for row in (*observed,detail)) else 'use_box')
                event['use_control_geometry'] = use_geometry
                if (any(row.get(use_geometry) is None for row in (*observed,detail))
                        or any(max(abs(a-b) for a,b in zip(row[use_geometry], detail[use_geometry])) > 2
                               for row in observed)):
                    raise ValueError('drink use button lacks fresh consensus')
            event.update(use_preinput_observation=detail, calibration_only=False, strategy_state_bound=True,
                         execution_submit_count=1, input_may_have_been_dispatched=True)
            click(detail['use_box'], captured_at, 'use')
            # Submission is not consumption evidence. The caller must resolve
            # a full new observation, including this exact physical slot.
            event.update(use_submitted=True, elapsed_ms=(time.perf_counter()-started)*1000)
            if _on_use_progress is not None:
                _on_use_progress('submitted', event)
            if current_use and not _wait_drink_panel(context, fresh, opened=False, timeout=10):
                raise ValueError('drink detail did not return after use')
            return [event]
        if upstream:
            if not close_miyu_drink_panel(context, fresh=fresh, click=click,
                    frame_identifier=frame_identifier, names=names, observed_target_only=_inspection_click is not None):
                raise ValueError(failure or 'drink detail did not return after cancel')
        else:
            image, captured_at = fresh()
            cancel = read_drink_detail(context, image, names, closing=True)['cancel_box']
            click(cancel, captured_at, 'cancel')
        if current_use:
            raise ValueError(failure)
        for attempt in range(6):
            try:
                if _inspection_single_frame:
                    after, after_controls, _ = current_controls()
                    after_slots = [after_controls]
                else:
                    frames, after, after_slots = capture_stable_drink_controls(
                        context, capture_frames, checkpoint, event)
                break
            except ValueError:
                if attempt == 5:
                    raise
                time.sleep(.08)
        event['after_controls'] = after_slots
        if (any(after[k] != before[k] for k in ('families', 'fields'))
                or not drink_controls_consistent([*slots, current_slots, *after_slots])):
            raise ValueError('Live or drink controls changed during inspection')
        event.update(returned_unchanged=True, after_frame_ids=after['frame_ids'])
        if failure:
            raise ValueError(failure)
        event.update(accepted=True, elapsed_ms=(time.perf_counter()-started)*1000)
        return [event]
    except Exception as error:
        event.update(reason=str(error), elapsed_ms=(time.perf_counter()-started)*1000)
        raise StatusInspectionError(str(error), [event]) from error


def _guard_live_checkpoint(checkpoint, validate):
    """Preserve explicit observation callbacks while guarding every result."""
    def guarded(frames, **kwargs):
        return validate(checkpoint(frames, **kwargs))

    callbacks = getattr(checkpoint, '__dict__', {})
    complete = callbacks.get('capture_complete')
    if callable(complete):
        def capture_complete(capture_frames, event):
            frames, state = complete(capture_frames, event)
            return frames, validate(state)
        guarded.capture_complete = capture_complete
    single = callbacks.get('single_frame_reobserve_allowed')
    if callable(single):
        guarded.single_frame_reobserve_allowed = single
    return guarded


def observe_occupied_drink_slots(context, *, capture_frames, checkpoint,
                                 frame_identifier, metadata, maximum_age_ms=1000):
    """Read every observed occupied control; unconfirmed slots stay UNKNOWN."""
    event = dict(phase='occupied_drink_slots_inspection',accepted=False,items=[],transactions=[],
                 input_click_count=0,drinks_consumed=0,full_drink_state_ready=False,strategy_state_bound=False)
    started = time.perf_counter()
    try:
        frames,before,observations = capture_stable_drink_controls(context,capture_frames,checkpoint,event)
        event['control_observations'] = observations
        if not drink_controls_consistent(observations):
            raise ValueError('drink controls changed during initial observation')
        slots = [index for index,box in enumerate(observations[0]) if box is not None]
        if not slots:
            raise ValueError('there are no recognized occupied drink controls')
        event['unknown_slots'] = [index for index in range(4) if index not in slots]
        names = tuple(row['name'] for row in metadata)

        def validate_live(state):
            if any(state[k]!=before[k] for k in ('families','fields')):
                raise ValueError('Live changed during drink-slot inspection')
            return state
        same_live = _guard_live_checkpoint(checkpoint, validate_live)

        for slot in slots:
            if getattr(context, 'gkh_hif_upstream_card_ui', False) is not True and time.perf_counter()-started>45:
                raise ValueError('drink-slot inspection budget reached')
            try:
                transactions = observe_drink_detail_transaction(context,capture_frames=capture_frames,
                    checkpoint=same_live,frame_identifier=frame_identifier,names=names,
                    maximum_age_ms=maximum_age_ms,slot_index=slot)
            except StatusInspectionError as error:
                event['transactions'].extend(error.events)
                raise
            event['transactions'].extend(transactions)
            if (not transactions or transactions[-1].get('accepted') is not True
                    or transactions[-1].get('returned_unchanged') is not True):
                raise ValueError('drink-slot transaction did not verify its return')
            observed = transactions[-1]['observation']
            matches = [row for row in metadata if _normalize_status_text(row['name'])==observed['name']]
            if len(matches)!=1:
                raise ValueError('drink business identity is missing or ambiguous')
            event['items'].append(dict(slot=slot,drink_id=matches[0]['id'],name=observed['name'],
                                       available=observed.get('available'),
                                       confidence=observed['confidence'],frame_ids=transactions[-1]['page_frame_ids']))
        # The last visit already checked its return to the original Live.
        event.update(accepted=True,returned_unchanged=True,
                     after_frame_ids=event['transactions'][-1]['after_frame_ids'],
                     elapsed_ms=(time.perf_counter()-started)*1000)
        event['input_click_count'] = sum(e['input_click_count'] for e in event['transactions'])
        return [event]
    except Exception as error:
        event.update(reason=str(error),elapsed_ms=(time.perf_counter()-started)*1000,
                     input_click_count=sum(e['input_click_count'] for e in event['transactions']))
        raise StatusInspectionError(str(error),[event]) from error


def _maintained_drink_occupancy(initial, details, *, inspected_slot=None):
    """Retain positive occupancy through verified, cancel-only detail visits."""
    from .opening_observer import valid_opening_frame_ids

    physical=initial.get('slots',[])
    initial_ids = initial.get('frame_ids', ())
    if (initial.get('accepted') is not True or initial.get('occupancy_complete') is not True
            or [s.get('slot') for s in physical]!=list(range(4))
            or any(s.get('state') not in ('OCCUPIED','EMPTY') or not s.get('box') for s in physical)
            or not valid_opening_frame_ids(initial_ids) or len(initial_ids) != 1):
        raise ValueError('initial drink occupancy is not complete')
    occupied=[s['slot'] for s in physical if s['state']=='OCCUPIED']
    if inspected_slot is not None and inspected_slot not in occupied:
        raise ValueError('scoped drink inspection target was not occupied')
    inspected = occupied if inspected_slot is None else [inspected_slot]
    transactions=details.get('transactions',[])
    if (details.get('accepted') is not True or details.get('returned_unchanged') is not True
            or details.get('drinks_consumed')!=0
            or [t.get('slot') for t in transactions]!=inspected
            or not inspected or details.get('input_click_count')!=sum(t.get('input_click_count',0) for t in transactions)):
        raise ValueError('drink occupancy maintenance lacks complete return evidence')
    controls=[tuple(tuple(s['box']) if s['state']=='OCCUPIED' else None for s in physical)]
    for transaction in transactions:
        phases = [item.get('phase') for item in transaction.get('inputs', ())]
        before_ids = transaction.get('before_frame_ids')
        before_controls = transaction.get('control_observations', ())
        after_ids = transaction.get('after_frame_ids', ())
        valid_visit = phases == ['open','cancel']
        if transaction.get('detail_flow') == 'miyu':
            valid_visit = any(phases == ['open',*opening_retry,*(['cancel'] * count)]
                for opening_retry in ([], ['open_retry']) for count in (1, 2, 3))
        if (transaction.get('accepted') is not True or transaction.get('returned_unchanged') is not True
                or transaction.get('drinks_consumed')!=0 or transaction.get('input_click_count')!=len(phases)
                or not valid_visit
                or transaction.get('use_submitted',False) is not False
                or transaction.get('execution_submit_count',0)!=0
                or len(before_controls) != 1
                or not valid_opening_frame_ids(before_ids) or len(before_ids) != 1
                or not valid_opening_frame_ids(after_ids) or len(after_ids) != 1
                or len(transaction.get('after_controls', ())) != len(after_ids)
                or len(transaction.get('preinput_controls',()))!=4):
            raise ValueError('drink occupancy maintenance requires verified cancel-only visits')
        controls.extend(transaction['control_observations'])
        controls.append(transaction['preinput_controls'])
        controls.extend(transaction['after_controls'])
    controls=[tuple(tuple(box) if box is not None else None for box in row) for row in controls]
    if not drink_controls_consistent(controls):
        raise ValueError('drink occupancy changed during detail visits')
    if details.get('after_frame_ids')!=transactions[-1]['after_frame_ids']:
        raise ValueError('drink occupancy maintenance does not end at the last verified return')
    return dict(slots=physical,frame_ids=details['after_frame_ids'])


def _retains_round_one_drinks(context):
    from .high_score.hif_round1_card_only import LIVE_RUN_ID

    run = getattr(context, 'gkh_hif_live_context', None)
    handoff = getattr(context, 'gkh_hif_battle_handoff', None)
    return getattr(run, 'run_id', None) == LIVE_RUN_ID and getattr(handoff, 'round_number', None) == 1


def _hud_drink_inventory(context, initial, metadata, plan, icons):
    occupied = [row['slot'] for row in initial['slots'] if row['state'] == 'OCCUPIED']
    catalog = {row['id']: row for row in metadata if row.get('plan') in (plan, 'free')}
    prefix = getattr(context, 'gkh_hif_opening_drink_prefix', None) or {}
    retained = {item['slot']:item for item in prefix.get('items', ())} if prefix.get('occupied') == occupied else {}
    # Only the untouched opening observer supplies this invocation's settled handoff.
    handoff = getattr(context, 'gkh_hif_opening_drink_handoff', None)
    route = {item.slot:item.drink_id for item in handoff.ledger.drinks} if handoff is not None else {}
    if sorted(route) != occupied:
        route = {}
    slots = []
    for physical in initial['slots']:
        slot = physical['slot']
        if physical['state'] == 'EMPTY':
            slots.append(dict(slot=slot, state='EMPTY', drink_id=None, available=False))
            continue
        if physical['state'] == 'UNKNOWN':
            slots.append(dict(slot=slot, state='UNKNOWN', drink_id=None, available=None,
                source='hud_bottle', frame_ids=list(initial['frame_ids']),
                hud_observation=icons.get(slot) or dict(reason='conflicting_controls' if physical.get('conflicting_controls')
                                                        else 'unresolved_occupancy')))
            continue
        icon = icons.get(slot) or dict(reason='unknown_bottle' if icons else 'hud_recognizer_unavailable')
        old = retained.get(slot)
        drink_id = icon.get('drink_id')
        known_ids = {value for value in (old and old['drink_id'], route.get(slot), drink_id) if value is not None}
        resolved = next(iter(known_ids)) if len(known_ids) == 1 else None
        if (resolved in catalog or resolved == EVENT_GREEN_JUICE_ID) and old:
            slots.append(dict(old, state='OCCUPIED', source='retained_opening_detail',
                              source_result=prefix['source_result'], hud_observation=icon))
        elif resolved in catalog or resolved == EVENT_GREEN_JUICE_ID:
            name = EVENT_GREEN_JUICE_NAME if resolved == EVENT_GREEN_JUICE_ID else catalog[resolved]['name']
            row = dict(slot=slot, state='OCCUPIED', drink_id=resolved, name=name,
                available=True, availability_basis='occupied_hud_control', frame_ids=list(initial['frame_ids']),
                hud_observation=icon)
            if slot in route:
                row.update(source='settled_opening_handoff', handoff_id=handoff.handoff_id,
                    ledger_revision=handoff.ledger.revision, source_frame_ids=list(handoff.source_frame_ids))
            else:
                row.update(source='hud_bottle', confidence=icon.get('confidence'))
            slots.append(row)
        else:
            slots.append(dict(slot=slot, state='OCCUPIED', drink_id=None, available=None,
                source='hud_bottle', frame_ids=list(initial['frame_ids']), hud_observation=icon))
            if len(known_ids) > 1 or resolved is not None:
                slots[-1]['identity_conflict'] = True
        if resolved == EVENT_GREEN_JUICE_ID:
            slots[-1].pop('availability_basis', None)
            slots[-1].update(available=None, strategy_supported=False)
        if resolved is not None and resolved == drink_id and 'available' in icon:
            slots[-1].update(available=icon['available'], availability_basis='hud_icon_saturation')
    return slots


def observe_complete_drink_inventory(context, *, capture_frames, checkpoint,
                                     frame_identifier, metadata, template, plan, run_id, revision,
                                     maximum_age_ms=1000, hud_recognizer=None,
                                     hud_detail_slot=None, hud_detail_navigate=None,
                                     single_frame_hud=False, default_icon_dir=None):
    """Read the HUD. A trusted caller may supply audited navigation for one unknown slot.

    The supplied capability must verify the run, revision, slot, phase and input
    deadline at the audited Maa dispatch, returning True only on success.
    """
    event=dict(phase='complete_drink_inventory',accepted=False,events=[],input_click_count=0,
               drinks_consumed=0,full_drink_state_ready=False,strategy_state_bound=False)
    started=time.perf_counter()
    baseline=None

    def validate_live(state):
        nonlocal baseline
        values={k:state[k] for k in ('families','fields')}
        if baseline is None:baseline=values
        elif values!=baseline:raise ValueError('Live changed during complete drink inventory')
        return state
    guarded = _guard_live_checkpoint(checkpoint, validate_live)

    def collect(function, **kwargs):
        try:rows=function(context,capture_frames=capture_frames,checkpoint=guarded,**kwargs)
        except StatusInspectionError as error:
            event['events'].extend(error.events)
            raise
        event['events'].extend(rows)
        if not rows or rows[-1].get('accepted') is not True:
            raise ValueError('drink inventory component did not complete')
        return rows[-1]

    try:
        if (plan not in ('sense','logic','anomaly') or type(run_id) is not str or not run_id.strip()
                or type(revision) is not int or revision<0):
            raise ValueError('drink inventory run identity is invalid')
        if type(maximum_age_ms) not in (int,float) or not 0<maximum_age_ms<=1000:
            raise ValueError('drink inventory freshness gate is invalid')
        reserved = _retains_round_one_drinks(context)
        hud = reserved or single_frame_hud or hud_recognizer is not None or getattr(context, 'gkh_hif_upstream_card_ui', False) is True
        if (hud_detail_slot is not None or hud_detail_navigate is not None) and (
                type(hud_detail_slot) is not int or not 0 <= hud_detail_slot < 4
                or not callable(hud_detail_navigate) or not hud or reserved):
            raise ValueError('HUD drink detail requires authorized navigation for one slot')
        if context.tasker.controller.set_screenshot_target_long_side(1280) is not True:
            raise ValueError('Maa HIF screenshot long-side profile could not be applied')
        event['capture_profile']=dict(target_long_side=1280,applied=True)
        identity_metadata = (*metadata, dict(id=EVENT_GREEN_JUICE_ID, name=EVENT_GREEN_JUICE_NAME, plan='free'))
        if hud and hud_recognizer is None:
            hud_recognizer = getattr(context, 'gkh_hif_drink_icon_recognizer', None)
        default_reader = None
        if hud and hud_recognizer is None and default_icon_dir is not None:
            from .drink_icons import load_default_drink_icons

            try:
                default_reader = load_default_drink_icons(default_icon_dir, tuple(row['id'] for row in identity_metadata))
            except (OSError, ValueError, cv2.error) as error:
                event['default_icon_library_error'] = str(error)
        direct_hud = single_frame_hud and default_reader is not None
        client_size = getattr(context, 'gkh_hif_client_size', (720,1280))
        frames = []
        if direct_hud:
            frames = capture_frames()
            if len(frames) != 1:
                raise ValueError('HUD drink inventory requires one current frame')
            current = guarded(frames, single_frame=True)
            if len(current['frame_ids']) != 1 or not current['frame_ids'][0]:
                raise ValueError('HUD drink inventory lacks one current frame ID')
            icons = dict(enumerate(default_reader.read_slots(frames[0], client_size=client_size)))
            initial = dict(frame_ids=current['frame_ids'], slots=[dict(slot=i, state=icon['state'])
                                                                   for i, icon in icons.items()])
        else:
            initial = collect(observe_drink_slot_occupancy, template=template, single_frame=single_frame_hud,
                              accepted_frames=frames if hud else None)
        occupied=[s['slot'] for s in initial['slots'] if s['state']=='OCCUPIED']
        final=initial
        if hud:
            if not direct_hud:
                recognizer = hud_recognizer if hud_recognizer is not None else default_reader
                icons = recognizer(frames, occupied, client_size=client_size) if occupied and recognizer is not None else {}
            slots = _hud_drink_inventory(context, initial, metadata, plan, icons)
            event.update(slots=slots, frame_ids=list(initial['frame_ids']), run_id=run_id, revision=revision,
                unknown_slots=[row['slot'] for row in slots if row['state']!='EMPTY' and row['drink_id'] is None],
                returned_unchanged=True, observation_source='hud_bottles', elapsed_ms=(time.perf_counter()-started)*1000)
            if (direct_hud and hud_detail_slot in event['unknown_slots']
                    and icons[hud_detail_slot].get('reason') == 'weak_icon'):
                # The separate input transaction still needs actual click targets.
                initial, = observe_drink_slot_occupancy(context, capture_frames=lambda: frames,
                    checkpoint=lambda *_args, **_kwargs: current, template=template, single_frame=True)
                event['events'].append(initial)
            if (hud_detail_slot in event['unknown_slots'] and initial.get('occupancy_complete')
                    and initial['slots'][hud_detail_slot]['state'] == 'OCCUPIED'
                    and not any(row.get('identity_conflict') for row in slots)):
                event['returned_unchanged'] = False
                controls = tuple(tuple(row['box']) if row['state'] == 'OCCUPIED' else None for row in initial['slots'])
                navigate = partial(hud_detail_navigate, run_id=run_id, revision=revision)
                transaction = collect(observe_drink_detail_transaction, frame_identifier=frame_identifier,
                    names=tuple(_normalize_status_text(row['name']) for row in identity_metadata), maximum_age_ms=maximum_age_ms,
                    slot_index=hud_detail_slot, _inspection_click=navigate, _inspection_controls=controls,
                    _inspection_single_frame=single_frame_hud)
                final = _maintained_drink_occupancy(initial, dict(transaction, transactions=[transaction]),
                    inspected_slot=hud_detail_slot)
                event.update(returned_unchanged=True, frame_ids=final['frame_ids'])
                detail = transaction['observation']
                matches = [row for row in identity_metadata if _normalize_status_text(row['name']) == detail['name']]
                confidence = detail.get('confidence')
                if (len(matches) != 1 or matches[0]['plan'] not in (plan, 'free') or detail['available'] is not True
                        or sum(row['id'] == matches[0]['id'] for row in identity_metadata) != 1
                        or type(confidence) not in (int, float) or not .95 <= confidence <= 1
                        or not transaction.get('page_frame_ids')):
                    raise ValueError('HUD drink detail has unresolved catalog identity or availability')
                slots[hud_detail_slot].update(state='OCCUPIED', drink_id=matches[0]['id'], name=detail['name'], available=True,
                    source='scoped_opening_detail', confidence=confidence, frame_ids=transaction['page_frame_ids'])
                if matches[0]['id'] == EVENT_GREEN_JUICE_ID:
                    slots[hud_detail_slot].update(available=None, strategy_supported=False)
                event['unknown_slots'].remove(hud_detail_slot)
                event.update(observation_source='hud_bottles_and_scoped_detail',
                    input_click_count=transaction['input_click_count'], elapsed_ms=(time.perf_counter()-started)*1000)
            unsupported = [row['slot'] for row in slots if row.get('strategy_supported') is False]
            if unsupported:
                event['unsupported_slots'] = unsupported
                raise ValueError('HUD drink identity is recognized but unsupported by the strategy catalog')
            if event['unknown_slots']:
                raise ValueError('occupied HUD drink identity is unknown; scope or evidence does not permit further detail')
            event.update(accepted=True, full_drink_state_ready=True,
                drinks=tuple(row['drink_id'] for row in slots if row['state']=='OCCUPIED'))
            return [event]
        if occupied:
            details=collect(observe_occupied_drink_slots,frame_identifier=frame_identifier,metadata=metadata,
                            maximum_age_ms=maximum_age_ms)
            if [row['slot'] for row in details['items']]!=occupied:
                raise ValueError('drink identities do not cover exactly the occupied slots')
            # Every detail visit rechecks the guarded core and occupied
            # controls before/after open+cancel. Nothing can consume or move
            # a drink here; preserve the initial positive empty-slot proof.
            final=_maintained_drink_occupancy(initial,details)
            event['occupancy_maintenance']=dict(basis='verified_cancel_only_details',
                initial_frame_ids=initial['frame_ids'],after_frame_ids=final['frame_ids'],
                inspected_slots=occupied)
        else:
            # No occupied slot means no detail navigation or input occurred.
            # The initial complete three-frame empty proof is already final.
            details={'items':[]}
        if [s['state'] for s in final['slots']]!=[s['state'] for s in initial['slots']]:
            raise ValueError('drink occupancy changed during inventory')
        slots=[]
        for physical in final['slots']:
            if physical['state']=='EMPTY':
                slots.append(dict(slot=physical['slot'],state='EMPTY',drink_id=None,available=False))
                continue
            detail=next(row for row in details['items'] if row['slot']==physical['slot'])
            definitions=[row for row in metadata if row['id']==detail['drink_id']]
            if len(definitions)!=1 or definitions[0]['plan'] not in ('free',plan):
                raise ValueError('drink catalog identity or plan does not match')
            if detail['available'] is not True:
                raise ValueError('drink use availability is unresolved')
            slots.append(dict(detail,state='OCCUPIED'))
        event.update(accepted=True,full_drink_state_ready=True,run_id=run_id,revision=revision,
                     slots=slots,drinks=tuple(s['drink_id'] for s in slots if s['state']=='OCCUPIED'),
                     frame_ids=final['frame_ids'],returned_unchanged=True,
                     elapsed_ms=(time.perf_counter()-started)*1000)
        event['input_click_count']=sum(e.get('input_click_count',0) for e in event['events'])
        return [event]
    except Exception as error:
        event.update(reason=str(error),elapsed_ms=(time.perf_counter()-started)*1000,
                     input_click_count=sum(e.get('input_click_count',0) for e in event['events']))
        raise StatusInspectionError(str(error),[event]) from error
