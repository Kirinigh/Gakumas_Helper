"""Rebuild HIF context through the current notebook, using ordinary Maa UI."""

import re
import time
import uuid
import unicodedata
from dataclasses import asdict

import cv2

from .hif_visual import locate_hif_opening_self_rank
from .hif_session import (
    NOTEBOOK_IDENTITY_FIELDS,
    bind_hif_round_context,
    bind_recovered_hif_run,
    notebook_identity_fields,
    observe_hif_notebook_identity,
)
from .hif_capture_adapter import captured_at as frame_captured_at


class HifRecoveryError(ValueError):
    def __init__(self, reason, event):
        super().__init__(reason)
        self.event = event


class _LiveNotReady(ValueError):
    pass


def recovery_menu_target(image):
    """The three complete white bars inside the bottom-right Live control."""
    from route.footer import cultivation_menu_target

    if getattr(image, 'shape', ()) != (1280,720,3):
        raise ValueError('HIF recovery image size is uncalibrated')
    mask = (image[1160:1240,615:698].min(axis=2) > 235).astype('uint8')
    # Keep the complete glyph and a margin; the outer button is translucent
    # and exposes animated white stage lights outside the three bars.
    mask[:20] = 0
    mask[60:] = 0
    mask[:, :15] = 0
    mask[:, 65:] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    bars = []
    template_required = False
    for label, row in enumerate(stats[1:], 1):
        x,y,w,h,area = map(int,row)
        if area < 20:
            continue
        if 31 <= w <= 37 and 2 <= h <= 6 and area < w*h*.8:
            # A translucent-button highlight can attach one sparse row above
            # or below an intact bar (1621). Preserve only a complete, solid
            # horizontal core; no gap or longer connected tail is repaired.
            component = labels[y:y+h,x:x+w] == label
            complete = [i for i in range(h) if component[i].all()]
            if (2 <= len(complete) <= 6
                    and complete == list(range(complete[0],complete[-1]+1))
                    and complete[0] <= 1 and h-complete[-1]-1 <= 1
                    and all(component[i].sum() <= w*.2 for i in range(h) if i not in complete)):
                y,h = y+complete[0],len(complete)
                area = w*h
            elif len(complete) == 1:
                # In 0028 one endpoint of the adjacent solid row is antialiased.
                # Require the same uniquely matched cultivation control before
                # using this bar. An internal gap or detached tail is not a row.
                adjacent = [i for i in (complete[0]-1, complete[0]+1) if 0 <= i < h]
                core = next((i for i in adjacent
                             if component[i, 1:].all() or component[i, :-1].all()), None)
                if (core is not None and min(core, complete[0]) <= 1
                        and h-1-max(core, complete[0]) <= 1):
                    template_required = True
                    bars.append((x,y,w,h))
                    continue
        if area < w*h*.8:
            raise ValueError('HIF recovery menu glyph is not unique')
        bars.append((x,y,w,h))
    bars.sort(key=lambda r:r[1])
    if (len(bars) != 3 or any(not (18 <= x <= 24 and 31 <= w <= 37 and 2 <= h <= 6)
            for x,y,w,h in bars)
            or any(abs(row[1]-target)>2 for row,target in zip(bars,(25,38,51),strict=True))):
        raise ValueError('HIF recovery menu glyph is not unique')
    x,y,w,h = bars[1]
    if template_required:
        try:
            tx,ty,tw,th = cultivation_menu_target(image)
        except ValueError as error:
            raise ValueError('HIF recovery menu glyph is not unique') from error
        if (tx+tw//2, ty+th//2) != (615+x+w//2, 1160+y+h//2):
            raise ValueError('HIF recovery menu glyph is not unique')
    return (615+x, 1160+y, w, h)


def _notebook_fraction_separator(image):
    """Prove the complete slash between the fixed current/max number fields."""
    x, y, w, h = 373, 55, 20, 32
    mask = (image[y:y+h, x:x+w].min(axis=2) > 230).astype('uint8')
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count != 2:
        raise ValueError('HIF notebook fraction separator is not one complete stroke')
    left, top, width, height, area = map(int, stats[1])
    if not (2 <= left and left+width <= w-2 and 10 <= width <= 16
            and 2 <= top and top+height <= h-2 and 18 <= height <= 24 and 40 <= area <= 90):
        raise ValueError('HIF notebook fraction separator is incomplete')
    centers = []
    for row in mask[top:top+height]:
        pixels = row.nonzero()[0]
        if not 2 <= len(pixels) <= 4 or int(pixels[-1])-int(pixels[0])+1 != len(pixels):
            raise ValueError('HIF notebook fraction separator is not a continuous slash')
        centers.append((int(pixels[0])+int(pixels[-1]))/2)
    if (not 9 <= centers[0]-centers[-1] <= 13
            or any(not 0 <= before-after <= 1 for before, after in zip(centers, centers[1:]))):
        raise ValueError('HIF notebook fraction separator is not diagonal')
    return dict(source='complete_diagonal_separator', box=[x+left,y+top,width,height],
                roi=[x,y,w,h], white_threshold=230, row_centers=centers)


def _field(context, image, name, roi, pattern, *, frame_id=None, epoch=None):
    x,y,w,h = roi
    node = 'HifRecovery'+name
    detect = name.lower() in ('stage', 'title')
    config = {'recognition':'OCR','only_rec':not detect,'expected':'.+'}
    if detect:
        config['roi'] = list(roi)
    evidence = None
    crop_reader = getattr(context, 'recovery_crop_recognition', None)
    if name in ('Notebook', 'Help', 'song', 'idol', 'focus') and frame_id is not None and epoch is not None and callable(crop_reader):
        hit, evidence = crop_reader(image, name, config, frame_id=frame_id, epoch=epoch, roi=roi,
                                   capture_controller=context.tasker.controller)
    else:
        hit = context.run_recognition(node,image if detect else image[y:y+h,x:x+w].copy(),
                                      pipeline_override={node:config})
    rows = list(hit.filtered_results) if hit and hit.hit else []
    if name.lower() == 'stage' and len(rows) == 1 and 0 <= rows[0].score < .95:
        # 1799's tight detector crop turns a period into a comma. Independently
        # read the entire located label with background margin, never replace
        # punctuation or promote the detector's low-confidence text.
        box = rows[0].box
        if (len(box) == 4 and all(type(value) is int for value in box)
                and min(box[2:]) > 0 and x <= box[0]-4 and y <= box[1]-4
                and box[0]+box[2]+4 <= x+w and box[1]+box[3]+4 <= y+h):
            complete = (box[0]-4, box[1]-4, box[2]+8, box[3]+8)
            result = _field(context, image, 'StageComplete', complete, pattern)
            return dict(result, source='independent_complete_stage_label',
                        source_text=rows[0].text, source_confidence=rows[0].score,
                        detected_box=list(box))
    if detect:
        # Locate the complete label inside its bounded header, excluding the
        # adjacent campaign/round badge. Never repair clipped OCR characters.
        rows = [row for row in rows if re.fullmatch(pattern,
            ''.join(unicodedata.normalize('NFKC',row.text).split()))]
    if len(rows) != 1:
        raise ValueError('HIF recovery field is missing: '+name)
    row = rows[0]
    text = ''.join(unicodedata.normalize('NFKC',row.text).split())
    if (name == 'StageComplete' and not re.fullmatch(pattern, text)
            and re.fullmatch(r'H[.,][I1][.,]F[.]?本戦', text)):
        # On small clients the white Latin strokes bleed into the purple badge.
        # Use the complete detected glyph box; preserve the original OCR text
        # unless a new reading recognizes the full label at normal confidence.
        glyphs = image[y+4:y+h-4, x+4:x+w-4].min(axis=2)
        ink = cv2.addWeighted(glyphs, -2.4, glyphs, 0, 591)
        ink = cv2.resize(ink, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
        ink = cv2.copyMakeBorder(ink, 6, 6, 7, 7, cv2.BORDER_CONSTANT, value=255)
        hit = context.run_recognition(node, cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR), pipeline_override={node:config})
        rows = list(hit.filtered_results) if hit and hit.hit else []
        if len(rows) == 1:
            row = rows[0]
            text = ''.join(unicodedata.normalize('NFKC',row.text).split())
    if (name == 'stamina' and tuple(roi) == (324,45,110,58)
            and re.fullmatch(r'\d{3,5}', text)):
        # Native 983/984 reads the visible slash as "1". Its independent
        # complete diagonal stroke plus two high-confidence numeral fields
        # establishes the fraction; no character replacement or old resource
        # value supplies any evidence. Tight vertical bounds exclude the bar.
        separator = _notebook_fraction_separator(image)
        current = _field(context, image, 'NotebookCurrentTight', (326,53,48,36), r'\d{1,2}')
        maximum = _field(context, image, 'NotebookMaximum', (391,y,38,h), r'\d{1,2}')
        if current['text']+'1'+maximum['text'] != text:
            raise ValueError('HIF notebook fraction segments contradict the original row')
        return dict(text=current['text']+'/'+maximum['text'],
            score=min(current['score'],maximum['score']), box=list(roi),
            source_text=text, source_confidence=row.score,
            segments=[current,maximum], separator=separator)
    if name == 'stamina' and 0 <= row.score < .95 and re.fullmatch(pattern, text):
        # Notebook current/max numerals have different font sizes. Read both
        # fixed fields independently; their exact joined value must agree
        # with the visible slash-delimited row, never a predicted resource.
        current = _field(context, image, 'NotebookCurrent', (326,y,48,h), r'\d{1,2}')
        maximum = _field(context, image, 'NotebookMaximum', (391,y,38,h), r'\d{1,2}')
        if current['text']+'/'+maximum['text'] == text:
            return dict(text=text, score=min(current['score'],maximum['score']),box=list(roi),
                        source_confidence=row.score,segments=[current,maximum])
    if not .95 <= row.score <= 1 or not re.fullmatch(pattern,text):
        error = ValueError('HIF recovery field is unresolved: '+name)
        error.evidence = dict(field=name,text=text,confidence=row.score,roi=roi)
        raise error
    box = list(row.box) if detect else list(roi)
    if (len(box) != 4 or min(box[2:]) <= 0 or not
            (x <= box[0] and y <= box[1] and box[0]+box[2] <= x+w and box[1]+box[3] <= y+h)):
        raise ValueError('HIF recovery field is outside its region: '+name)
    result = dict(text=text,score=row.score,box=box)
    if evidence is not None:
        result['recognition_evidence'] = evidence
    return result


def _close_target(context, image):
    node = 'HifRecoveryNotebookClose'
    hit = context.run_recognition(node,image,pipeline_override={node:{
        'recognition':'TemplateMatch','template':['close_icon.png'],
        'roi':[300,1140,120,110],'threshold':.8}})
    rows = list(hit.filtered_results) if hit and hit.hit else []
    if len(rows) != 1 or not .8 <= rows[0].score <= 1:
        raise ValueError('HIF notebook close control is not unique')
    return tuple(rows[0].box)


def read_recovery_page(context, image, *, round_number=1, frame_id=None, epoch=None, require_live_menu=True):
    rank = locate_hif_opening_self_rank(image,context=context)
    live_failure = dict(rank=rank, shape=list(getattr(image, 'shape', ())))
    if rank['accepted']:
        try:
            target = recovery_menu_target(image) if require_live_menu else None
            stamina = _field(context,image,'LiveStamina',(573,212,49,32),r'\d{1,2}')
            turns = _field(context,image,'LiveTurns',(40,68,43,42),r'\d{1,2}')
            return dict(phase='live',target=target,stamina=int(stamina['text']),turns=int(turns['text']))
        except ValueError as error:
            live_failure.update(reason=str(error), evidence=getattr(error, 'evidence', None))
    try:
        _field(context,image,'Schedule',(58,1083,176,53),r'スケジュール')
        try:
            formation = _field(context,image,'Formation',(538,1083,80,52),r'編成')
            notebook_layout = 'legacy'
        except ValueError:
            formation = _field(context,image,'FormationEvents',(478,1083,202,52),r'編成・イベント')
            notebook_layout = 'campaign_banner'
        _field(context,image,'Round',*notebook_identity_fields(round_number)['round'])
    except ValueError:
        try:
            notebook = _field(context,image,'Notebook',(243,1004,74,35),r'P手帳',frame_id=frame_id,epoch=epoch)
        except ValueError as error:
            error.evidence = dict(live=live_failure, menu=getattr(error, 'evidence', None))
            raise
        help_field = _field(context,image,'Help',(407,1004,75,35),r'ヘルプ',frame_id=frame_id,epoch=epoch)
        result = dict(phase='menu',target=tuple(notebook['box']))
        if 'recognition_evidence' in notebook or 'recognition_evidence' in help_field:
            result['rows'] = [notebook, help_field]
        return result
    try:
        _field(context,image,'Title',*NOTEBOOK_IDENTITY_FIELDS['title'])
    except ValueError:
        return dict(phase='notebook',target=tuple(formation['box']))
    rows = [_field(context,image,name,*spec)
            for name,spec in notebook_identity_fields(round_number, layout=notebook_layout).items()]
    identity = observe_hif_notebook_identity(rows, round_number=round_number)
    return dict(phase='identity',target=_close_target(context,image),identity=identity,rows=rows,
                notebook_layout=notebook_layout)


def recover_hif_live_context(context, *, revision=1, round_number=1, expected_live=None, entry_identity=None,
        completed_phases=()):
    """Bind a fresh entry from its route and current Live, or observe the notebook."""
    from utils.maa_input import verify_maa_input_endpoints
    from card_selection.model import frame_identifier

    started = time.perf_counter()
    event = dict(phase='hif_context_recovery',accepted=False,input_click_count=0,
                 inputs=[],observations=[],returned_to_live=False)
    for name in ('gkh_hif_live_context','gkh_hif_opening_state','gkh_hif_committed_state',
                 'gkh_hif_committed_event','gkh_hif_round_context'):
        setattr(context,name,None)
    controller = context.tasker.controller
    size = getattr(context,'gkh_hif_client_size',(720,1280))
    binding = None
    initial_live = None
    identities = None
    visits = set(completed_phases)
    destination = None
    poll = getattr(context, 'gkh_hif_poll_battle_progress', None)

    def read_page(image):
        page = (read_recovery_page(context, image) if round_number == 1 else
                read_recovery_page(context, image, round_number=round_number))
        if page['phase'] == 'live':
            if expected_live is not None and any(page.get(k) != v for k, v in expected_live.items()):
                raise ValueError('HIF opening recovery Live differs from its original unstarted source')
            # The menu glyph already exists during stage-entry effects. The
            # same established playable node used by normal battle input must
            # be visible before a menu click or a completed return to Live.
            detail = context.run_recognition('ProduceRecognitionSkipRound', image)
            if detail is None or getattr(detail, 'hit', None) is not True:
                raise _LiveNotReady('HIF opening effects have not returned playable Live')
        return page

    def endpoints():
        result = verify_maa_input_endpoints(controller.info,expected_client_size=size)
        if binding is not None and result['game']['pid'] != binding:
            raise ValueError('game process changed during HIF recovery')
        return result

    try:
        if expected_live is not None and (not isinstance(expected_live, dict)
                or set(expected_live) != {'stamina', 'turns'}
                or any(type(v) is not int or v < 0 for v in expected_live.values())):
            raise ValueError('HIF opening recovery source fields are incomplete')
        notebook_identity_fields(round_number)
        binding = endpoints()['game']['pid']
        for _ in range(5):
            if destination is not None:
                transition_started = time.perf_counter()
                first_destination_frame = destination_frame = None
                polling = dict(expected=list(destination),phase_counts={},confirmed=False)
                event.setdefault('destination_observations',[]).append(polling)
                try:
                    while True:
                        if poll is None and (time.perf_counter()-transition_started > 8 or time.perf_counter()-started > 60):
                            raise ValueError('HIF recovery destination page was not confirmed')
                        job = controller.post_screencap().wait()
                        if not job.succeeded:
                            raise ValueError('HIF recovery transition capture failed')
                        destination_frame = job.get()
                        if poll is not None and poll(destination_frame):
                            continue
                        if first_destination_frame is None:
                            first_destination_frame = destination_frame
                        observation = dict(frame_id=frame_identifier(destination_frame),
                            elapsed_ms=(time.perf_counter()-transition_started)*1000)
                        try:
                            arrived = read_page(destination_frame)
                            observation['result'] = arrived
                            counts = polling['phase_counts']
                            counts[arrived['phase']] = counts.get(arrived['phase'],0)+1
                            polling['confirmed'] = arrived['phase'] in destination
                        except ValueError as error:
                            rejection = dict(reason=str(error),evidence=getattr(error,'evidence',None))
                            observation.update(rejection)
                            rejections=event.setdefault('destination_rejections',[])
                            rejections.append(rejection)
                            del rejections[:-3]
                        polling.setdefault('first',observation)
                        polling['last'] = observation
                        if polling['confirmed']:
                            break
                        time.sleep(.1)
                except (ValueError,RuntimeError):
                    sink = getattr(context, 'gkh_hif_recovery_failure_sink', None)
                    if callable(sink) and first_destination_frame is not None:
                        sink(first_destination_frame)
                        if destination_frame is not first_destination_frame:
                            sink(destination_frame)
                    raise
                event.setdefault('transitions',[]).append(dict(destination=arrived['phase'],
                    elapsed_ms=(time.perf_counter()-transition_started)*1000))
                # The menu labels appear before its entrance animation enables
                # input. Take the actionable observation after that animation.
                if arrived['phase'] == 'menu':
                    time.sleep(.5)
            content = lambda r: {k:v for k,v in r.items() if k not in ('frame_id','rows','target')}
            readiness_started = None
            retried = False
            while True:
                if poll is None and time.perf_counter()-started > 60:
                    raise ValueError('HIF recovery time budget exceeded')
                job = controller.post_screencap().wait()
                if not job.succeeded:
                    raise ValueError('HIF recovery capture failed')
                frame = job.get()
                captured_at = frame_captured_at(controller, frame)
                try:
                    row = read_page(frame)
                    break
                except _LiveNotReady as error:
                    now = time.perf_counter()
                    readiness_started = now if readiness_started is None else readiness_started
                    wait = event.setdefault('readiness_observations', dict(count=0))
                    observation = dict(frame_id=frame_identifier(frame), reason=str(error),
                        elapsed_ms=(now-readiness_started)*1000)
                    wait['count'] += 1
                    wait.setdefault('first', observation)
                    wait['last'] = observation
                    if poll is not None:
                        poll(frame)
                    elif now-readiness_started >= 8:
                        sink = getattr(context, 'gkh_hif_recovery_failure_sink', None)
                        if callable(sink):
                            sink(frame)
                        raise ValueError('HIF opening playable wait exceeded its navigation budget') from error
                    time.sleep(.1)
                except ValueError as error:
                    if poll is not None:
                        close_drink = getattr(context, 'gkh_hif_close_unexpected_drink', None)
                        if not poll(frame) and not visits and close_drink is not None:
                            close_drink(frame)
                        time.sleep(.1)
                        continue
                    event.setdefault('sampling_rejections', []).append([
                        dict(reason=str(error), evidence=getattr(error, 'evidence', None))])
                    sink = getattr(context, 'gkh_hif_recovery_failure_sink', None)
                    if callable(sink):
                        sink(frame)
                    if retried:
                        raise
                    retried = True
            phase = row['phase']
            batch = [dict(row, frame_id=frame_identifier(frame))]
            event['observations'].append(batch)
            if entry_identity is not None and not visits and phase == 'live':
                from .hif_entry_round import bind_entry, read_entry_turn
                if entry_identity.get('round_number') == round_number:
                    bound = bind_entry(context, entry_identity, batch, [read_entry_turn(context, frame)],
                        run_id='hif-recovered-'+uuid.uuid4().hex, revision=revision, game_pid=binding)
                    if bound is not None:
                        return bound
            if phase == 'live' and identities is not None:
                if (initial_live is None and not 0 <= batch[0]['stamina'] <=
                        observe_hif_notebook_identity(identities[0]['rows'], round_number=round_number)[3]):
                    raise ValueError('HIF recovered Live stamina exceeds its verified maximum')
                if initial_live is not None and content(batch[0]) != initial_live:
                    raise ValueError('HIF recovery returned to a changed Live')
                endpoints()
                run = bind_recovered_hif_run(identities,run_id='hif-recovered-'+uuid.uuid4().hex,
                                             revision=revision,round_number=round_number)
                round_context = bind_hif_round_context(identities, run=run, round_number=round_number)
                context.gkh_hif_live_context = run
                context.gkh_hif_round_context = round_context
                context.gkh_hif_context_game_pid = binding
                event.update(accepted=True,returned_to_live=True,context=asdict(run),game_pid=binding)
                event['round_context'] = asdict(round_context)
                return run,event
            if phase in visits or event['input_click_count'] >= 4:
                raise ValueError('HIF recovery navigation did not advance')
            visits.add(phase)
            if phase == 'live':
                initial_live = content(batch[0])
            if phase == 'identity':
                identities = batch
            verified = endpoints()
            frame_age_ms = (time.perf_counter()-captured_at)*1000
            if frame_age_ms > 1000:
                raise ValueError('HIF recovery input observation expired')
            x,y,w,h = batch[-1]['target']
            navigation = dict(phase=phase,target=(x,y,w,h),endpoints=verified,
                frame_ids=[row['frame_id'] for row in batch],frame_age_ms=frame_age_ms,
                captured_at=captured_at,post_click_started_at=None,post_click_finished_at=None,
                succeeded=None)
            event['inputs'].append(navigation)
            event['input_click_count'] += 1
            navigation['post_click_started_at'] = time.perf_counter()
            try:
                navigation['succeeded'] = controller.post_click(x+w//2,y+h//2).wait().succeeded
            finally:
                navigation['post_click_finished_at'] = time.perf_counter()
            if not navigation['succeeded']:
                raise ValueError('Maa HIF recovery navigation failed')
            destination = {'live':('menu',),'menu':('notebook','identity'),
                           'notebook':('identity',),'identity':('live',)}[phase]
        raise ValueError('HIF recovery navigation budget exceeded')
    except (ValueError,RuntimeError) as error:
        event['reason'] = str(error)
        raise HifRecoveryError(str(error),event) from error
    finally:
        event['elapsed_ms'] = (time.perf_counter()-started)*1000
