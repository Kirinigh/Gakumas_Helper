"""Normal Maa SKIP submission from a verified HIF end-turn decision."""
import json
import math
import time

from .inventory import _inventory_ocr
from .selection import same_live_candidate_geometry, live_selection_geometry_evidence
from .high_score.hif import build_hif_policy
from .hif_capture_adapter import captured_at as frame_captured_at
from .high_score.contracts import PlanType, ActionKind, DecisionKind, DecisionRequest, CapabilityReport


def _skip_reading(context, frame):
    rows=_inventory_ocr(context,frame,'HifEndTurnSkip',roi=[628,785,70,35],only_rec=True)
    if len(rows)!=1 or rows[0].text.strip()!='SKIP' or not .95<=rows[0].score<=1:
        raise ValueError('HIF SKIP label is not exact in all three frames')
    return dict(text=rows[0].text,confidence=rows[0].score)


def _skip_core_matches(bundle, baseline, current, event):
    if baseline.get('verification_scope') == 'rule_maintained_information_boundary_v1':
        # The same bound-state check as drinks: an exact derived card and its
        # visual upgrade family need not have identical candidate lists.
        from .hif_joint_execution import _rule_drink_core_binding

        proof = _rule_drink_core_binding(bundle, baseline, current)
        event['preinput_rule_binding'] = proof
        return proof['accepted'] is True
    return all(json.dumps(current[k]) == json.dumps(baseline['core'][k])
               for k in ('fields', 'families'))


def _capture_empty_skip_certificate(context, checkpoint, event, *, nonempty=False):
    """Read each fresh frame once, before capturing the next one."""
    timing=event.setdefault('input_timing',dict(
        schedule='nonempty_hand_per_frame' if nonempty else 'empty_hand_per_frame',frames=[]))
    cores=[]
    readings=[]
    for index in range(3):
        started=time.perf_counter()
        frame=context.tasker.controller.post_screencap().wait().get()
        captured_at=frame_captured_at(context.tasker.controller,frame)
        row=dict(index=index,captured_at=captured_at,capture_ms=(captured_at-started)*1000)
        timing['frames'].append(row)
        if frame is None:
            raise ValueError('HIF SKIP needs three fresh frames')
        if index:
            span=(captured_at-timing['frames'][0]['captured_at'])*1000
            timing['capture_span_ms']=span
            if not math.isfinite(span) or not 0<=span<=1000:
                raise ValueError('HIF SKIP capture window expired')
        core=checkpoint((frame,),single_frame=True)
        if nonempty:
            candidates=core.get('candidates',())
            geometry=live_selection_geometry_evidence(candidates)
            # The single-frame checkpoint already rejects every observed
            # SELECT. Retain the batch path's independent raised/ambiguous
            # geometry rejection, including the existing 18px raise boundary.
            if (not candidates or geometry['ambiguous'] or geometry['raised_slots']
                    or geometry['selected_slot'] is not None):
                raise ValueError('HIF SKIP requires independently unselected card geometry')
            if cores and not same_live_candidate_geometry(cores[0]['candidates'],candidates):
                raise ValueError('HIF SKIP hand geometry changed across fresh frames')
            row['unselected_geometry']=geometry
        checked_at=time.perf_counter()
        row.update(checkpoint_ms=(checked_at-captured_at)*1000,frame_ids=core['frame_ids'])
        cores.append(core)
        readings.append(_skip_reading(context,frame))
        ended=time.perf_counter()
        row['skip_ocr_ms']=(ended-checked_at)*1000
        if index<2:
            # Preserve the existing 80 ms minimum frame interval; OCR work can
            # occupy the wait instead of postponing all reads until frame 3.
            time.sleep(max(0,.08-(ended-captured_at)))
    if any(json.dumps(core[k])!=json.dumps(cores[0][k])
           for core in cores[1:] for k in ('fields','families')):
        raise ValueError('HIF empty-hand resources changed across SKIP frames')
    current=dict(cores[-1],frame_ids=tuple(key for core in cores for key in core['frame_ids']))
    if len(current['frame_ids'])!=3 or len(set(current['frame_ids']))!=3:
        raise ValueError('HIF SKIP needs three distinct fresh frame observations')
    if nonempty:
        timing['geometry_stable']=True
        current['title_family_corroboration']=[proof for core in cores
            for proof in core.get('title_family_corroboration',())]
    return current,readings,captured_at


def execute_empty_hand_turn(context, baseline, *, capture_frames, checkpoint, observe_settlement,
                            bound_policy=None):
    event=dict(event='live_end_turn',accepted=False,end_turn_submit_count=0,execution_submit_count=0)
    run=context.gkh_hif_live_context
    bundle=context.gkh_hif_opening_state
    state=bundle.state
    try:
        from .high_score.maintenance import complete_action_baseline
        if (not complete_action_baseline(bundle, baseline)
                or (baseline.get('run_id'),baseline.get('revision'))!=(run.run_id,run.revision)
                or (bundle.run_id,bundle.revision)!=(run.run_id,run.revision)
                or (state.deck.hand and bound_policy is None)
                or state.actions_remaining<1 or state.turns_remaining<1):
            raise ValueError('HIF empty-hand turn requires a complete nonterminal bound state')
        if bound_policy is None:
            if state.drinks and bundle.drink_slots is None:
                raise ValueError('HIF physical drink slots are missing')
            request=DecisionRequest('hif-empty-turn',PlanType.SENSE,DecisionKind.LIVE,state,
                CapabilityReport(True,True,True,True,True),ruleset_id=bundle.kernel.ruleset_id,
                drink_slots=bundle.drink_slots or ())
            response=build_hif_policy(bundle.kernel).decide(request)
        else:
            if bound_policy.bundle != bundle:
                raise ValueError('HIF end-turn policy state does not match observation')
            response=bound_policy.response
        event['proposal']=response.to_dict()
        if response.action is None or response.action.kind is not ActionKind.END_TURN:
            raise ValueError('HIF policy did not choose end turn')
        branches=bundle.kernel.transition(state,response.action,plan_type=PlanType.SENSE)
        from .high_score.hif_fast_native import persist_projected_events
        persist_projected_events(context,bound_policy,branches)
        if (not branches or abs(sum(b.probability for b in branches)-1)>1e-9
                or any(b.state.pending_choice or b.state.resolution_stack or b.state.turn!=state.turn+1
                       or b.state.turns_remaining!=state.turns_remaining-1 for b in branches)):
            raise ValueError('HIF next-turn outcomes are incomplete')
        event['outcome_count']=len(branches)
        context.gkh_hif_expected_empty_hand=(run.run_id,run.revision) if not state.deck.hand else None
        current,readings,captured_at=_capture_empty_skip_certificate(
            context,checkpoint,event,nonempty=bool(state.deck.hand))
        if not _skip_core_matches(bundle,baseline,current,event):
            raise ValueError('HIF bound hand or resources changed before SKIP')
        event.update(readings=readings,core={key:current[key] for key in ('fields','families','frame_ids')})
        from utils.maa_input import verify_maa_input_endpoints
        controller=context.tasker.controller
        endpoint_started=time.perf_counter()
        endpoints=verify_maa_input_endpoints(controller.info,
            expected_client_size=getattr(context,'gkh_hif_client_size',(720,1280)))
        event['input_timing']['endpoint_ms']=(time.perf_counter()-endpoint_started)*1000
        event['endpoints']=endpoints
        def before_dispatch():
            checked_at=time.perf_counter()
            age_ms=(checked_at-captured_at)*1000
            event['input_timing']['final_capture_age_ms']=age_ms
            if not math.isfinite(age_ms) or not 0<=age_ms<1000:
                raise ValueError('HIF SKIP input certificate expired')
            event.update(end_turn_submit_count=1,execution_submit_count=1,
                         input_may_have_been_dispatched=True)
            return checked_at
        guarded_click=getattr(context,'gkh_hif_post_live_click',None)
        if callable(guarded_click):
            job=guarded_click(663,802,before_dispatch=before_dispatch,
                input_deadline=captured_at+1,authorization_floor=captured_at)
        else:
            before_dispatch()
            job=controller.post_click(663,802)
        if not job.wait().succeeded:
            raise ValueError('HIF Maa SKIP input did not succeed')
        context.gkh_hif_expected_empty_hand=None
        context.gkh_hif_committed_state=None
        context.gkh_hif_committed_event=None
        time.sleep(.8)
        if all(branch.state.terminal for branch in branches):
            settlement=observe_settlement(tuple(b.state for b in branches),baseline)
            event.update(accepted=settlement.get('accepted') is True,settlement=settlement,
                         reason=settlement.get('reason'))
            return event
        arrival_started=time.perf_counter()
        expected_fields={(b.state.turns_remaining,b.state.stamina,b.state.genki,
                          b.state.score,b.state.score_multiplier) for b in branches}
        for attempt in range(20):
            try:
                arrived=checkpoint(capture_frames())
                if tuple(arrived['fields'][:5]) not in expected_fields:
                    raise ValueError('HIF next-turn public resources are not settled')
                event['arrival_core']={key:arrived[key] for key in ('fields','families','frame_ids')}
                break
            except ValueError as error:
                event.setdefault('arrival_observations',[]).append(str(error))
                if attempt==19 or time.perf_counter()-arrival_started>12:
                    raise
                time.sleep(.2)
        settlement=observe_settlement(tuple(b.state for b in branches),baseline)
        event.update(accepted=settlement.get('accepted') is True,settlement=settlement,
            reason=settlement.get('reason'))
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as error:
        event['reason']=str(error)
    finally:
        context.gkh_hif_expected_empty_hand=None
    return event
