"""Resume a held card after a proven undispatched preparation and noisy reread."""


def selected_noise_stop(row):
    from ..opening_observer import (
        selected_refresh_noise_reobserve_allowed,
        _selected_resource_noise_reobserve_allowed,
    )
    from ..hif_live_recovery import closed_observer_record
    from .native_card_evidence import _fresh, _frames
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    try:
        if (row.get('status') != 'STOP' or row.get('live_return') is not False
                or row.get('error') or row.get('failure_event') or row.get('transaction_closed')
                or not closed_observer_record(row) or not row.get('native_inprocess_source')
                or row.get('input_count') != 1 or len(row.get('native_input_dispatches', ())) != 1
                or row.get('native_navigation_preparation') or len(row['live_events']) != 1):
            return False
        report = row['live_events'][0]
        if (report.get('event') != 'live_selection_stop' or report.get('executed') is not False
                or report.get('runtime_phase') != 'execute'
                or report.get('reason') != 'authoritative_live_resource_transaction_unresolved'
                or report.get('execution_selection_click_count') != 1
                or report.get('execution_submit_count') != 0
                or any(report.get(k, 0) for k in ('identity_probe_click_count', 'drinks_consumed', 'end_turn_submit_count'))
                or report.get('foreground_preparation_events') or report.get('shadow_transaction_events')):
            return False
        first, previous = report['authoritative_refresh_events']
        failed, retry, noise, final = report['failure_events']
        action = first['proposal']['candidate_traces']['high_score']['action']
        slot = action['source_slot']
        if action.get('kind') != 'PLAY_CARD' or type(slot) is not int or slot < 0 or not action.get('card_instance_id'):
            return False
        _fresh(first, slot)
        _fresh(previous, slot)
        preview = previous['resource_transaction']['card_effect_preview']
        if (first.get('phase') != 'authoritative_refresh_a'
                or previous.get('phase') != 'authoritative_refresh_b'
                or previous.get('expected_selected_slot') != slot
                or previous.get('generation') != dict(first['generation'],score=preview['preview_score'])
                or previous.get('allow_selected_score_preview_change') is not True
                or previous.get('expected_generation') != first.get('generation')
                or previous['proposal']['candidate_traces']['high_score']['action'] != action
                or previous.get('resource_transaction', {}).get('accepted') is not True
                or previous.get('select_consensus', {}).get('observations') != [[slot]]*3):
            return False
        guard = row['native_input_preparation']
        if (guard.get('dispatch_state') != 'rejected_before_dispatch'
                or guard.get('input_may_have_been_dispatched') is not False
                or guard.get('reason') != 'snapshot_expired_after_input_preparation'
                or guard.get('input_deadline') != previous['input_deadline']
                or not previous['authorization_floor'] <= guard['authorization_floor'] < previous['input_deadline']
                or failed != dict(phase='execution_submit', attempted_slot=slot,
                    authorization_floor=guard['authorization_floor'], click_dispatched=False,
                    input_may_have_been_dispatched=False, reason=guard['reason'])
                or retry != dict(phase='execution_submit_reobservation',attempted_slot=slot,
                    click_dispatched=False,input_may_have_been_dispatched=False,
                    reason='final_preparation_expired_without_dispatch',retry_count=1,
                    prior_frame_ids=previous['frame_ids'])):
            return False
        steps = report['execution_selection_events']
        if (not steps or steps[0] != dict(phase='execution_selection',attempted_slot=slot,
                accepted=True,click_dispatched=True,reason='target_selection_dispatched',
                authorization_floor=steps[0].get('authorization_floor'))
                or not first['authorization_floor'] <= steps[0]['authorization_floor'] < first['input_deadline']
                or any(s.get('phase') not in ('hif_selected_single_plus','hif_selected_skill_support')
                    or s.get('click_dispatched') is not False or s.get('receipt',{}).get('accepted') is not True
                    for s in steps[1:])):
            return False
        groups = [_frames(e) for e in (first,previous,noise,final)]
        if len(set().union(*groups)) != 12:
            return False
        count = first['generation']['hand_count']
        if not selected_refresh_noise_reobserve_allowed(final,row['formal_params'],slot,count):
            return False
        for event in (noise,final):
            if (event.get('phase') != 'authoritative_refresh_b'
                    or event.get('click_dispatched') is not False
                    or event.get('expected_selected_slot') != slot
                    or event.get('geometry_stable') is not True
                    or event.get('selected_identity_authorized') is not True
                    or event.get('select_consensus', {}).get('observations') != [[slot]]*3
                    or event['resource_transaction'].get('card_effect_preview') != preview
                    or event['resource_transaction'].get('frame_ids') != event['frame_ids']
                    or not _selected_resource_noise_reobserve_allowed(event['resource_transaction'],[.95]*3)
                    or any(event.get(k) != v for k,v in
                        (('health_votes',3),('playable_votes',3),('secondary_votes',0),('popup_votes',0)))):
                return False
        dispatched = row['native_input_dispatches'][0]
        for event, target in ((first,dispatched),(previous,guard)):
            from ..hif_visual import hif_selection_click_box

            box = hif_selection_click_box(event['candidate_boxes'][-1],slot)
            if (target.get('kind') is not None
                    or (target.get('x'),target.get('y')) != (box[0]+box[2]//2,box[1]+box[3]//2)):
                return False
            _verified_endpoints(target['endpoints'],row['endpoints']['game']['pid'])
            _same_input_host(row['endpoints'],target['endpoints'])
        return True
    except (KeyError,TypeError,ValueError,AttributeError,IndexError):
        return False


def require_repaired_selected_noise(checkpoint):
    """Replay old pixels and state, retaining old failure and its rejected intent."""
    import cv2

    from ..hif_gray_selection import ordinary_selected_resource_preview

    row, module = checkpoint['row'], checkpoint['observer_module']
    if (not selected_noise_stop(row) or checkpoint.get('selected') is not True
            or checkpoint.get('pending') or checkpoint['policy']._phase != 'pending_intent'):
        raise ValueError('HIF noisy resource recovery lacks one undispatched held selection')
    expected = ordinary_selected_resource_preview(checkpoint['bundle'],
        complete_event=checkpoint['event'],journal=row['gray_selection_journal'])
    pending = checkpoint['policy']._pending
    if (pending[0] != checkpoint['state'] or pending[1].source_slot != expected['selected_slot']
            or pending[1].card_instance_id != checkpoint['state'].deck.hand[expected['selected_slot']].instance_id):
        raise ValueError('HIF noisy resource recovery changed the retained action')
    for event in row['live_events'][0]['failure_events'][2:]:
        transaction = event['resource_transaction']
        if transaction['card_effect_preview'] != expected:
            raise ValueError('HIF noisy resource preview differs from complete restored rules')
        for name in ('stamina','shield'):
            original = transaction['observations'][name+'_delta']['observations']
            observations = []
            for fid, old in zip(event['frame_ids'],original,strict=True):
                frames = [frame for frame in row['frames'] if frame['frame_id'] == fid]
                if len(frames) != 1:
                    raise ValueError('HIF noisy resource original image is ambiguous')
                image = cv2.imread(frames[0]['path'])
                if image is None or module.frame_identifier(image) != fid:
                    raise ValueError('HIF noisy resource original image changed or is missing')
                _, proof = module._build_red_only_resource_delta_image(image,old['roi'],allow_subcomponent_zero=True)
                if {k:v for k,v in proof.items() if k != 'subcomponent_specks'} != old['red_evidence']:
                    raise ValueError('HIF noisy resource original pixel evidence changed')
                observations.append(dict(old,red_evidence=proof))
            resolved = module._resolve_resource_delta_consensus(tuple(observations),
                baseline_value=expected['baseline_'+name],preview_value=transaction['preview_'+name])
            if resolved['accepted'] is not True or resolved['value'] != transaction['expected_'+name+'_delta']:
                raise ValueError('HIF noisy resource evidence is not repaired')
