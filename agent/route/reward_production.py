"""Current public-course receipt and read-only post-reward inventory census."""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict

from . import rewards
from .hif import active_hif_role
from .host import HifRoutePage
from .engine import Target, Outcome, Decision, Observation, propose_unique
from .deletion_production import PICK_DESTINATIONS, SEARCH_DESTINATIONS, DeletionTargetNotFound


def reward_transition_evidence(rows):
    # 182 retains the HIF side badge while the character/world UI is in
    # transition. This is diagnostic evidence only, never a destination,
    # resource receipt, role identity, or permission to tap the scene.
    if any(row['score'] >= .95 and row['text'] in ('H.I.F本戦まで', '所持スキルカード', '所持Pアイテム',
            'SELECT', 'SKIP', '早送り', 'SP') for row in rows):
        return None
    matches = [row for row in rows if row['score'] >= .95 and row['text'] == '輝き'
        and 650 <= row['box'][0] and 380 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 720 and row['box'][1] + row['box'][3] <= 540]
    return dict(kind='observed_hif_side_badge_during_unresolved_transition', label=matches[0]) if len(matches) == 1 else None


def navigation_checkpoint(host, gate):
    value = rewards.request(host)
    page = host._page
    pending = gate.pending
    if (value is None or value['stage'] not in rewards.ACTIVE_STAGES or pending is None
            or pending.target.kind not in rewards.REWARD_NAVIGATION or page is None
            or page.observation != pending.source or not isinstance(page.reward, dict)):
        raise ValueError('public-course census input lacks its exact current read-only page')
    proof = page.reward
    rewards.reward_frames(proof, value['source_frame_ids'])
    if tuple(proof['frame_ids']) != page.frame_ids:
        raise ValueError('public-course census input frames differ from its page')
    if pending.target.kind == 'training_reward_ack':
        if (value['stage'] != 'awaiting_ack' or proof.get('kind') != 'result'
                or proof.get('attribute') != value['attribute'] or proof.get('special') is not value['special']):
            raise ValueError('public-course acknowledgement is not its original result')
        rewards.validate_resources(host, proof)
    elif pending.target.kind == 'training_reward_upgrade_ack':
        if (value['stage'] != 'upgrade' or proof.get('kind') != 'upgrade'
                or proof.get('grant') != value['upgrade']['grant']):
            raise ValueError('course upgrade acknowledgement changed its observed card')
        rewards.validate_resources(host, proof)
    elif pending.target.kind.startswith('reward_delete_'):
        if (value['stage'] != 'deletion' or proof.get('kind') not in ('delete_choice', 'delete_search', 'delete_reselect', 'delete_selected', 'delete_confirm', 'delete_result')
                or proof.get('deletion', {}).get('request') != value['deletion']['request']):
            raise ValueError('course deletion input lacks its exact retained original instance')
        if proof.get('kind') == 'delete_reselect' and value['deletion'].get('last_menu', {}).get('kind') != 'delete_reselect':
            raise ValueError('free deletion reselection lacks its verified original miss')
    elif pending.target.kind == 'recovery_inventory':
        if value['stage'] != 'census' or proof.get('kind') != 'home' or proof.get('day') != value['next_day']:
            raise ValueError('public-course census must open from its actual next calendar day')
    elif proof.get('kind') != 'inventory' or proof.get('inventory', {}).get('census_id') != value['transaction_id']:
        raise ValueError('public-course census cannot use the old initial inventory')
    result = deepcopy(value)
    if pending.target.kind in ('reward_delete_select', 'reward_delete_scroll'):
        result['deletion'] = deepcopy(proof['deletion'])
    if proof.get('inventory') is not None:
        from .nonbattle_inventory import retain_inventory_registration
        result['inventory'] = retain_inventory_registration(host, page, proof['inventory'])
    result['navigation_source_frame_ids'] = list(proof['frame_ids'])
    return result


def decide_reward(host, observation, config):
    if host._page is None or host._page.observation != observation:
        return Decision(Outcome.STOP, 'hif_reward_current_page_not_bound')
    proof = host._page.reward
    value = rewards.request(host)
    if not isinstance(proof, dict) or value is None:
        return Decision(Outcome.STOP, 'hif_reward_original_reservation_required')
    if proof['kind'] == 'delete_result':
        if value.get('deletion', {}).get('status') != 'committed_pending_census':
            return Decision(Outcome.STOP, 'deletion_result_requires_original_successful_removal')
        return propose_unique(observation, observation.targets, 'hif-acknowledge-original-course-deletion', ('reward_home',))
    if proof['kind'] == 'delete_search':
        return propose_unique(observation, observation.targets, 'hif-locate-requested-deletion-instance', SEARCH_DESTINATIONS)
    if proof['kind'] in ('delete_choice', 'delete_reselect', 'delete_selected', 'delete_confirm'):
        if value.get('deletion', {}).get('status') != 'active':
            return Decision(Outcome.STOP, 'course_deletion_already_removed_or_unbound')
        from dataclasses import replace

        from .ledger import HifLedger
        from .deletion import plan_deletion
        generic = {'reward_delete_select': 'select_delete_card', 'reward_delete_open': 'open_delete_confirmation',
            'reward_delete_confirm': 'execute_card_delete'}
        translated = replace(observation, scene='reward_delete_choice' if proof['kind'] == 'delete_reselect' else observation.scene,
            targets=tuple(replace(target, kind=generic[target.kind]) for target in observation.targets))
        shadow = HifLedger(host.ledger.pending.before if value.get('source_target') == 'finish_consult'
            else host.ledger.pending.expected, card_catalog=host._catalog, drink_ids=host._drink_ids)
        retained = dict(status='active', request=value['deletion']['request'])
        planned = plan_deletion(translated, shadow, host._page.deletion, retained)
        if planned.decision.outcome != Outcome.ACTION:
            return planned.decision
        expected = {'delete_choice': PICK_DESTINATIONS, 'delete_reselect': ('reward_delete_selected',),
            'delete_selected': ('reward_delete_confirm', 'reward_delete_result'),
            'delete_confirm': ('reward_delete_result',)}[proof['kind']]
        return propose_unique(observation, observation.targets, planned.decision.proposal.rule_id, expected)
    retained = rewards.retained_course_deck(value, host.ledger.pending.before)
    unavailable = proof.get('inventory_opened') is False and rewards.optional_course_deck_review(value)
    if proof['kind'] == 'home' and not observation.targets and (retained or unavailable or (value.get('inventory') or {}).get('complete') is True):
        host.bind_observed_training_reward(host._page)
        return Decision(Outcome.PAUSE, 'hif_observed_training_reward_complete' if retained or unavailable else 'hif_observed_training_reward_census_complete')
    destination = {'result': rewards.COURSE_RESULT_DESTINATIONS, 'upgrade': rewards.COURSE_RESULT_DESTINATIONS,
        'home': ('reward_inventory',),
        'inventory': ('reward_home',) if ((proof.get('inventory') or {}).get('complete') is True
            or rewards.retained_course_deck(dict(value, inventory=proof.get('inventory')), host.ledger.pending.before))
            and all(target.kind == 'recovery_close' for target in observation.targets) else ('reward_inventory',)}[proof['kind']]
    return propose_unique(observation, observation.targets, 'hif-observe-original-public-course-rewards', destination)


class HifRewardProducer:
    def __init__(self, parent):
        self.parent = parent
        self.inventory = None
        self.inventory_scope = None

    def _read(self, image, rows):
        from .daily import complete_daily_resources
        from .calendar import read_calendar, read_route_resources, read_calendar_countdown
        from .production import read_bound_drink_slots, read_nonbattle_inventory
        from .nonbattle_observation import _within
        host = self.parent.host
        value = rewards.request(host)
        from .deletion_production import read_course_deletion_shell, read_course_deletion_result
        deleted = read_course_deletion_result(self.parent, image, rows, value)
        if deleted is not None:
            return deleted
        pending = host.hooks.transition_gate.pending
        deletion = read_course_deletion_shell(image, rows, value,
            opened=pending is not None and pending.target.kind == 'reward_delete_open')
        if deletion is not None:
            return deletion
        if value['stage'] == 'awaiting_ack' and any(row['score'] >= .95 and row['text'] == value['attribute'] + '.公開レッスン'
                and row['box'][0] < 200 and row['box'][1] < 80 for row in rows):
            from .training import read_training_result, complete_training_sp_field
            # This result was already independently observed when confirming
            # the course. Still read both HUD numbers from the current image.
            result = value.get('result') or {}
            completed, crops = complete_daily_resources(self.parent.context, image, rows,
                submitted_stamina=(result.get('stamina'), result.get('max_stamina')))
            choice = host.handoff['nonbattle_context']['training_choice']
            completed, sp_proof = complete_training_sp_field(self.parent.context, image, completed,
                expected_special=choice['selected_special'])
            page = read_training_result(completed, choice)
            rewards.validate_resources(host, page)
            return dict(page, kind='result', resource_crops=crops, sp_field_evidence=sp_proof,
                target=page['result_target'], target_kind='training_reward_ack', target_key='training_reward_ack')
        gate = host.hooks.transition_gate
        if value['stage'] == 'awaiting_ack' and (gate.pending is None or gate.pending.target.kind != 'training_reward_ack'):
            raise ValueError('public-course result must be acknowledged before its census')
        if value['stage'] in ('awaiting_ack', 'upgrade') and value.get('source_target') != 'finish_consult':
            from .activity_production import support_award
            award = support_award(image, rows, host, upgrade_only=True)
            if award is not None:
                completed, crops = complete_daily_resources(self.parent.context, image, rows)
                resources = read_route_resources(completed)
                rewards.validate_resources(host, resources)
                return dict(resources, kind='upgrade', day=value['day'], grant=award['grant'], resource_crops=crops,
                    target=award['target'], target_kind='training_reward_upgrade_ack', target_key='training_reward_upgrade_ack')
        try:
            day, _ = read_calendar(rows)
            require_card_control = False
        except ValueError:
            day = None
            require_card_control = True
            home = value.get('home') or {}
            if (host.ledger.pending.effect.target_key == 'confirm_training' and
                    (value['stage'] in ('awaiting_ack', 'upgrade') or value['stage'] == 'census'
                     and home.get('kind') == 'home' and home.get('day') == value['next_day'])):
                # fb8 RestAuto recognizes these day-three/six schedules by
                # the countdown and the visible 休む control.
                try:
                    day, _ = read_calendar_countdown(rows)
                except ValueError:
                    pass
                if day in (3, 6):
                    require_card_control = not any(row['text'] == '休む'
                        and _within(row['box'], (580, 755, 140, 150)) for row in rows)
                if day is not None and value['stage'] == 'awaiting_ack':
                    from .input_receipt import original_input_report
                    original_input_report(host, gate)
        if day is not None:
            completed, crops = complete_daily_resources(self.parent.context, image, rows)
            if day != value['next_day']:
                raise ValueError('public-course reward calendar is not the actual next day')
            resources = read_route_resources(completed)
            rewards.validate_resources(host, resources)
            from .footer import read_footer_offset
            offset, footer_evidence = read_footer_offset(image, context=self.parent.context)
            slots = read_bound_drink_slots(self.parent.context, image, host.ledger.pending.before.drinks,
                footer_offset=offset, maintained_revision=host.ledger.pending.before.revision)
            page = dict(resources, day=day, kind='home', drink_slots=slots,
                drinks=[asdict(drink) for drink in host.ledger.pending.before.drinks], resource_crops=crops,
                footer_offset=offset, footer_evidence=footer_evidence)
            open_inventory = ((value.get('inventory') or {}).get('complete') is not True
                and not rewards.retained_course_deck(value, host.ledger.pending.before))
            if open_inventory or require_card_control:
                from .production import recognize_route_inventory_button as recognize_inventory_button
                matches, _, _, _ = recognize_inventory_button(self.parent.context, image,
                    tuple(self.parent.context.gkh_hif_client_size))
                if not matches and not require_card_control and rewards.optional_course_deck_review(value):
                    return dict(page, inventory_opened=False)
                if len(matches) != 1:
                    error = ValueError('public-course reward owned-card control is not uniquely observed')
                    error.evidence = dict(field='reward_inventory_control',candidate_count=len(matches))
                    raise error
                if open_inventory:
                    page.update(target=tuple(matches[0].box), target_kind='recovery_inventory', target_key='recovery_inventory')
            return page
        page = read_nonbattle_inventory(rows)
        if page['phase'] != 'inventory':
            raise ValueError('public-course reward requires the skill-card inventory')
        from .nonbattle_inventory import bind_inventory_census
        saved = value.get('inventory')
        if saved is not None:
            rewards.reward_frames(dict(frame_ids=saved.get('review_frame_ids')), ())
            if not saved.get('census_session'):
                raise ValueError('retained course inventory lacks its original opening')
        saved = bind_inventory_census(self, ['training', value['transaction_id']], saved)
        if rewards.optional_course_deck_review(value):
            return dict(kind='inventory', target=page['target'], target_kind='recovery_close', target_key='recovery_close')
        if self.inventory is None:
            from .nonbattle_inventory import FlatInventoryCollector, load_flat_inventory_readers
            self.inventory = FlatInventoryCollector(*load_flat_inventory_readers())
            self.inventory.census_session = self.inventory_scope
            if saved is not None:
                try:
                    self.parent._restore_inventory(collector=self.inventory, retained=saved, phase='reward_inventory')
                    if self.inventory.previous is None:
                        raise ValueError('partial public-course census original image registration was not restored')
                except BaseException:
                    self.inventory = None
                    raise
        return dict(kind='inventory', target=page['target'], target_kind='recovery_close', target_key='recovery_close')

    def _review_ordinary_deck(self, value, images, rows, keys, gate):
        from .nonbattle_inventory import FlatInventoryCollector, flat_inventory_count, load_flat_inventory_readers
        if not len(images) == len(rows) == len(keys) == 1:
            raise ValueError('course inventory review requires one current frame')
        # hiro reviews an empty/unfinished deck once and closes the physical
        # list even if its total, a thumbnail or the final scroll cannot be read.
        saved = value.get('inventory') or {}
        if saved.get('census_id') != value['transaction_id'] or saved.get('census_session') != self.inventory_scope:
            saved = {}
        result = deepcopy(saved)
        scroll = gate.pending is not None and gate.pending.target.kind == 'recovery_inventory_scroll'
        detail = (int(gate.pending.target.key.split(':')[1]) if gate.pending is not None
            and gate.pending.target.kind == 'recovery_inventory_detail' else None)
        if not result.get('read_finished'):
            try:
                if self.inventory is None:
                    self.inventory = FlatInventoryCollector(*load_flat_inventory_readers())
                    self.inventory.census_session = self.inventory_scope
                    if saved and not saved.get('complete'):
                        self.parent._restore_inventory(collector=self.inventory, retained=saved, phase='reward_inventory')
                        if self.inventory.previous is None:
                            raise ValueError('original course deck view could not be registered')
                result = self.inventory.observe(self.parent.context, images, rows, keys, inspect_index=detail,
                    allow_unknown_badges=True, allow_unknown_total=True)
                result['review_screens'] = saved.get('review_screens', 1) + int(scroll)
                # hiro 8380622 deck_review.MAX_SCREENS finishes this review at six screens.
                result['read_finished'] = (result['review_screens'] >= 6 or not result['visible_boxes']
                    or len(result['slots']) == result['total'] or
                    scroll and result['displacement'] <= saved.get('displacement', 0))
                if detail is not None and result.get('selected_index') != detail:
                    result.update(complete=False, read_finished=True, read_error='original detail unreadable')
            except (ValueError, RuntimeError, OSError) as error:
                if not saved:
                    try:
                        totals = {flat_inventory_count(group) for group in rows}
                    except ValueError:
                        totals = set()
                    result = dict(total=totals.pop() if len(totals) == 1 else None,
                        displacement=0, slots=[], visible_boxes={}, selected_index=None, frame_ids=list(keys))
                result.update(complete=False, read_finished=True, read_error=str(error))
        result.update(review_frame_ids=list(keys), census_id=value['transaction_id'], census_session=deepcopy(self.inventory_scope))
        if result.get('complete') or result['read_finished']:
            return result, None
        return result, dict(kind='recovery_inventory_scroll', key=f'inventory-scroll:{result["displacement"]}',
            target=(350, 770, 1, 80))

    def observe(self, *, reconciliation=False):
        from .production import _current_rows, nonbattle_drink_slots_consistent
        parent, host = self.parent, self.parent.host
        value = rewards.request(host)
        gate = host.hooks.transition_gate
        deadline = parent.clock.now() + (gate.timeout_ms if gate.pending is not None and not reconciliation else 8000)
        if gate.pending is not None and not reconciliation:
            deadline = min(deadline, gate.issued_at_ms + gate.timeout_ms)
        scrolling = gate.pending is not None and gate.pending.target.kind == 'recovery_inventory_scroll'
        if gate.pending is not None and gate.pending.target.kind == 'reward_delete_scroll':
            remaining = gate.issued_at_ms + 800 - parent.clock.now()
            if remaining > 0:
                time.sleep(remaining / 1000)
        picking = gate.pending is not None and gate.pending.target.kind == 'reward_delete_select'
        retry_ready = (value.get('deletion') or {}).get('last_menu', {}).get('kind') == 'delete_reselect'
        if picking:
            remaining = gate.issued_at_ms + 600 - parent.clock.now()
            if remaining > 0:
                time.sleep(remaining / 1000)
        pick_deadline = parent.clock.now() + 5000 if picking else None
        pick_checks = 0
        previous = None
        if scrolling:
            # hiro deck_review._still checks viewport motion after a swipe.
            # Only the resulting image is read for cards, resources and input.
            time.sleep(.8)
        while True:
            started = parent.clock.now()
            if started >= deadline or parent.context.tasker.stopping:
                raise ValueError('public-course reward observation reached its original deadline')
            capture = parent.context.tasker.controller.post_screencap().wait()
            if not capture.succeeded:
                raise ValueError('public-course reward capture failed')
            image = capture.get()
            key = f'route-frame-{parent.game_pid}-{started}'
            if scrolling:
                from .first_day import deck_view_changed
                if previous is None:
                    settle_until = min(deadline, parent.clock.now() + 2500)
                if previous is None or started < settle_until and deck_view_changed(previous, image):
                    previous = image
                    time.sleep(.3)
                    continue
                scrolling = False
            rows = _current_rows(parent.context, image)
            try:
                page = self._read(image, rows)
                if page['kind'] in ('delete_choice', 'delete_selected', 'delete_confirm'):
                    from .deletion_production import bind_course_deletion, current_deletion_pick
                    if page['kind'] != 'delete_confirm' and (picking or retry_ready):
                        pick_checks += 1
                        page = current_deletion_pick(parent, image, rows, key, value, page)
                        if picking and page['kind'] == 'delete_reselect':
                            if parent.clock.now() < pick_deadline or pick_checks < 3:
                                time.sleep(.15)
                                continue
                            if gate.pending.source.scene == 'reward_delete_reselect':
                                raise ValueError('original deletion instance is still not selected after its one reselection')
                    else:
                        page = bind_course_deletion(parent, [image], [rows], (key,), value, page)
            except ValueError as error:
                metadata = dict(frame_id=key, phase='unknown', reason=str(error), rows=rows)
                if isinstance(getattr(error, 'evidence', None), dict):
                    metadata['field_evidence'] = error.evidence
                if gate.pending is not None and gate.pending.target.kind == 'training_reward_ack':
                    transition = reward_transition_evidence(rows)
                    if transition is not None:
                        metadata.update(phase='reward_transition_wait', transition_evidence=transition,
                            original_issued_at_ms=gate.issued_at_ms, original_timeout_ms=gate.timeout_ms)
                parent.events.append(metadata)
                if parent.evidence_sink:
                    parent.evidence_sink(image, metadata)
                from .training import unresolved_training_sp_field
                weak_original_result = value['stage'] == 'awaiting_ack' and unresolved_training_sp_field(error)
                missing_navigation = getattr(error,'evidence',{}) == dict(field='reward_inventory_control',candidate_count=0)
                if isinstance(error, DeletionTargetNotFound) or gate.pending is None and not weak_original_result and not missing_navigation:
                    raise
                if picking and parent.clock.now() >= pick_deadline and pick_checks >= 3:
                    raise
                time.sleep(.1)
                continue
            if parent.evidence_sink:
                parent.evidence_sink(image, dict(frame_id=key, phase='reward_' + page['kind'], observation=page, rows=rows))
            images, observations, keys = [image], [rows], (key,)
            if page['kind'] == 'inventory' and rewards.optional_course_deck_review(value):
                inventory, navigation = self._review_ordinary_deck(value, images, observations, tuple(keys), gate)
                if navigation is not None:
                    page.update(target=navigation['target'], target_kind=navigation['kind'], target_key=navigation['key'])
                page['inventory'] = inventory
            elif page['kind'] == 'inventory':
                index = int(gate.pending.target.key.split(':')[1]) if gate.pending is not None and gate.pending.target.kind == 'recovery_inventory_detail' else None
                try:
                    # Unconfirmed detail batches remain diagnostic only. Keep
                    # the original registered source for the next fresh batch.
                    original = {name: getattr(self.inventory, name) for name in (
                        'total', 'displacement', 'previous', 'previous_boxes', 'boxes', 'slots', 'frame_ids',
                        'detail_rejection', 'detail_wait_reason', 'selected_index', 'previous_frame_ids', 'registration')}
                    inventory = self.inventory.observe(parent.context, images, observations, tuple(keys), inspect_index=index)
                    if index is not None and inventory.get('selected_index') != index:
                        parent.events.append(dict(phase='reward_detail_unconfirmed', target_index=index,
                            expected_card_id=self.inventory.slots.get(index, {}).get('card_id'),
                            reason=inventory.get('detail_rejection') or inventory.get('detail_wait_reason')
                                or 'original detail instance is not yet confirmed',
                            frame_ids=list(keys), original_issued_at_ms=gate.issued_at_ms,
                            original_timeout_ms=gate.timeout_ms))
                        for name, value_before in original.items():
                            setattr(self.inventory, name, value_before)
                        continue
                except ValueError as error:
                    if str(error) not in ('inventory scroll has no unique overlapping translation',
                            'flat inventory scroll has not settled', 'inventory customization badge is ambiguous'):
                        raise
                    time.sleep(.1)
                    continue
                inventory = dict(inventory, census_id=value['transaction_id'], review_frame_ids=list(keys))
                navigation = self.inventory.next_navigation()
                if navigation is not None:
                    page.update(target=navigation['target'], target_kind=navigation['kind'], target_key=navigation['key'])
                page['inventory'] = inventory
            elif page['kind'] == 'home':
                if not nonbattle_drink_slots_consistent(page['drink_slots']):
                    raise ValueError('public-course reward drink controls do not identify all four slots')
            break
        boxes = {page['target_key']: tuple(page['target'])} if 'target' in page else {}
        revision = parent.clock.now()
        scene = 'training_result' if page['kind'] == 'result' else 'reward_' + page['kind']
        inventory = page.get('inventory', {})
        inventory_view = ((inventory.get('total'), tuple(sorted(map(int, inventory.get('visible_boxes', {})))))
            if page['kind'] == 'inventory' else None)
        page_id = (f'reward:{value["transaction_id"]}:result' if page['kind'] == 'result' else
            f'reward:{value["transaction_id"]}:{page["kind"]}:{inventory_view}:{inventory.get("selected_index")}')
        if inventory.get('read_finished') and not inventory.get('complete'):
            page_id += ':incomplete'
        if page['kind'] == 'upgrade':
            page_id += f':{page["grant"]["source_instance_id"]}:{page["grant"]["card_id"]}'
        if page['kind'] == 'delete_search':
            cursor = page['deletion']['search']
            page_id += f':{cursor["direction"]}:{cursor["steps"]}'
        observation = Observation(host.run_id, revision, scene,
            page_id,
            value['day'] if page['kind'] in ('result', 'upgrade') or page['kind'].startswith('delete_') else value['next_day'],
            *active_hif_role(host), revision - started, len(keys), 'HIF',
            tuple(Target(key, page['target_kind'], available=True) for key in boxes))
        proof = dict(page, frame_ids=list(keys))
        parent.events.append(dict(phase=scene, frame_ids=list(keys), capture_age_ms=observation.capture_age_ms))
        parent.last_frame_ids = tuple(keys)
        menu = None
        if page['kind'] in ('delete_choice', 'delete_reselect', 'delete_selected', 'delete_confirm'):
            from .ledger import CardInstance
            from .deletion import DeletionMenu, DeletionRequest
            raw = proof['deletion']['request']['card']
            card = CardInstance(**dict(raw, customization=tuple(map(tuple, raw['customization']))))
            stage = {'delete_choice': 'choice', 'delete_reselect': 'choice', 'delete_selected': 'selected', 'delete_confirm': 'confirm'}[page['kind']]
            menu = DeletionMenu(DeletionRequest(card, 0), stage, None if stage == 'choice' else card,
                'reward_delete_select', 'reward_delete_open', 'reward_delete_confirm', observed_fee=page.get('fee'))
        return HifRoutePage(observation, boxes, frame_ids=tuple(keys), reward=proof, deletion=menu)

    def prepare(self, gate, report):
        if gate.pending is None or gate.pending.target.kind not in ('training_reward_ack', 'training_reward_upgrade_ack',
                'reward_delete_select', 'reward_delete_scroll', 'reward_delete_open', 'reward_delete_confirm', 'reward_delete_result_ack',
                'recovery_inventory', 'recovery_inventory_detail', 'recovery_inventory_scroll', 'recovery_close'):
            raise ValueError('cold reward navigation needs its original registered inventory image context')
        proof = self.observe(reconciliation=True)
        if gate.pending.target.kind in ('training_reward_ack', 'training_reward_upgrade_ack', 'recovery_close'):
            return self.parent.host.reconcile_reward_acknowledgement(gate, report, proof)
        if gate.pending.target.kind.startswith('reward_delete_'):
            return self.parent.host.reconcile_reward_deletion_navigation(gate, report, proof)
        return self.parent.host.reconcile_reward_inventory_navigation(gate, report, proof)
