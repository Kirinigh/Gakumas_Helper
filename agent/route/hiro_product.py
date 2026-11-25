"""The full Hiro campaign under a single real Maa Produce task."""

import os
import json
from types import SimpleNamespace
from pathlib import Path
from contextlib import ExitStack, contextmanager

from agent.live_decision.hiro_runtime import hiro_module, hiro_route_file
from agent.live_decision.hiro_maa_device import HiroMaaDevice
from agent.live_decision.cultivation_maa_exam import CultivationEntry, cultivation_idol
from agent.live_decision.high_score.tower_gnrpa import load_native_tower

from .hiro_hif import _read_ocr, run_hiro_hif_campaign
from .hiro_exam import play_cultivation_exam
from .hiro_opening import HiroOpening
from .hiro_hif_exam import play_hif_exam, play_formal_hif_finals
from .hiro_cultivation import _replace, nia_fallback, _capture_sink, maa_next_nodes, run_hiro_cultivation


@contextmanager
def qualifier_formation(rt, settings, context, device, report):
    """Use the already supported local gallery and memory scans inside Hiro setup."""
    from utils.runtime_paths import resolve_runtime_path
    from live_decision.catalog import SkillCardFeatureCatalog
    from live_decision.hif_capture_adapter import adapt_hif_context

    from .qualifier import QualifierUi, _open_rental, rent_support, _support_step, scan_qualifier_memories
    from .qualifier_support import FormationValue, SupportGallery, read_own_support, write_own_support

    card_ids = tuple(filter(None, settings.get('memory_cards', '').split(',')))
    support_ids = tuple(filter(None, settings.get('support_cards', '').split(',')))
    gallery_path = settings.get('support_gallery')
    score_path = settings.get('support_effects')
    gallery = SupportGallery(os.path.expandvars(gallery_path)) if gallery_path else None
    score = FormationValue.load(os.path.expandvars(score_path), settings['attribute']) if score_path else None
    if (support_ids or score is not None) and gallery is None:
        raise ValueError('HIF support composition requires its configured local gallery')
    ui = QualifierUi(adapt_hif_context(context), device.authorise, clock=rt.clock, sleep=rt.sleep)
    if card_ids:
        data = resolve_runtime_path(Path('assets/data/skill_cards.csv'), Path('data/skill_cards.csv'))
        nav = hiro_module('runtime.nav')
        nav.goto_home(rt)
        report['memory_inventory'] = scan_qualifier_memories(ui, int(settings['memory_set']),
            SkillCardFeatureCatalog.load(data), card_ids)

    def write_support(_device, plan):
        start = ui.need(_support_step, 'qualifier support formation')
        write_own_support(ui, start, gallery, support_ids, ui.support_changes)
        report['support_changes'] = ui.support_changes

    def rental_pick(plan, attr, main_attr):
        start = ui.need(_support_step, 'qualifier support formation before rental')
        own = read_own_support(ui, start, gallery, stats=True)
        start = ui.need(_support_step, 'qualifier support formation before rental')
        rent_support(ui, _open_rental(ui, start), attr, gallery=gallery, scorer=score, own=own)
        report['rental'] = ui.rental_receipt
        return True

    steps = hiro_module('produce.formation_steps')
    with ExitStack() as scope:
        if support_ids:
            scope.enter_context(_replace(steps, 'autofix_support', write_support))
        if score is not None:
            scope.enter_context(_replace(steps, 'rental_pick', rental_pick))
        yield


@contextmanager
def product_formation(rt, segment, context, device, settings, report):
    """Keep Hiro's formation entry; use Maa's existing selector if its art is absent."""
    opening = hiro_module('runtime.opening')
    assets = hiro_module('vision.assets').source()
    source = getattr(assets, 'game_data', lambda _: None)

    def maa_idol_selection(runtime, _skin):
        blocked = hiro_module('runtime.loop').ResumeBlocked
        runtime.drive(lambda frame: True if opening.nav.find(frame, 'Produce.TextStepIndicator1') is not None
                      else None, opening.STEP_TIMEOUT, 'wait for the idol selection step')
        device.authorise()
        job = context.run_task('ProduceChooseIdolFlag', pipeline_override={
            'ProduceChooseSupport': {'recognition': 'DirectHit', 'action': 'DoNothing', 'next': []},
        })
        if not job or job.status.succeeded is not True:
            raise blocked('the Maa idol selection did not reach the support step')

    with ExitStack() as scope:
        if segment != 'main' and source('idol_cards') is None:
            scope.enter_context(_replace(opening, 'step1', maa_idol_selection))
        if segment == 'qualify':
            scope.enter_context(qualifier_formation(rt, settings, context, device, report))
        yield


def _selected_skin(context, master):
    custom = context.get_node_data('ProduceChooseIdol')['recognition']['param']['custom_recognition_param']
    choose = json.loads(custom) if isinstance(custom, str) else custom
    rows = [SimpleNamespace(text=choose[key]) for key in ('idol_name', 'song_name')]
    tables = {key: master.rows(key) for key in ('Character', 'IdolCard')}
    idol = cultivation_idol(rows, tables)
    skin = 'i_card-skin-' + idol.removeprefix('i_card-')
    if not any(row['id'] == skin and row['idolCardId'] == idol for row in master.rows('IdolCardSkin')):
        raise ValueError('selected idol has no base skin in the pinned master')
    return skin, idol


def run_hiro_product(context, argv, *, campaign, log):
    from .bootstrap import _endpoint, _product_paths, _instance_endpoint
    from .ownership import write_hif_record, locked_hif_record
    from .battle_policy import FAST_PROFILE, STRONG_PROFILE, UINET21_PROFILE, selected_profile, preflight_strong_provider

    hiro_route_file(Path('assets/pack/ui/templates.json'))
    node = context.get_node_data('ProduceCultivationScenarioSettings') or {}
    produce = (node.get('attach') or {}).get('produce')
    settings = context.get_node_data('ProduceCultivationBattleSettings') or {}
    algorithm = (settings.get('attach') or {}).get('algorithm', 'uinet-2.1')
    if campaign:
        produce = 'produce-007'
        mode, difficulty = 'qualify', None
    else:
        options = (('hajime', hiro_module('runtime.hajime').MODES),
                   ('nia', hiro_module('runtime.nia_sample').MODES))
        mode, difficulty = next((name, level) for name, modes in options
                                for level, value in modes.items() if value == produce)
    _, tables = load_native_tower()
    master = tables.master()
    skin, idol = _selected_skin(context, master)
    profile = None
    if campaign:
        config = (context.get_node_data('ProduceHiroHifFinalsSettings') or {}).get('attach') or {}
        profile = selected_profile(config.get('battle_policy', 'uinet-2.1'))
        if profile in (FAST_PROFILE, STRONG_PROFILE):
            row, = (row for row in master.rows('IdolCard') if row['id'] == idol)
            if row['planType'] != 'ProducePlanType_Plan1':
                raise ValueError('the existing HIF fast and strong finals profiles require a Sense idol')
        if profile == STRONG_PROFILE:
            preflight_strong_provider()
        elif profile == FAST_PROFILE:
            from live_decision.hif_gkms_runtime import require_resources

            require_resources()
    prices = {row['id']: row['actionPointQuantity'] for row in master.rows('Produce')}
    costs = {mode: prices[produce]}
    if campaign:
        costs['main'] = prices['produce-008']
    formation_node = 'ProduceHifQualifierPreparationSettings' if campaign else 'ProduceHiroFormationSettings'
    settings = context.get_node_data(formation_node)['attach']
    formation = {key: int(settings[key]) for key in ('support_set', 'memory_set')}
    formation.update(rental_attr=settings['attribute'], main_attr=settings['attribute'],
        note_boost=bool((context.get_node_data('ProduceUseNote') or {}).get('enabled', False)),
        pt_boost=bool((context.get_node_data('ProduceUsePt') or {}).get('enabled', False)))
    refill = 'ProduceLackAPRestore' in maa_next_nodes(context, 'ProduceLackAP')

    path, runtime, maa = _product_paths()
    task = argv.task_detail
    if (task.entry != 'Produce' or not task.status.running or type(task.task_id) is not int
            or argv.node_name != 'ProduceHiroCampaignEntry' or argv.custom_action_name != 'ProduceRunHiroCampaign'):
        raise ValueError('Hiro cultivation requires the current native Produce task')
    endpoint = _instance_endpoint(_endpoint(context, expected_maa=maa))
    game = endpoint['game']
    run_id = f"hiro-{game['pid']}-{game['created_at_100ns']}-{task.task_id}"
    report = dict(run_id=run_id, task_id=task.task_id, produce=produce, idol=idol, status='running',
                  starts={}, exams=[], events=[], inputs=[])
    record_file = runtime / f'result-{run_id}.json'

    def report_event(data):
        if data['event'] == 'hiro_exam_deck':
            # Count the rows actually written before admission, independently
            # of Hiro's sequence IDs for frames and nested observations.
            data = dict(data, through_event=sum(row['segment'] == data['segment'] for row in report['events']))
        log(data)
        if data['event'] in ('cultivation_exam_result', 'finals_round_result',
                             'hiro_hif_selected_qualifier_memory', 'hiro_exam_deck', 'hiro_formal_hif_result'):
            report['exams'].append(data)
        elif data['event'] == 'hiro_cultivation_events':
            report['events'].append(data)
        elif data['event'] in ('hiro_maa_input', 'idol_road_maa_input'):
            report['inputs'].append(data)

    def authorize():
        current = context.get_task_job()
        if context.tasker.stopping or current.job_id != task.task_id or not current.status.running:
            raise ValueError('the active Maa Produce task changed')
        if _instance_endpoint(_endpoint(context, expected_maa=maa)) != endpoint:
            raise ValueError('the active Maa input endpoint changed')

    def quote_receipt(starts):
        report['starts'] = starts
        write_hif_record(record_file, report)

    with locked_hif_record(path):
        if campaign and path.is_file():
            active = json.loads(path.read_bytes())
            if (active['ownership']['owner'] == 'task275' or active['transition']['pending'] is not None
                    or (active.get('ledger') or {}).get('pending') is not None):
                raise ValueError('the existing HIF input receipt is unresolved')
        write_hif_record(record_file, report, create=True)
        try:
            authorize()
            if context.tasker.controller.set_screenshot_target_long_side(1280) is not True:
                raise ValueError('the Maa Produce viewport could not be configured')
            device = HiroMaaDevice(context, authorize, log=report_event)
            opening = HiroOpening(costs=costs, persist=quote_receipt, refill=refill)

            def opening_submitted(segment):
                return segment in opening.starts

            def enter(rt, segment=mode, memory=()):
                def setup(runtime):
                    return product_formation(runtime, segment, context, device, settings, report)

                opening.enter(rt, segment, skin=skin, difficulty=difficulty, memory=memory, formation=formation,
                              setup=setup)

            def ocr(image):
                return _read_ocr(context, image)

            if campaign:
                report['finals_policy_profile'] = profile
                finals = None
                if profile != UINET21_PROFILE:
                    # The parent holds the shared route lock. This scoped
                    # battle session retains its own receipts under this run
                    # instead of overwriting the legacy cultivation record.
                    finals = play_formal_hif_finals(context, path=record_file.with_suffix('.battle.json'),
                        runtime=runtime, run_id=run_id, game=game, profile=profile,
                        authorize=authorize, log=report_event)
                play = play_hif_exam(context, device, idol, log=report_event, play_finals=finals)
                code = run_hiro_hif_campaign(device, ocr, skin=skin, enter=enter, play_exam=play, log=report_event,
                    main_attr=formation['main_attr'], save_frame=_capture_sink(report_event),
                    opening_submitted=opening_submitted)
                report['status'] = 'settled' if code == 0 else 'stopped'
            else:
                battle = play_cultivation_exam(context, CultivationEntry(produce, idol, None), algorithm,
                                               log=report_event)
                failure = 'ProduceNIAFailedFlag' if mode == 'nia' else 'ProduceFailedFlag'
                retry = 'ProduceFailedStop' not in maa_next_nodes(context, failure)
                run_hiro_cultivation(device, ocr, mode=mode, difficulty=difficulty, skin=skin, enter=enter,
                    play_exam=battle, log=report_event, save_frame=_capture_sink(report_event), retry=retry,
                    opening_submitted=lambda: opening_submitted(mode),
                    fallback=nia_fallback(context, authorize) if mode == 'nia' else None)
                report['status'] = 'settled'
        except Exception as error:
            report.update(status='stopped', reason=str(error))
            raise
        finally:
            write_hif_record(record_file, report)
    return report
