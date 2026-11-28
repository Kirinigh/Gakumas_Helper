"""Pinned Hiro HIF transitions over Maa's active Produce task."""

import sys
from types import SimpleNamespace
from pathlib import Path

from agent.live_decision.ocr_nodes import read_ocr_node
from agent.live_decision.hiro_runtime import hiro_module, hiro_route_file, hiro_hif_campaign
from agent.live_decision.hiro_maa_device import HiroMaaDevice, hiro_maa_route

from .hiro_cultivation import _replace, _hiro_diagnostics, run_hiro_cultivation
from .qualifier_result import qualifier_certificate


def _read_ocr(context, image):
    name = 'ProduceHifCurrentIntervalOCR'
    h, w = image.shape[:2]
    return read_ocr_node(context, image, name, override={'recognition': 'OCR',
                        'roi': [0, 0, w, h], 'expected': '.+'}, all_results=True).rows


def _read_hiro_memory(campaign, device):
    tools = hiro_module('tool_device')
    factory, system = tools.open_tool_device, campaign.sys
    tools.open_tool_device = lambda: device
    campaign.sys = SimpleNamespace(path=sys.path[:])
    try:
        return campaign.read_memory()
    finally:
        tools.open_tool_device, campaign.sys = factory, system


def run_hiro_hif_campaign(device, read_ocr, *, skin, enter, play_exam, log,
                          main_attr=None, standalone_memory=None, save_frame=None, opening_submitted=None,
                          reward_policy='hiro', resume_qualifier=False):
    """Run Hiro's qualifier → confirmed memory → finals campaign in Maa.

    enter receives the current runtime, segment and Hiro's verified memory
    expectation. It owns the native, once-only AP submission for the segment.
    """
    from contextlib import ExitStack

    campaign = hiro_hif_campaign()
    loop = hiro_module('runtime.loop')
    device_errors = hiro_module('devices.base')

    def segment(mode, trace, common, extra):
        memory = tuple(extra[1].split(',')) if extra else ()
        try:
            setup = None if resume_qualifier and mode == 'qualify' else lambda rt: enter(rt, mode, memory)
            run_hiro_cultivation(device, read_ocr, mode=mode, skin=skin, main_attr=main_attr,
                enter=setup, play_exam=play_exam, log=log, save_frame=save_frame,
                opening_submitted=(lambda: opening_submitted(mode)) if opening_submitted is not None else None,
                reward_policy=reward_policy, resume_saved=setup is None)
        except (loop.ResumeBlocked, device_errors.DeviceError) as error:
            log(dict(event='hiro_hif_segment_stopped', mode=mode, reason=str(error)))
            return 3
        return 0

    def probe(kind):
        if kind != 'memory':
            raise ValueError('unexpected Hiro campaign probe')
        with hiro_maa_route(device, read_ocr), _hiro_diagnostics(log):
            memory = _read_hiro_memory(campaign, device)
        log(dict(event='hiro_hif_selected_qualifier_memory', memory=memory))
        if not memory:
            log(dict(event='hiro_hif_segment_stopped', mode='main', reason='qualifier_memory_unread'))
        return {'memory': memory}

    args = ['--idol', skin, '--no-settings-check']
    if main_attr:
        args.extend(('--main-attr', main_attr))
    if standalone_memory is not None:
        args.extend(('--main-only', '--memory', ','.join(map(str, standalone_memory))))
    with hiro_maa_route(device, read_ocr), ExitStack() as scope:
        # Replace the command-line process boundaries only. All game input is
        # mediated by the same Maa task and the original campaign decides when
        # the second segment may start.
        scope.enter_context(_replace(campaign, 'subprocess', SimpleNamespace(call=lambda *a, **kw: 0)))
        scope.enter_context(_replace(campaign, 'run', segment))
        scope.enter_context(_replace(campaign, '_probe', probe))
        scope.enter_context(_replace(campaign, 'print',
            lambda message, **_: log(dict(event='hiro_hif_campaign_message', message=message))))
        # Hiro's public CLI also applies its own close-game setting at exit.
        # Maa owns the game window, so run only the cultivation orchestration.
        return campaign._main(args)


def finish_hiro_qualifier(ui, frame):
    """Use Hiro to claim the certificate and memory and read the new selection."""
    from .qualifier_result import finish_qualifier

    if not qualifier_certificate(frame):
        raise ValueError('qualifier settlement was not identified')
    try:
        hiro_route_file(Path('assets/pack/ui/templates.json'))
        campaign = hiro_hif_campaign()
    except (ImportError, FileNotFoundError):
        return finish_qualifier(ui, frame)
    device = HiroMaaDevice(ui.context, ui.authorised, log=ui.attempts.append)
    finish = hiro_module('runtime.finish')
    nav = hiro_module('runtime.nav')
    opening = hiro_module('runtime.opening')
    with hiro_maa_route(device, lambda image: _read_ocr(ui.context, image), clock=ui.clock, sleep=ui.sleep) as rt:
        finish.finish_produce_end(rt)
        current = rt.screen.frame
        if nav.at_home(current):
            return dict(status='settled', boundary='home')
        if not opening.on_start_confirm(current):
            raise ValueError('Hiro qualifier settlement did not reach the HIF main confirmation')
        memory = _read_hiro_memory(campaign, device)
        result = dict(status='settled', boundary='next_start')
        if memory is not None:
            result['selected_memory'] = dict(zip(('Vo', 'Da', 'Vi'), memory, strict=True))
        return result
