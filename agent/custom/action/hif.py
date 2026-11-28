"""Disabled-by-default HIF bridge; runtime hooks are supplied by the host."""

import json
from dataclasses import replace

from utils import logger, log_diagnostic, diagnostic_only
from route.hif import HifConfig, training_main_attribute
from route.host import HifRouteHost
from route.maa_bridge import HifRuntimeHooks, run_hif_sequence
from maa.custom_action import CustomAction
from maa.agent.agent_server import AgentServer
from live_decision.hif_recovery_reports import recovery_report_scope

from .produce import _report_cultivation, _report_cultivation_stop

STATUS_TEXT = {
    "HIF explicit battle policy differs from its retained cultivation": "HIF 已停止：当前培育已固定战斗策略，请选择继承存档或原策略；新策略从新的授权培育生效。",
    "hif_disabled": "HIF 自动推进已关闭。",
    "hif_runtime_observer_not_bound": "HIF 已暂停：尚未接入本局身份、稳定画面和授权观测。",
    "hif_transition_store_not_bound": "HIF 已暂停：尚未接入待确认动作的保存与恢复。",
    "transition_checkpoint_failed": "HIF 已暂停：动作状态保存失败，保留待确认结果，不重复操作。",
    "capture_not_authorised": "HIF 已暂停：尚未取得本轮画面读取授权。",
    "input_not_authorised_or_endpoint_unverified": "HIF 已暂停：尚未取得本轮输入授权或确认实际输入端权限。",
    "battle_requires_task275": "HIF 已暂停：已到战斗入口，等待公共出牌能力接入。",
    "sequence_action_budget_reached": "HIF 已暂停：已达到本轮自动推进动作上限。",
    "sequence_timeout": "HIF 已暂停：已达到本轮自动推进时间上限。",
}


@AgentServer.custom_action("ProduceHifAdvance")
class ProduceHifAdvance(CustomAction):
    @recovery_report_scope
    def run(self, context, argv):
        binding = None
        try:
            config = HifConfig.from_mapping(json.loads(argv.custom_action_param or "{}"))
            if config.mode == 'advance':
                config = replace(config, main_attribute=training_main_attribute(context) or config.main_attribute)
            if not any(getattr(context, name, None) is not None for name in (
                    'gkh_hif_runtime_hooks', 'gkh_hif_route_host', 'gkh_hif_cultivation_runtime')):
                from route.bootstrap import bind_hif_product_resume

                binding = bind_hif_product_resume(context, argv, config)
                if binding is not None:
                    context = binding.context
            # The host may bind runtime hooks to this context after separate
            # capture/input authorisation. JSON parameters cannot bind them.
            hooks = getattr(context, "gkh_hif_runtime_hooks", None)
            host = getattr(context, "gkh_hif_route_host", None)
            if host is not None:
                if not isinstance(host, HifRouteHost) or (hooks is not None and hooks is not host.hooks):
                    raise ValueError("conflicting HIF runtime host binding")
                hooks = host.hooks
            if hooks is not None and not isinstance(hooks, HifRuntimeHooks):
                raise ValueError("invalid runtime hook binding")
            cultivation = getattr(context, 'gkh_hif_cultivation_runtime', None)
            with diagnostic_only():
                if cultivation is not None:
                    from route.cultivation import HifCultivationRuntime
                    if not isinstance(cultivation, HifCultivationRuntime) or cultivation.bound is not context:
                        raise ValueError('conflicting trusted HIF cultivation runtime')
                    decision = cultivation.run(config)
                    context.gkh_hif_cultivation_report = cultivation.report
                else:
                    decision = host.run(context, config) if host is not None else run_hif_sequence(context, config, hooks)
            context.gkh_hif_route_decision = decision
            log_diagnostic(f"HIF: {decision.outcome.value}: {decision.reason}")
            if not getattr(getattr(context, 'tasker', None), 'stopping', False):
                if cultivation is not None:
                    for outcome in cultivation.report.get('battle_outcomes', ()):
                        if outcome['status'] == 'TERMINAL_VERIFIED' and outcome['returned_to_nonbattle']:
                            terminal = outcome['terminal_settlement']
                            _report_cultivation(dict(event='hiro_formal_hif_result', status=outcome['status'],
                                round_number=terminal['round_number'], score=terminal['score']))
                logger.info(STATUS_TEXT.get(decision.reason, "HIF 自动推进已停止，请核对当前画面和状态。"))
        except Exception as error:
            producer = getattr(context, 'gkh_hif_route_producer', None)
            if producer is not None:
                producer.events.append(dict(phase='route_failure', type=type(error).__name__, reason=str(error)))
            _report_cultivation_stop(context, 'HIF 自动培育', error, message=STATUS_TEXT.get(str(error)))
        finally:
            if binding is not None:
                binding.close()
        # The bounded sequence ends at a boundary, failure or budget. It always
        # yields to StopTask rather than generic Produce fallbacks or retries.
        return False


@AgentServer.custom_action("ProduceHifQualifierPrepare")
class ProduceHifQualifierPrepare(CustomAction):
    def run(self, context, argv):
        from route.qualifier import run_native_qualifier

        try:
            options = context.get_node_data('ProduceHifQualifierPreparationSettings')['attach']
            logger.info('HIF 选拔编成开始。')
            with diagnostic_only():
                result = run_native_qualifier(context, argv, options)
            for exam in result.get('exams', ()):
                _report_cultivation(dict(exam, event='cultivation_exam_result', label=f'qualify-{exam["exam"]}'))
            if result['status'] == 'settled':
                logger.info("HIF 选拔已完成。")
            elif result['status'] == 'prepared':
                logger.info("HIF 选拔编成已准备，请确认开始。")
            else:
                _report_cultivation_stop(context, 'HIF 选拔', result['status'])
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            _report_cultivation_stop(context, 'HIF 选拔准备', error)
        return False


@AgentServer.custom_action("ProduceRunHiroCultivation")
class ProduceRunHiroCultivation(CustomAction):
    def run(self, context, argv):
        from pathlib import Path

        from route.bootstrap import _endpoint, _product_paths, _instance_endpoint

        from agent.route.hiro_hif import _read_ocr
        from agent.route.hiro_exam import play_cultivation_exam
        from agent.route.hiro_cultivation import (
            nia_fallback,
            _capture_sink,
            maa_next_nodes,
            run_hiro_cultivation,
            selected_reward_policy,
        )
        from agent.live_decision.hiro_runtime import hiro_module, hiro_route_file
        from agent.live_decision.hiro_maa_device import HiroMaaDevice

        from .produce import _cultivation_option, _cultivation_entries

        try:
            task, settings = _cultivation_option(context, argv)
            try:
                hiro_route_file(Path('assets/pack/ui/templates.json'))
            except (FileNotFoundError, ImportError):
                # Installed packages may carry only the pinned battle module.
                # Keep their existing Maa route and opening receipt.
                return True
            entry = _cultivation_entries[task]
            algorithm = settings['algorithm']
            hajime = hiro_module('runtime.hajime').MODES
            nia = hiro_module('runtime.nia_sample').MODES
            mode, difficulty = next((mode, difficulty) for mode, choices in (('hajime', hajime), ('nia', nia))
                for difficulty, produce in choices.items() if produce == entry.produce)
            if mode == 'nia':
                mirror = context.get_node_data('ProduceMirrorFlag') or {}
                attach = mirror.get('attach') or {}
                if any(attach.get(f'focus_{stage}') for stage in (1, 2, 3)):
                    # Retain the player's explicit manual audition choice.
                    return True
            failure = 'ProduceNIAFailedFlag' if mode == 'nia' else 'ProduceFailedFlag'
            retry = 'ProduceFailedStop' not in maa_next_nodes(context, failure)
            _cultivation_entries.pop(task)
            maa = _product_paths()[2]
            endpoint = _instance_endpoint(_endpoint(context, expected_maa=maa))

            def authorize():
                current = context.get_task_job()
                if current.job_id != task or not current.status.running or context.tasker.stopping:
                    raise ValueError('current Maa Produce task is no longer running')
                if _instance_endpoint(_endpoint(context, expected_maa=maa)) != endpoint:
                    raise ValueError('current Maa Produce input endpoint changed')

            logger.info(f'{"初" if mode == "hajime" else "NIA"}自动培育开始。')
            device = HiroMaaDevice(context, authorize, log=_report_cultivation)
            run_hiro_cultivation(device, lambda image: _read_ocr(context, image),
                mode=mode, difficulty=difficulty, skin=entry.idol,
                play_exam=play_cultivation_exam(context, entry, algorithm, log=_report_cultivation),
                log=_report_cultivation, fallback=nia_fallback(context, authorize) if mode == 'nia' else None,
                save_frame=_capture_sink(_report_cultivation), retry=retry, reward_policy=selected_reward_policy(context))
            _report_cultivation(dict(event='hiro_cultivation_settled', produce=entry.produce))
            return False  # StopTask owns both settlement and interrupted routes.
        except (ImportError, KeyError, OSError, RuntimeError, StopIteration, TypeError, ValueError) as error:
            _report_cultivation_stop(context, '自动培育', error)
            return False


@AgentServer.custom_action("ProduceRunHiroCampaign")
class ProduceRunHiroCampaign(CustomAction):
    def run(self, context, argv):
        stop_reason = None

        def report(data):
            nonlocal stop_reason
            if data['event'] == 'hiro_hif_segment_stopped':
                stop_reason = data.get('reason')
            _report_cultivation(data)

        try:
            from agent.route.hiro_product import run_hiro_product

            params = argv.custom_action_param
            params = json.loads(params) if isinstance(params, str) else params
            settings = params or {}
            phase = 'HIF 自动培育' if settings.get('hif') else '自动培育'
            logger.info(f'{phase}开始。')
            result = run_hiro_product(context, argv, campaign=bool(settings.get('hif')),
                                      hif_segment=settings.get('hif_segment'), log=report)
            if result['status'] == 'settled':
                logger.info(f'{phase}已完成。')
            else:
                _report_cultivation_stop(context, phase, stop_reason or result.get('reason') or result['status'])
        except (ImportError, KeyError, OSError, RuntimeError, StopIteration, TypeError, ValueError) as error:
            _report_cultivation_stop(context, '自动培育', error)
        return False
