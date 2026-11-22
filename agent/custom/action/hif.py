"""Disabled-by-default HIF bridge; runtime hooks are supplied by the host."""

import json
from dataclasses import replace

from utils import logger
from route.hif import HifConfig, training_main_attribute
from route.host import HifRouteHost
from route.maa_bridge import HifRuntimeHooks, run_hif_sequence
from maa.custom_action import CustomAction
from maa.agent.agent_server import AgentServer
from live_decision.hif_recovery_reports import recovery_report_scope

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
            if cultivation is not None:
                from route.cultivation import HifCultivationRuntime
                if not isinstance(cultivation, HifCultivationRuntime) or cultivation.bound is not context:
                    raise ValueError('conflicting trusted HIF cultivation runtime')
                decision = cultivation.run(config)
                context.gkh_hif_cultivation_report = cultivation.report
            else:
                decision = host.run(context, config) if host is not None else run_hif_sequence(context, config, hooks)
            context.gkh_hif_route_decision = decision
            logger.info(STATUS_TEXT.get(decision.reason, "HIF 自动推进已停止，请核对当前画面和状态。"))
            logger.info(f"HIF: {decision.outcome.value}: {decision.reason}")
        except Exception as error:
            producer = getattr(context, 'gkh_hif_route_producer', None)
            if producer is not None:
                producer.events.append(dict(phase='route_failure', type=type(error).__name__, reason=str(error)))
            logger.error(STATUS_TEXT.get(str(error), f"HIF stopped: {type(error).__name__}: {error}"))
        finally:
            if binding is not None:
                binding.close()
        # The bounded sequence ends at a boundary, failure or budget. It always
        # yields to StopTask rather than generic Produce fallbacks or retries.
        return False
