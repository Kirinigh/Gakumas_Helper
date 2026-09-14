"""Maa custom action for local capture configuration and quiet quota logging."""

import json

from utils import logger
from maa.context import Context
from utils.local_data import LocalDataError, configure_runtime
from maa.custom_action import CustomAction
from maa.agent.agent_server import AgentServer


def _notify_failure(context: Context, status: str) -> None:
    focus_content = status.replace("\\", "＼").replace("/", "／")
    context.run_action(
        "LocalDataStatusNotification",
        pipeline_override={
            "LocalDataStatusNotification": {
                "focus": {
                    "Node.ActionNode.Succeeded": {
                        "content": focus_content,
                        "display": ["log", "notification"],
                    }
                }
            }
        },
    )


@AgentServer.custom_action("LocalDataConfigure")
class LocalDataConfigure(CustomAction):
    """Apply safe UI settings, log quota usage, and notify only failures."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = json.loads(argv.custom_action_param or "{}")
            store, _report, status = configure_runtime(params)
        except (json.JSONDecodeError, LocalDataError, OSError, TypeError, ValueError) as error:
            status = f"本地采集已停用；配置或清理失败；未写入新图像：{error}"
            logger.error(status)
            _notify_failure(context, status)
            return True

        logger.bind(ui_visible=False).info(status)
        if store.config.enabled:
            mode_labels = {
                "failure": "仅失败现场",
                "roi": "局部图像与失败现场",
                "screenshot": "截图、局部图像与失败现场",
            }
            logger.info(f"本地画面保存已开启：{mode_labels[store.config.mode]}；仅保存在本机。")
        return True
