import json
import time
import random
from typing import Any, Dict, List, Optional
from collections import Counter

from utils import logger
from maa.context import Context
from maa.custom_action import CustomAction
from maa.agent.agent_server import AgentServer

from agent.live_decision.ocr_nodes import read_ocr_node

_cultivation_entries = {}


def _cultivation_option(context, argv):
    task = getattr(argv, 'task_detail', None)
    if getattr(task, 'entry', None) != 'Produce':
        return None, {}
    node = context.get_node_data('ProduceCultivationBattleSettings') or {}
    scenario = context.get_node_data('ProduceCultivationScenarioSettings') or {}
    return task.task_id, {**(node.get('attach') or {}), **(scenario.get('attach') or {})}


@AgentServer.custom_action("ProduceBindCultivationExam")
class ProduceBindCultivationExam(CustomAction):
    def run(self, context, argv):
        task, settings = _cultivation_option(context, argv)
        algorithm = settings.get('algorithm')
        if task is not None and algorithm in ('uinet-1.1', 'uinet-2.1'):
            from live_decision.inventory import _inventory_ocr

            from agent.live_decision.cultivation_maa_exam import (
                _stamina,
                _parameters,
                cultivation_master,
                read_cultivation_entry,
                read_cultivation_briefing,
            )

            try:
                image = context.tasker.controller.post_screencap().wait().get()
                rows = tuple(_inventory_ocr(context, image, 'ProduceRecognitionScore'))
                if argv.node_name == 'ProduceLauncher':
                    _cultivation_entries.pop(task, None)
                    _cultivation_entries[task] = read_cultivation_entry(rows, settings['produce'], cultivation_master())
                elif (entry := _cultivation_entries.get(task)) is not None:
                    phase, number = read_cultivation_briefing(rows)
                    if phase:
                        if entry.round_key != phase:
                            entry.audition_number = None
                        entry.round_key = phase
                        entry.audition_number = number if number is not None else entry.audition_number
                        entry.parameters = _parameters(rows) or entry.parameters
                        entry.max_stamina = _stamina(rows) or entry.max_stamina
            except (KeyError, TypeError, ValueError, RuntimeError, OSError, ImportError) as error:
                logger.error(f'培育考试开局资料读取停止 {error}')
                return False
        click_params = {'target_offset': [0, 0, 0, -10]} if argv.node_name in (
            'ProduceButton', 'ProduceNIAButton') else {}
        original = {'action': {'type': 'Click', 'param': click_params}}
        action = context.run_action(argv.node_name, box=argv.box, pipeline_override={argv.node_name: original})
        return bool(action and action.success)


def _ocr_score_pair(context, image, roi):
    return read_ocr_node(context, image, "ProduceRecognitionScore", override={"roi": roi},
                         accept=lambda rows: len(rows) == 2).rows


class ProduceChooseEventBase(CustomAction):
    """
    培育事件选择基类。

    子类需定义以下类属性：
    - SUGGESTION_CONFIG: 老师建议匹配配置
    - EVENT_CONFIG: 事件图片配置
    - RUN_TASK_MAP: 需要执行run_task的事件映射

    可覆盖的钩子方法：
    - _get_preference: 获取属性偏好设置
    - _choose_extra_before_attrs: SP之后、属性课程之前的额外优先事件
    - _choose_extra_after_outing: 外出之后的额外优先事件
    """

    # 子类需覆盖
    SUGGESTION_CONFIG: dict = {}
    EVENT_CONFIG: dict = {}
    RUN_TASK_MAP: dict = {}
    SUGGESTION_ROI: list = [270, 160, 350, 80]
    EVENT_ICON_OFFSET = (0, 0)
    REST_COUNT_ROI = [580, 755, 135, 75]

    # 阈值常量
    CLICK_DELAY = 0.5
    ACTION_DELAY = 3.0
    PREFERENCE_LIST = ["Da", "Vi", "Vo"]
    FIRST_NEAR_FULL_RATIO = 0.85
    ATTR_STOP_RATIO = 0.8
    LOW_HEALTH_RATIO = 0.2
    LOW_HEALTH_VALUE = 8

    def __init__(self):
        super().__init__()
        self.first = "Vi"
        self.second = "Da"

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        logger.success("事件: 选择事件")

        preference = self._get_preference(context, argv)
        self.first = preference["first"]
        self.second = preference["second"]
        logger.info(f"第一属性: {self.first}, 第二属性: {self.second}")

        image = self._get_screenshot(context)

        suggestion = ""
        if context.get_node_data("ProduceSuggestion").get("enabled", True):
            suggestion = self._get_suggestion(context, image)

        observed_health = self._get_health(context, image)
        health_data = observed_health or {"current": 34, "max": 34, "ratio": 1.0}
        points = self._get_current_points(context, image) or 0
        score = self._get_current_score(context, image) or {"Vo": 0, "Da": 0, "Vi": 0, "max": 1}
        events = self._get_available_events(context, image)

        best_event = self._choose_best_event(suggestion, health_data, points, score, events)
        if not best_event:
            logger.info("无可用事件")
            return True

        # 休息前检查：次数用完（检测到"あと0回"）时不可休息，改选其他事件
        if best_event.get("run_task") == "ProduceChooseRest" and not self._is_rest_available(context, image):
            logger.warning("休息次数已用完，改选其他事件")
            best_event = self._choose_best_event(suggestion, health_data, points, score, events, allow_rest=False)
            if not best_event:
                logger.info("无可用事件")
                return True

        # 老师建议和保底分支也保留实际课程的 SP 标记，不改变选择优先级。
        name = best_event["name"]
        if name in ("Vo", "Da", "Vi") and any(event.get("SP") and event.get(name) == best_event["box"] for event in events):
            best_event["name"] = name + "_SP"
        box = best_event["box"]
        logger.info(f"选择事件: {best_event['name']}, 坐标: ({box[0] + box[2] // 2}, {box[1] + box[3] // 2})")

        task, _ = _cultivation_option(context, argv)
        if (entry := _cultivation_entries.get(task)) is not None:
            values = tuple(score[key] for key in ('Vo', 'Da', 'Vi'))
            if all(value > 0 for value in values):
                entry.parameters = values
            if observed_health is not None and observed_health['max'] > 0:
                entry.max_stamina = observed_health['max']
            label = best_event['name']
            entry.lesson_kind = ('sp' if label.endswith('_SP') else 'normal'
                                 if label in ('Vo', 'Da', 'Vi') else None)

        return self._execute_event(context, best_event)

    def _get_preference(self, context: Context, argv: CustomAction.RunArg) -> dict:
        """获取偏好设置，默认从argv解析。"""
        try:
            params = json.loads(argv.custom_action_param)
            first = params.get("first") if isinstance(params, dict) else None
            second = params.get("second") if isinstance(params, dict) else None
            return {"first": first if first in self.PREFERENCE_LIST else "Vi", "second": second if second in self.PREFERENCE_LIST else "Da"}
        except (json.JSONDecodeError, AttributeError):
            logger.warning("偏好设置解析失败，使用默认值")
            return {"first": "Vi", "second": "Da"}

    def _is_rest_available(self, context: Context, image) -> bool:
        """检测休息是否可用：休息按钮剩余次数区域出现「あと0回」时不可休息。"""
        reco_detail = context.run_recognition(
            "ProduceRecognitionRestCount",
            image,
            pipeline_override={
                "ProduceRecognitionRestCount": {
                    "recognition": "OCR",
                    "roi": self.REST_COUNT_ROI,
                    "expected": ["あと[0０]回", "剩余[0０]次"],  # 兼容全角/半角 0
                }
            },
        )
        if reco_detail and reco_detail.hit:
            logger.info("休息次数已用完")
            return False
        return True

    def _choose_best_event(
        self, suggestion: str, health_data: dict, points: int, score: dict, events: list, allow_rest: bool = True
    ) -> Optional[dict]:
        """
        根据获取到的信息从可用事件中选择最佳事件。

        Returns:
            dict: {"name": str, "box": [x, y, w, h], "run_task": str}，None表示无事件可选
        """
        max_score = score.get("max", 1)
        first_ratio = score.get(self.first, 0) / max_score if max_score > 0 else 0
        second_ratio = score.get(self.second, 0) / max_score if max_score > 0 else 0
        is_first_near_full = first_ratio >= self.FIRST_NEAR_FULL_RATIO
        is_first_stopped = first_ratio >= self.ATTR_STOP_RATIO
        is_second_stopped = second_ratio >= self.ATTR_STOP_RATIO

        # 0. 低体力处理
        current_health = health_data["current"] if health_data else 0
        ratio_health = health_data["ratio"] if health_data else 1.0
        if current_health < self.LOW_HEALTH_VALUE or ratio_health < self.LOW_HEALTH_RATIO:
            go_out = self._find_event_by_name(events, "外出")
            if go_out and points >= 100:
                return self._make_event("外出", go_out)
            if allow_rest:
                logger.info("体力过低，选择休息")
                return {"name": "rest", "box": [0, 0, 0, 0], "run_task": "ProduceChooseRest"}
            # 休息不可用（次数用完），继续走后续优先级

        # 1. 老师建议
        suggestion_attr = self._parse_suggestion(suggestion)
        if suggestion_attr:
            event = self._find_attr_event(events, suggestion_attr)
            if event:
                return self._make_event(suggestion_attr, event)

        # 2. SP（第一属性，仅当属性未达80%时）
        if not is_first_stopped:
            event = self._find_attr_event(events, self.first, need_sp=True)
            if event:
                return self._make_event(f"{self.first}_SP", event)

        # 3. SP（第二属性，仅当属性未达80%时）
        if not is_second_stopped:
            event = self._find_attr_event(events, self.second, need_sp=True)
            if event:
                return self._make_event(f"{self.second}_SP", event)

        # 钩子：SP之后、属性课程之前的额外优先事件（NIA的"营业"）
        extra = self._choose_extra_before_attrs(events)
        if extra:
            return extra

        # 4. 第一属性课程（属性未达80%时）
        if not is_first_stopped:
            event = self._find_attr_event(events, self.first, need_sp=False)
            if event:
                return self._make_event(self.first, event)

        # 5. 第二属性课程（第一属性停止或快满时，且第二属性未达80%）
        if (is_first_stopped or is_first_near_full) and not is_second_stopped:
            event = self._find_attr_event(events, self.second, need_sp=False)
            if event:
                return self._make_event(self.second, event)

        # 6. 交谈/活动（优先交谈，不满足则活动）
        if points >= 100:
            event = self._find_event_by_name(events, "交谈")
            if event:
                return self._make_event("交谈", event)
        event = self._find_event_by_name(events, "活动")
        if event:
            return self._make_event("活动", event)

        # 7. 上课（部分子类无此事件，由EVENT_CONFIG控制）
        event = self._find_event_by_name(events, "上课")
        if event:
            return self._make_event("上课", event)

        # 8. 外出
        event = self._find_event_by_name(events, "外出")
        if event:
            return self._make_event("外出", event)

        # 钩子：外出之后、其他属性SP之前的额外优先事件（NIA的"指导"）
        extra = self._choose_extra_after_outing(events)
        if extra:
            return extra

        # 9. SP（其他属性）
        for attr in ["Vo", "Da", "Vi"]:
            if attr not in [self.first, self.second]:
                event = self._find_attr_event(events, attr, need_sp=True)
                if event:
                    return self._make_event(f"{attr}_SP", event)

        # 保底：从剩余事件中随机选择
        if events:
            fallback = self._find_any_event(events)
            if fallback:
                logger.warning(f"所有优先级未命中，保底选择: {fallback['name']}")
                return fallback

        return None

    def _choose_extra_before_attrs(self, events: list) -> Optional[dict]:
        """钩子：SP之后、属性课程之前的额外优先事件。子类可覆盖。"""
        return None

    def _choose_extra_after_outing(self, events: list) -> Optional[dict]:
        """钩子：外出之后的额外优先事件。子类可覆盖。"""
        return None

    def _execute_event(self, context: Context, event: dict) -> bool:
        """执行事件：双击坐标，等待动画后执行后续任务（如果有）。"""
        run_task = event.get("run_task")
        box = event["box"]

        x = box[0] + box[2] // 2
        y = box[1] + box[3] // 2
        context.tasker.controller.post_click(x, y).wait()
        time.sleep(self.CLICK_DELAY)
        context.tasker.controller.post_click(x, y).wait()
        time.sleep(self.ACTION_DELAY)
        if run_task:
            logger.info(f"执行任务{run_task}")
            context.run_task(run_task)
        return True

    def _find_attr_event(self, events: list, attr: str, need_sp: bool = False) -> Optional[list]:
        """查找指定属性事件的坐标。need_sp=True时只返回有SP标记的属性事件。"""
        for event in events:
            if attr in event:
                if need_sp and not event.get("SP"):
                    continue
                return event[attr]
        return None

    def _find_event_by_name(self, events: list, event_name: str) -> Optional[list]:
        """从事件列表中查找指定名称事件的坐标。"""
        for event in events:
            for key in event:
                if event_name in key:
                    return event[key]
        return None

    def _find_any_event(self, events: list) -> Optional[dict]:
        """从事件列表中获取任意第一个可用事件。"""
        for event in events:
            for key in event:
                if key not in ["SP"] and isinstance(event[key], list):
                    run_task = self.RUN_TASK_MAP.get(key, "")
                    return self._make_event(key, event[key], run_task=run_task)
        return None

    def _make_event(self, name: str, box: list, run_task: str = "") -> dict:
        """创建事件字典，run_task默认为空时从RUN_TASK_MAP自动获取。"""
        return {"name": name, "box": box, "run_task": run_task or self.RUN_TASK_MAP.get(name, "")}

    def _parse_suggestion(self, suggestion: str) -> Optional[str]:
        """从老师建议中解析属性名称，匹配SUGGESTION_CONFIG中的关键词。"""
        if not suggestion:
            return None
        for attr in ["Vo", "Da", "Vi"]:
            for keyword in self.SUGGESTION_CONFIG[attr].get("keyword", []):
                if keyword in suggestion:
                    return attr
        return None

    @staticmethod
    def _get_screenshot(context: Context):
        """获取屏幕截图"""
        return context.tasker.controller.post_screencap().wait().get()

    def _get_suggestion(self, context: Context, image) -> str:
        """获取建议课程"""
        reco_detail = context.run_recognition(
            "ProduceChooseEventSuggestion",
            image,
            pipeline_override={"ProduceChooseEventSuggestion": {"recognition": "OCR", "roi": self.SUGGESTION_ROI}},
        )

        if not (reco_detail and reco_detail.hit):
            return ""

        suggestion_text = "".join(item.text for item in reco_detail.filtered_results)
        logger.info(f"老师建议: {suggestion_text}")
        return suggestion_text

    @staticmethod
    def _get_health(context: Context, image) -> Optional[dict]:
        """获取体力比例"""
        found = read_ocr_node(context, image, "ProduceRecognitionHealth").best
        if not found:
            return None

        try:
            health_parts = found.text.split("/")
            current_health = int("".join(filter(str.isdigit, health_parts[0])))
            max_health = int("".join(filter(str.isdigit, health_parts[1])))
            ratio = current_health / max_health
            logger.info(f"体力: {current_health}/{max_health} ({ratio:.2%})")
            return {"current": current_health, "max": max_health, "ratio": ratio}
        except (ValueError, IndexError, ZeroDivisionError):
            logger.warning("体力数据解析失败")
            return None

    @staticmethod
    def _get_current_points(context: Context, image) -> Optional[int]:
        """获取当前积分"""
        found = read_ocr_node(context, image, "ProduceRecognitionPoint",
                              override={"roi": [320, 90, 130, 54]}).best
        if not found:
            return None
        try:
            points = int(found.text.replace(",", ""))
            logger.info(f"积分: {points}")
            return points
        except ValueError:
            logger.warning("积分数据解析失败")
            return None

    @staticmethod
    def _get_current_score(context: Context, image) -> Optional[dict]:
        """获取当前得分"""
        score = {
            "Vo": 0,
            "Da": 0,
            "Vi": 0,
            "max": 0,
        }
        for i in range(3):
            rows = _ocr_score_pair(context, image, [150 + i * 150, 668, 136, 80])
            if len(rows) == 2:
                current_score = int("".join(filter(str.isdigit, rows[0].text)))
                max_score = int("".join(filter(str.isdigit, rows[1].text)))
                logger.debug(f"第{i + 1}列得分: {current_score} / {max_score}")
                score[["Vo", "Da", "Vi"][i]] = current_score
                score["max"] = max_score if score["max"] < max_score < 9999 else score["max"]
        try:
            logger.info(f"当前得分: Vo={score['Vo']}, Da={score['Da']}, Vi={score['Vi']}, Max={score['max']}")
            return score
        except ValueError:
            logger.warning("积分数据解析失败")
            return None

    def _get_available_events(self, context: Context, image) -> List[Dict[str, Any]]:
        """逐项识别可用事件，按课程图标的位置检查 SP，兼容不同按钮数量。"""
        available_events = []

        for event_name, event_img in self.EVENT_CONFIG.items():
            reco_detail = context.run_recognition(
                "ProduceRecognitionEvent",
                image,
                pipeline_override={
                    "ProduceRecognitionEvent": {"recognition": "TemplateMatch", "template": event_img, "roi": [0, 880, 720, 220]}
                },
            )
            if reco_detail and reco_detail.hit:
                box = list(reco_detail.best_result.box)
                event = {event_name: box}
                if event_name in ["Da", "Vi", "Vo"]:
                    # 从图标本体还原按钮锚点，徽标范围同时覆盖初/NIA。
                    offset_x, offset_y = self.EVENT_ICON_OFFSET
                    sp_roi = [max(0, box[0] - offset_x - 50), max(0, box[1] - offset_y - 100), 110, 140]
                    event["SP"] = self._get_sp_course(context, image, sp_roi)
                available_events.append(event)

        logger.info("可用事件: " + ", ".join(name for event in available_events for name in event if name != "SP"))
        logger.debug(available_events)
        return available_events

    def _get_sp_course(self, context: Context, image, sp_roi: List[int]) -> bool:
        """获取SP课程选择"""
        reco_detail = context.run_recognition(
            "ProduceChooseEventSp",
            image,
            pipeline_override={"ProduceChooseEventSp": {"recognition": "TemplateMatch", "template": "produce/sp.png", "roi": sp_roi}},
        )
        if reco_detail and reco_detail.hit:
            logger.debug(f"{sp_roi}存在SP课程")
            return True
        return False


@AgentServer.custom_action("ProduceChooseEventAuto")
class ProduceChooseEventAuto(ProduceChooseEventBase):
    """
    自动选择培育事件。

    选择优先级顺序（从高到低）：
    1. 低体力保护：体力 < 20% 或体力值 < 8时，优先外出恢复体力，其次休息
    2. 老师建议：根据OCR结果匹配Vo/Da/Vi属性关键词
    3. SP课程（第一属性，仅当该属性分数占比 < 80%时）
    4. SP课程（第二属性，仅当该属性分数占比 < 80%时）
    5. 第一属性课程（仅当该属性分数占比 < 80%时）
    6. 第二属性课程（仅当第一属性分数占比 >= 85%时，且第二属性占比 < 80%）
    7. 交谈/活动：优先交谈消耗points商店购物（需points >= 100），不满足则活动获得points
    8. 上课：提升Vo/Da/Vi
    9. 外出：恢复体力
    10. SP课程（非第一/第二属性）
    保底：以上均未命中时，从剩余事件中任选一个
    """

    SUGGESTION_CONFIG = {
        "Vo": {"img": "produce/Vo.png", "keyword": ["ボーカル", "唱歌"]},
        "Da": {"img": "produce/Da.png", "keyword": ["ダンス", "舞蹈"]},
        "Vi": {"img": "produce/Vi.png", "keyword": ["ビジュアル", "视觉"]},
        "体力": {"img": "produce/rest.png", "keyword": ["体力"]},
        "交谈": {"img": "produce/chat.png", "keyword": ["先生に相談して", "咨询"]},
    }

    # 只匹配图标正文，排除渐变底色、边角及事件标签对相关系数的干扰。
    EVENT_ICON_OFFSET = (30, 20)
    EVENT_CONFIG = {
        "Vo": "produce/Vo.png",
        "Da": "produce/Da.png",
        "Vi": "produce/Vi.png",
        "交谈": "produce/chat.png",
        "上课": "produce/lesson.png",
        "活动": "produce/event.png",
        "外出": "produce/go_out.png",
    }

    RUN_TASK_MAP = {
        "交谈": "ProduceShoppingEntry",
    }

    def _get_preference(self, context: Context, argv: CustomAction.RunArg) -> dict:
        preference = context.get_node_data("ProduceChooseNIAEventFlag").get("action", {}).get("param", {}).get("custom_action_param", {})
        if "first" not in preference or "second" not in preference:
            logger.warning("偏好设置解析失败，使用默认值")
        return {"first": preference.get("first", "Vi"), "second": preference.get("second", "Da")}


@AgentServer.custom_action("ProduceChooseNIAEventAuto")
class ProduceChooseNIAEventAuto(ProduceChooseEventBase):
    """
    自动选择NIA培育事件。

    选择优先级顺序（从高到低）：
    1. 低体力保护：体力 < 20% 或体力值 < 8时，优先外出恢复体力，其次休息
    2. 老师建议：根据OCR结果匹配Vo/Da/Vi属性关键词
    3. SP课程（第一属性，仅当该属性分数占比 < 80%时）
    4. SP课程（第二属性，仅当该属性分数占比 < 80%时）
    5. 营业：获得score
    6. 第一属性课程（仅当该属性分数占比 < 80%时）
    7. 第二属性课程（仅当第一属性分数占比 >= 85%时，且第二属性占比 < 80%）
    8. 交谈/活动：优先交谈消耗points商店购物（需points >= 100），不满足则活动获得points
    9. 外出：恢复体力
    10. 指导
    11. SP课程（非第一/第二属性）
    保底：以上均未命中时，从剩余事件中任选一个
    """

    SUGGESTION_CONFIG = {
        "Vo": {"img": "produce/NIA/Vo.png", "keyword": ["ボーカル", "唱歌"]},
        "Da": {"img": "produce/NIA/Da.png", "keyword": ["ダンス", "舞蹈"]},
        "Vi": {"img": "produce/NIA/Vi.png", "keyword": ["ビジュアル", "视觉"]},
        "体力": {"img": "produce/rest.png", "keyword": ["体力"]},
        "指导": {"img": "produce/NIA/guide.png", "keyword": ["特别指導", "特别指导"]},
        "交谈": {"img": "produce/NIA/chat.png", "keyword": ["相談", "咨询"]},
    }

    EVENT_CONFIG = {
        "Vo": "produce/NIA/Vo.png",
        "Da": "produce/NIA/Da.png",
        "Vi": "produce/NIA/Vi.png",
        "交谈": "produce/NIA/chat.png",
        "活动": "produce/NIA/activity.png",
        "指导": "produce/NIA/guide.png",
        "外出": "produce/NIA/go_out.png",
        "工作": "produce/NIA/work.png",
    }

    RUN_TASK_MAP = {
        "交谈": "ProduceShoppingEntry",
        "指导": "ProduceGuideEntry",
        "工作": "ProduceWorkEntry",
    }

    def _choose_extra_before_attrs(self, events: list) -> Optional[dict]:
        """营业：获得score"""
        event = self._find_event_by_name(events, "工作")
        if event:
            return self._make_event("工作", event)
        return None

    def _choose_extra_after_outing(self, events: list) -> Optional[dict]:
        """指导（点击后需执行任务）"""
        event = self._find_event_by_name(events, "指导")
        if event:
            return self._make_event("指导", event)
        return None


@AgentServer.custom_action("ProduceSkipShowAuto")
class ProduceSkipShowAuto(CustomAction):
    """按固定坐标尝试右上角两次、左上角一次，并限制点击在当前画面内。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        for target_x, target_y in ((1192, 80), (1192, 80), (20, 20)):
            if context.tasker.stopping:
                return False
            image = context.tasker.controller.post_screencap().wait().get()
            if context.tasker.stopping:
                return False
            if image is None or image.size == 0:
                return False
            height, width = image.shape[:2]
            if height >= width:
                # 演出已结束或尚在切屏，交还流水线判断，避免点击结算页面。
                return True

            x = max(0, min(width - 1, target_x))
            y = max(0, min(height - 1, target_y))
            if not context.tasker.controller.post_click(x, y).wait().succeeded:
                return False
            time.sleep(0.3)

        return True


@AgentServer.custom_action("ProduceCardsAuto")
class ProduceCardsAuto(CustomAction):
    """
    自动识别根据系统提示出牌
    15秒未检测到提示牌，则打出最高分的牌
    识别不到体力退出函数
    处理是否打出该牌的弹窗
    """

    # 阈值常量
    CLICK_DELAY = 0.3
    TIME_OUT = 15.0

    # 卡牌Y轴位置边界（超出范围视为识别异常，不点击）
    CARD_Y_MIN = 840
    CARD_Y_MAX = 1000

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        try:
            live_params = json.loads(argv.custom_action_param or "{}")
        except (json.JSONDecodeError, TypeError):
            live_params = {}
        if live_params is None:
            live_params = {}
        live_mode = live_params.get("live_decision_mode")
        if live_mode == "observe":
            from .live_decision import ProduceLiveDecisionObserve

            return ProduceLiveDecisionObserve().run(context, argv)
        # 调试执行只授权出牌，不消耗饮料。
        if live_mode == "debug_execute":
            logger.info(
                json.dumps(
                    {
                        "event": "live_decision_debug_preserve_drinks",
                        "mode": "debug_execute",
                        "executed": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            if not self._wait_until_playable(context):
                logger.success("事件: 自定义决策等待期间已退出出牌")
                return True
            self.start_time = time.time()
            while not context.tasker.stopping:
                from .live_decision import ProduceLiveDecisionObserve

                image = context.tasker.controller.post_screencap().wait().get()
                if not self._is_playing_card(context, image):
                    logger.success("事件: 自定义决策完成出牌")
                    return True
                executed = ProduceLiveDecisionObserve().run(context, argv)
                if not executed:
                    logger.error(
                        json.dumps(
                            {
                                "event": "live_decision_custom_stop",
                                "mode": "debug_execute",
                                "fallback": "disabled",
                                "reason": "custom_decision_not_executed",
                                "executed": False,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    return False
                time.sleep(1)
                self._wait_until_playable(context)
                self.start_time = time.time()
            return True
        else:
            task, settings = _cultivation_option(context, argv)
            algorithm = settings.get('algorithm')
            if algorithm in ('uinet-1.1', 'uinet-2.1'):
                return self._run_cultivation_policy(context, task, algorithm)
            # 使用饮料
            self._wait_until_playable(context)
            image = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition("ProduceCheckDrinkButton", image)
            hit_count = len(reco_detail.filtered_results)  # 获取饮料数
            logger.info(f"识别到{hit_count}瓶饮料")
            for _ in range(hit_count):
                context.run_task("ProduceUseDrink")
                self._wait_until_playable(context, 2)

        return self._run_recommended_exam(context)

    def _run_cultivation_policy(self, context, task, algorithm):
        from agent.live_decision.cultivation_maa_exam import CultivationMaaExam

        from .live_decision import _ocr_live_score, _locate_live_self_rank

        def report(row):
            logger.info(json.dumps(row, ensure_ascii=False))

        def read_score(ctx, image):
            result = _ocr_live_score(ctx, image, _locate_live_self_rank(ctx, image))
            return result.get('value') if result.get('accepted') else None

        try:
            entry = _cultivation_entries.get(task)
            if entry is None:
                raise ValueError('cultivation_exam_launch_not_observed')

            def factory(ctx, verify, *, log, client_size):
                return CultivationMaaExam(ctx, verify, log=log, entry=entry, algorithm=algorithm,
                                         client_size=client_size, score_reader=read_score)

            battle, _ = _open_idol_road_maa(context, report, factory=factory)
            result = battle.run_cultivation()
            if result is None:
                return False
            report(dict(event='cultivation_exam_result', produce=entry.produce, idol=entry.idol,
                        rank=result.rank, score=result.score, algorithm=algorithm))
            return True
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            logger.error(f'培育考试停止 {error}')
            return False

    def _run_recommended_exam(self, context, *, result_reader=None, authorize=None):
        from live_decision.exam_hud import read_lesson_goal, read_exam_ranking
        from live_decision.exam_loop import (
            LESSON_END_DELAY,
            EXAM_STALL_TIMEOUT,
            EXAM_CHANGE_TIMEOUT,
            ExamCycle,
            ExamOutcome,
            ExamRecovery,
            run_exam_session,
        )
        from live_decision.inventory import _inventory_ocr
        from live_decision.hif_card_retry import same_retry_scene, read_exam_select_markers
        from live_decision.hif_entry_round import read_entry_turn

        self.start_time = time.time()
        recovery = ExamRecovery()
        opening = lesson = latest = None
        pending = None
        force_skip = False
        last_progress = time.monotonic()
        exit_seen = None

        def ocr(image, **kwargs):
            return _inventory_ocr(context, image, 'ProduceRecognitionScore', **kwargs)

        def hit(node, image):
            return getattr(context.run_recognition(node, image), 'hit', False)

        def click_stage(point):
            if authorize is not None:
                authorize()
            context.tasker.controller.post_click(*point).wait()

        def snapshot(image, hand, current_lesson):
            if getattr(image, 'shape', ()) != (1280, 720, 3):
                return None
            turn = read_entry_turn(context, image)
            if turn is None:
                return None
            marks = tuple(sorted((row.box[0] // 10, row.box[1] // 10, row.label)
                                 for row in (hand.all_results if hand and hand.hit else ())
                                 if self.CARD_Y_MIN <= row.box[1] <= self.CARD_Y_MAX))
            return dict(turns_remaining=turn, marks=marks, stamina=None, shield=None, lesson=current_lesson)

        def execute(_index):
            nonlocal opening, lesson, latest, pending, force_skip, last_progress, exit_seen
            if context.tasker.stopping:
                logger.info("任务中断")
                return "finished"
            image = context.tasker.controller.post_screencap().wait().get()
            playing = self._is_playing_card(context, image)
            stalled = time.monotonic() - last_progress >= EXAM_STALL_TIMEOUT
            if not playing:
                result = result_reader(image) if result_reader is not None else read_exam_ranking(image, ocr)
                if result is not None:
                    logger.info(json.dumps(dict(event='produce_exam_result', opening=opening,
                        last_live=latest, **result), ensure_ascii=False))
                    return 'result', result
                if result_reader is not None:
                    if stalled:
                        return "stalled"
                    recovery.advance_animation(context, image,
                        is_animating=lambda: hit('ProduceRecognitionHealthFlag', image), click=click_stage)
                    return None
                if opening is not None:
                    # Like Hiro's lesson detector, a disappearing HUD can be
                    # a transition. Leave the result page for the outer task.
                    exit_seen = time.monotonic() if exit_seen is None else exit_seen
                    if time.monotonic() - exit_seen < (LESSON_END_DELAY if lesson is not None else EXAM_CHANGE_TIMEOUT):
                        return None
                if lesson is not None:
                    self.exam_result = dict(opening=opening, last_live=latest, lesson=lesson)
                    logger.info(json.dumps(dict(event='produce_lesson_hud', opening=opening,
                        last_live=latest, **lesson), ensure_ascii=False))
                logger.success("事件: 退出出牌")
                return "finished"

            exit_seen = None
            current_lesson = read_lesson_goal(image, ocr)
            if current_lesson is not None:
                lesson = current_lesson
            hand = context.run_recognition("ProduceRecognitionCards", image)
            current = snapshot(image, hand, current_lesson)
            if current is not None:
                opening = opening or current
                if latest != current:
                    last_progress = time.monotonic()
                latest = current
            if time.monotonic() - last_progress >= EXAM_STALL_TIMEOUT:
                logger.warning("出牌循环未观察到进展")
                return "stalled"

            if getattr(image, 'shape', ()) == (1280, 720, 3):
                skip_ready = hit('ProduceRecognitionSkipRound', image)
                if not skip_ready:
                    if self._handle_move_cards(context, image):
                        return None
                    if hit('ProduceButton', image):
                        if authorize is not None:
                            authorize()
                        context.run_task('ProduceButton')
                        return None
                    recovery.advance_animation(context, image,
                        is_animating=lambda: hit('ProduceRecognitionHealthFlag', image), click=click_stage)
                    return None
                if pending is not None:
                    before, until = pending
                    if current is None or read_exam_select_markers(context, image)[1]:
                        return None
                    unchanged = same_retry_scene(before, current) and before['lesson'] == current['lesson']
                    if unchanged and time.monotonic() < until:
                        return None
                    force_skip = recovery.record(no_effect=unchanged)
                    pending = None

            if hand and hand.hit:
                suggested, unusable, playable, suggestion, best = self._get_card_info(hand.all_results)
                skip = unusable > 0 and not (suggested or playable)
                box = suggestion if suggested else best if (
                    playable == 1 or time.time() - self.start_time > self.TIME_OUT) else None
            else:
                no_cards = context.run_recognition("ProduceRecognitionNoCards", image)
                skip, box = bool(no_cards and no_cards.hit), None

            if skip or force_skip:
                skip_button = context.run_recognition("ProduceRecognitionSkipRound", image)
                if skip_button and skip_button.hit:
                    if authorize is not None:
                        authorize()
                    if getattr(context.run_task("ProduceRecognitionSkipRound"), 'succeeded', True) is not False:
                        if force_skip:
                            logger.info("连续三次动作无效，跳过当前回合")
                        force_skip = False
                        pending = (current, time.monotonic() + EXAM_CHANGE_TIMEOUT) if current is not None else None
                        return "skip"
            elif box is not None and self.CARD_Y_MIN <= box[1] <= self.CARD_Y_MAX:
                self._recommended_selection_attempted = False
                if authorize is not None:
                    authorize()
                confirmed = (self._play_a_card(context, box, authorize=authorize) if authorize is not None
                             else self._play_a_card(context, box))
                if current is not None and (confirmed or self._recommended_selection_attempted):
                    pending = (current, time.monotonic() + EXAM_CHANGE_TIMEOUT)
                if confirmed:
                    return "play"
            return None

        def observe(action):
            if action in ("finished", "stalled") or isinstance(action, tuple):
                return ExamCycle(ExamOutcome.RESULT, action)
            if action == "skip":
                self.start_time = time.time()
            return ExamCycle(ExamOutcome.PROGRESS if action else ExamOutcome.REOBSERVE, action)

        def recover(_):
            time.sleep(self.CLICK_DELAY)
            return True

        def commit(action):
            if action == "stalled":
                return False
            if action == "finished":
                return result_reader is None
            if isinstance(action, tuple):
                self.exam_result = dict(opening=opening, last_live=latest, display=action[1])
                return action[1] if result_reader is not None else True
            time.sleep(self.CLICK_DELAY)
            return None

        return run_exam_session(execute=execute, observe=observe, commit=commit, recover=recover)

    @classmethod
    def _get_card_info(cls, results: list):
        """
        从手牌区域内的识别结果中获取卡牌信息，忽略场景中的误检

        Args:
            results (list): 识别结果列表

        Returns:
            suggestions (int): 建议牌数量
            useless (int): 无用牌数量
            cards (int): 可用牌数量
            suggestions_box (list): 建议牌区域，格式为[x, y, w, h]（x、y为区域左上角的坐标）
            best_box (list): 可用牌区域，格式为[x, y, w, h]（x、y为区域左上角的坐标）
        """
        label_counts = Counter()
        suggestions_box = [0, 0, 1, 1]
        best_box = [0, 0, 1, 1]
        best_score = 0
        suggestions_score = 0
        for result in results:
            # 先过滤再计数/选牌。越界推荐牌不能阻止有效普通牌的超时兜底，
            # 越界高分普通牌也不能覆盖手牌区域内的候选。
            if not cls.CARD_Y_MIN <= result.box[1] <= cls.CARD_Y_MAX:
                continue
            label_counts[result.label] += 1
            if result.label == "suggestions" and result.score > suggestions_score:
                suggestions_score = result.score
                suggestions_box = result.box
            if result.label == "cards":
                # 选出识别分数最高分的卡牌，避免错误
                if result.score > best_score:
                    best_score = result.score
                    best_box = result.box
        suggestions = label_counts["suggestions"]
        useless = label_counts["useless"]
        cards = label_counts["cards"]
        return suggestions, useless, cards, suggestions_box, best_box

    def _play_a_card(self, context: Context, box: list, *, authorize=None) -> bool:
        """Play the current recommendation using the shared SELECT sequence."""
        from live_decision.exam_loop import ExamNoProgress, select_exam_card
        from live_decision.hif_card_retry import same_retry_scene, read_exam_select_markers
        from live_decision.hif_entry_round import read_entry_turn
        from live_decision.upstream_hif_card_ui import BattleRecognitionMixin

        controller = context.tasker.controller
        detector = BattleRecognitionMixin()

        def stopped():
            return context.tasker.stopping

        def capture():
            return controller.post_screencap().wait().get()

        def hand(image):
            result = context.run_recognition("ProduceRecognitionCards", image)
            return detector._hif_card_boxes(result.all_results if result else ())

        def target(rows):
            matches = [card for card, _ in rows if detector._hif_duplicate_box(box, card)]
            return matches[0] if len(matches) == 1 else None

        def selected(image):
            current = target(hand(image))
            if current is not None and detector._hif_has_select(context, image, current):
                return tuple(current), True
            markers, present = read_exam_select_markers(context, image)
            if current is not None:
                center = current[0] + current[2] // 2
                if any(abs(x + width // 2 - center) <= 70 for x, _, width, _ in markers):
                    return tuple(current), True
            return None, present

        def signature(image, rows):
            turns = read_entry_turn(context, image)
            return dict(turns_remaining=turns, marks=detector._hif_geometry(rows), stamina=None, shield=None) if turns else None

        image = capture()
        rows = hand(image)
        current = target(rows)
        if current is None or stopped():
            return False

        def retry_target(retry):
            later = hand(retry)
            candidate = target(later)
            if (candidate is not None and not any(detector._hif_has_select(context, retry, card) for card, _ in later)
                    and same_retry_scene(signature(image, rows), signature(retry, later))):
                return tuple(candidate)
            return None

        confirmed = False

        def tap(card, *, purpose):
            nonlocal confirmed
            x, y, width, height = card
            if authorize is not None:
                authorize()
            if stopped() or not controller.post_click(x + width // 2, y + height // 2).wait().succeeded:
                raise ExamNoProgress(purpose)
            self._recommended_selection_attempted = True
            confirmed = purpose == 'confirm-card'

        initial = tuple(current) if detector._hif_has_select(context, image, current) else None
        try:
            select_exam_card(tuple(current), tap=tap, capture=capture, stopping=stopped, read_select=selected,
                             retry_target=retry_target, reconcile=lambda: None, initial_selection=initial)
        except ExamNoProgress:
            return False
        if not confirmed:
            return False
        logger.info("出牌 耗时:{:.2f}秒".format(time.time() - self.start_time))
        self.start_time = time.time()
        return True

    @staticmethod
    def _handle_move_cards(context: Context, image=None) -> bool:
        """
        一种处理移动卡片界面的笨方法

        Args:
            context: maa的Context类
            image: 截图

        Returns:
            bool: 如果出现了移动卡牌界面并处理成功，返回True；
                如果没有出现移动卡牌界面或处理过程出现问题，返回False。
        """
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()

        reco_detail = context.run_recognition("ProduceRecognitionChooseMoveCards", image)
        if reco_detail and reco_detail.hit:
            logger.info("战斗 - 移动／复制选牌")
            y = 450
            while y < 1100:
                # for x in [140, 285, 440, 585]:
                for x in [140, 285]:
                    context.tasker.controller.post_click(x, y).wait()
                    time.sleep(0.2)
                    context.tasker.controller.post_click(x, y).wait()
                    time.sleep(0.2)
                image = context.tasker.controller.post_screencap().wait().get()
                reco_detail = context.run_recognition("ProduceRecognitionChooseMoveCards", image)
                if reco_detail and not reco_detail.hit:
                    result = context.run_task("ProduceMoveCards")
                    if result and result.status.succeeded:
                        logger.info("战斗 - 已点击移动／复制确认")
                    else:
                        logger.warning("战斗 - 移动／复制确认未完成")
                    return True
                else:
                    y = y + 100
                # 检测任务中止的情况，防止卡死，检测成功时返回False
                if context.tasker.stopping:
                    return False
        return False

    @staticmethod
    def _is_playing_card(context: Context, image=None) -> bool:
        """
        判断是否处于出牌场景

        Args:
            context: maa的Context类
            image: 截图

        Returns:
            bool: 如果处于出牌场景，返回True；否则返回False。
        """
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()

        reco_detail = context.run_recognition("ProduceRecognitionHealthFlag", image)
        if reco_detail and reco_detail.hit:
            return True
        # reco_detail = context.run_recognition('ProduceRecognitionChooseMoveCards', image)
        # if reco_detail and reco_detail.hit:
        #     return True
        return False

    def _wait_until_playable(self, context: Context, confirmation_count=1):
        """
        等待直到处于可出牌状态

        Args:
            context: maa的Context类
            confirmation_count：重复核对的次数，用来应对识别对象一闪而过的假True情况（主要存在于喝饮料的时候）

        Returns:
            bool: 处于出牌场景时，返回True；不处于出牌场景时，返回False
        """
        count_playable = 0
        count_exit = 0
        while True:
            image = context.tasker.controller.post_screencap().wait().get()

            # 通过跳过回合按钮检测是否处于可出牌状态
            reco_detail = context.run_recognition("ProduceRecognitionSkipRound", image)
            if reco_detail and reco_detail.hit:
                count_playable += 1
                if count_playable >= confirmation_count:
                    return True

            # 检测血条是否存在，如连续n次检查不到血条，则认为已退出出牌场景
            # 如果识别时间太长，2次就够了，主要避免CLEAR效果遮住血条的情况
            # 如果识别时间太短，导致提前出函数，CLEAR转PERFECT的那一回合出牌计时会差很远。如果出现这种情况，就设置为3次
            reco_detail = context.run_recognition("ProduceRecognitionHealthFlag", image)
            if not (reco_detail and reco_detail.hit):
                count_exit += 1
                if count_exit >= 2:
                    return False
            else:
                count_exit = 0

            # 解决莫名其妙的误触问题
            reco_detail = context.run_recognition("ProduceButton", image)
            if reco_detail and reco_detail.hit:
                context.run_task("ProduceButton")

            # 处理移动卡片界面
            if self._handle_move_cards(context, image):
                count_playable = 0
                count_exit = 0

            # 检测任务中止的情况，防止卡死，检测成功时返回False
            if context.tasker.stopping:
                return False

            # 睡一会
            time.sleep(1)


def _open_idol_road_maa(context, log, *, factory=None):
    from route.bootstrap import _endpoint, _product_paths

    from agent.live_decision.idol_road_maa import IdolRoadMaaBattle

    maa = _product_paths()[2]
    controller = context.tasker.controller
    endpoint = _endpoint(context, expected_maa=maa, controller_info=controller.info)
    if controller.set_screenshot_target_long_side(1280) is not True:
        raise ValueError('idol_road_maa_screenshot_profile_unavailable')

    def verify_input(bound_controller):
        # Maa's RemoteTasker replaces its controller handle on each lookup.
        return _endpoint(context, expected_maa=maa, controller_info=bound_controller.info) == endpoint

    if factory is not None:
        return factory(context, verify_input, log=log,
                       client_size=tuple(endpoint['game']['observed_client_size'])), endpoint
    return IdolRoadMaaBattle(context, verify_input, log=log), endpoint


@AgentServer.custom_action("IdolRoadChooseFloor")
class IdolRoadChooseFloor(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            navigation, _ = _open_idol_road_maa(context, lambda row: logger.info(json.dumps(row, ensure_ascii=False)))
            params = getattr(argv, 'custom_action_param', None)
            params = json.loads(params) if isinstance(params, str) else params
            if params and params.get('hiro'):
                from agent.live_decision.hiro_runtime import hiro_module

                run = hiro_module('runtime.tower').TowerRun(tower_id='idol-road')
                run.gave_up_three.extend(params.get('advanced_floors', ()))
                navigation.enter_challengeable_floor(run)
            else:
                navigation.enter_challengeable_floor()
            return True
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            logger.error(f'偶像之路 选层停止 {error}')
            return False


@AgentServer.custom_action("IdolRoadNavigate")
class IdolRoadNavigate(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = argv.custom_action_param
            params = json.loads(params) if isinstance(params, str) else params
            idol = (params or {}).get('idol', '').strip()
            if not idol:
                return False
            navigation, _ = _open_idol_road_maa(context, lambda row: logger.info(json.dumps(row, ensure_ascii=False)))
            navigation.enter_idol_road(idol)
            return True
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            logger.error(f'偶像之路 入场停止 {error}')
            return False


@AgentServer.custom_action("IdolRoadPolicyBattle")
class IdolRoadPolicyBattle(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        def report(row):
            logger.info(json.dumps(row, ensure_ascii=False))

        try:
            params = argv.custom_action_param
            params = json.loads(params) if isinstance(params, str) else params
            algorithm = params['algorithm']
            is_retry = params.get('retry') is True
            battle, endpoint = _open_idol_road_maa(context, report)
            from route.bootstrap import _process_instance
            from utils.local_data import DEFAULT_RUNTIME_ROOT

            from agent.live_decision.idol_road_navigation import TowerAttempts, TowerLoadoutCache

            directory = DEFAULT_RUNTIME_ROOT / 'idol-road'
            cache = TowerLoadoutCache(directory, _process_instance(endpoint['game']['pid'])['created_at_100ns'])
            client_size = tuple(endpoint['game']['observed_client_size'])
            if params.get('hiro'):
                from agent.live_decision.hiro_runtime import hiro_module

                hiro_module('exec.exam')  # Resolve the pinned executor before challenging the floor.
                tower_runtime = hiro_module('runtime.tower')
                battle.hiro_policy_route = True
                attempts = TowerAttempts(directory, tower_runtime.MAX_TRIES)
                while True:
                    result = battle.run(algorithm, client_size=client_size, attempts=attempts, saved=(cache, is_retry))
                    verdict = battle.settle_hiro_tower(result, attempts)
                    if verdict == 'stop':
                        return False
                    if verdict == 'next':
                        start = context.get_node_data('IdolRoadFloorOverview') or {}
                        prior = start.get('custom_action_param') or {}
                        prior = json.loads(prior) if isinstance(prior, str) else prior
                        advanced = sorted({*prior.get('advanced_floors', ()), battle.loadout.overview.layer})
                        return context.override_pipeline({'IdolRoadFloorOverview': {'custom_action_param': {
                            'hiro': True, 'advanced_floors': advanced}}})
                    is_retry = True
            attempts = None
            get_node = getattr(context, 'get_node_data', None)
            if get_node is not None:
                retry_limit = (get_node('IdolRoadRetryConfirm') or {}).get('max_hit')
                if type(retry_limit) is int and retry_limit >= 0:
                    attempts = TowerAttempts(directory, retry_limit + 1)
            result = battle.run(algorithm, client_size=client_size, attempts=attempts, saved=(cache, is_retry))
            report(dict(event='idol_road_battle_finished', algorithm=battle.algorithm, rank=result.rank, score=result.score))
            return True
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            logger.error(f'偶像之路 战斗停止 {error}')
            return False


@AgentServer.custom_action("IdolRoadContinueRetry")
class IdolRoadContinueRetry(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            battle, _ = _open_idol_road_maa(context, lambda row: logger.info(json.dumps(row, ensure_ascii=False)))
            battle.enter_same_formation_retry()
            return True
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            logger.error(f'偶像之路再挑战停止 {error}')
            return False


@AgentServer.custom_action("IdolRoadResultContinue")
class IdolRoadResultContinue(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        from live_decision.inventory import _inventory_ocr
        from live_decision.idol_road_battle import read_idol_road_result_frame

        target = None
        if argv.node_name == 'IdolRoadRetry':
            x, y, w, h = argv.box
            target = [x + w // 2, y + h // 2, 1, 1]
        try:
            frame = context.tasker.controller.post_screencap().wait().get()
            if argv.node_name == 'IdolRoadResumeClearRetry':
                from live_decision.idol_road_navigation import read_tower_clear_stars

                stars = read_tower_clear_stars(frame)
                if stars is None or stars == 3:
                    return False

            def ocr(image, **kwargs):
                return _inventory_ocr(context, image, 'IdolRoadResultOCR', **kwargs)

            if target is not None and frame is not None:
                # Match Hiro's retry button text inside the recognized button.
                for row in ocr(frame, roi=(x, y, w, h)):
                    if '再挑戦' in ''.join(row.text.split()):
                        left, top, width, height = row.box
                        target = [left + width // 2, top + height // 2, 1, 1]
                        break
            result = read_idol_road_result_frame(ocr, frame) if frame is not None else None
            logger.info(json.dumps(dict(event='idol_road_result_seen', result=result), ensure_ascii=False))
        except (ValueError, RuntimeError, TypeError) as error:
            logger.warning(f'偶像之路结算画面未读到有效分数 {error}')
        original = {'action': {'type': 'Click', 'param': {'target': target} if target else {}}}
        action = context.run_action(argv.node_name, box=argv.box, pipeline_override={argv.node_name: original})
        return bool(action and action.success)


@AgentServer.custom_action("ProduceChooseWorkAuto")
class ProduceChooseWorkAuto(CustomAction):
    """
    处理选择工作类型的窗口

    选择逻辑：
    - 体力 > 10 时：选择会扣除体力对应的工作（health_position 指定的位置）
      - 单个位置 [2] 或 [3]：选择对应的工作
      - 无位置或多位置 [2,3]：随机选择
    - 体力 <= 10 时：选择不扣除体力对应的工作
      - [2] → 选择第三个（位置3不扣体力）
      - [3] → 选择第二个（位置2不扣体力）
      - 其他情况 → 选择第一个
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        first_roi = [100, 760]
        second_roi = [100, 880]
        third_roi = [100, 1000]
        image = context.tasker.controller.post_screencap().wait().get()
        health_data = self._get_health(context, image)
        current = health_data["current"] if health_data else 0

        health_position = self._get_health_position(context, image)

        if current > 10:
            if health_position and len(health_position) == 1:
                box = second_roi if health_position[0] == 2 else third_roi
            else:
                box = random.choice([first_roi, second_roi, third_roi])
        else:
            if health_position == [2]:
                box = third_roi
            elif health_position == [3]:
                box = second_roi
            else:
                box = first_roi
        context.tasker.controller.post_click(box[0], box[1]).wait()
        return True

    @staticmethod
    def _get_health(context: Context, image) -> Optional[dict]:
        """获取当前体力数值"""
        found = read_ocr_node(context, image, "ProduceRecognitionHealth").best
        if not found:
            return None

        try:
            health_parts = found.text.split("/")
            current_health = int("".join(filter(str.isdigit, health_parts[0])))
            max_health = int("".join(filter(str.isdigit, health_parts[1])))
            ratio = current_health / max_health
            logger.info(f"体力: {current_health}/{max_health} ({ratio:.2%})")
            return {"current": current_health, "max": max_health, "ratio": ratio}
        except (ValueError, IndexError, ZeroDivisionError):
            logger.warning("体力数据解析失败")
            return None

    @staticmethod
    def _get_health_position(context: Context, image) -> Optional[list]:
        """
        获取扣除体力的图标位置

        Returns:
            None: 未检测到扣体力图标
            [2]: 第二个位置扣体力
            [3]: 第三个位置扣体力
            [2,3]: 第二和第三个位置都扣体力
        """
        reco_detail = context.run_recognition(
            "ProduceRecognitionHealthFlag",
            image,
            pipeline_override={"ProduceRecognitionHealthFlag": {"roi": [370, 830, 320, 290]}},
        )
        if reco_detail.hit:
            position = []
            for result in reco_detail.filtered_results:
                health_box_y = result.box[1]
                if health_box_y > 970:
                    position.append(3)
                else:
                    position.append(2)
            return position
        return None


@AgentServer.custom_action("ProduceChooseOptionsAuto")
class ProduceChooseOptionsAuto(CustomAction):
    """
    处理选择选项的窗口

    根据当前得分自动选择属性选项：
    1. 获取第一、第二、第三属性的当前分数
    2. THIRD_ATTR_PROBABILITY 概率选择第三属性，剩余概率选择第二属性
    3. 在可用选项中执行双击选择
    4. 若目标选项不可用，按优先级 fallback：null > 第二属性 > 第三属性 > 随机
    """

    OPTIONS_CONFIG = {
        "Vo": "produce/choose_Vo.png",
        "Da": "produce/choose_Da.png",
        "Vi": "produce/choose_Vi.png",
        "null": "produce/choose_null.png",
    }
    CLICK_DELAY = 0.5
    THIRD_ATTR_PROBABILITY = 0.6  # 选择第三属性的概率，剩余概率选择第二属性

    def __init__(self):
        super().__init__()
        self.first = "Vi"
        self.second = "Da"

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """
        执行自动选择选项

        Returns:
            bool: 选择成功返回 True，否则返回 False
        """
        preference = (
            context.get_node_data("ProduceChooseNIAEventFlag")
            .get("action", {})
            .get("param", {})
            .get("custom_action_param", {"first": "Vo", "second": "Vi"})
        )
        self.first = preference["first"]
        self.second = preference["second"]
        image = context.tasker.controller.post_screencap().wait().get()
        score = self._get_current_score(context, image) or {"Vo": 0, "Da": 0, "Vi": 0, "max": 1}
        options = self._get_available_options(context, image)

        # 计算选择
        first_score = score.get(self.first, 0)
        second_score = score.get(self.second, 0)
        # 计算第三属性（Vo、Da、Vi 中非 first/second 的那个）
        all_attrs = ["Vo", "Da", "Vi"]
        third = next(attr for attr in all_attrs if attr != self.first and attr != self.second)
        third_score = score.get(third, 0)
        logger.info(f"第一属性 {self.first}={first_score}, 第二属性 {self.second}={second_score}, 第三属性 {third}={third_score}")

        # 判断逻辑：根据 THIRD_ATTR_PROBABILITY 概率选择第三属性，剩余概率选择第二属性
        if random.random() < self.THIRD_ATTR_PROBABILITY:
            choice = third
            logger.debug(f"随机数 < {self.THIRD_ATTR_PROBABILITY}，选择第三属性: {choice}")
        else:
            choice = self.second
            logger.debug(f"随机数 >= {self.THIRD_ATTR_PROBABILITY}，选择第二属性: {choice}")

        # 找到目标选项
        target_box = None
        for opt in options:
            if choice in opt:
                target_box = opt[choice]
                break

        # 如果目标选项不可用，尝试 fallback（null 优先）
        if target_box is None:
            logger.debug(f"选项 {choice} 不可用，尝试 fallback")
            # 按 null > 第二属性 > 第三属性 > 随机的优先级选择
            priority = ["null", self.second, third, None]
            for p in priority:
                if p is None:
                    # 最后从所有可用选项中随机选择
                    if options:
                        random_opt = random.choice(options)
                        choice = next(iter(random_opt.keys()))
                        target_box = random_opt[choice]
                        logger.debug(f"Fallback 随机选择: {choice}")
                    break
                for opt in options:
                    if p in opt:
                        target_box = opt[p]
                        choice = p
                        logger.debug(f"Fallback 选择 {p}")
                        break
                if target_box:
                    break

        # 执行双击
        if target_box:
            # box 格式为 [x, y, w, h]，计算中心点
            center_x = target_box[0] + target_box[2] // 2
            center_y = target_box[1] + target_box[3] // 2
            self._double_click(context, center_x, center_y)
            logger.info(f"已选择选项: {choice}, 坐标: ({center_x}, {center_y})")
        else:
            logger.warning("没有可用选项，无法选择")
            return True  # 没有选项可选时默认返回True，避免卡死在这里
        return True

    @staticmethod
    def _get_current_score(context: Context, image) -> Optional[dict]:
        """获取当前得分"""
        score = {
            "Vo": 0,
            "Da": 0,
            "Vi": 0,
            "max": 0,
        }
        roi0_list = [[70 + i * 230, 430, 136, 80] for i in range(3)]
        roi1_list = [[150 + i * 150, 325, 136, 80] for i in range(3)]
        first_rows = _ocr_score_pair(context, image, roi0_list[0])
        use_roi_list = roi0_list if first_rows else roi1_list

        for i in range(3):
            rows = first_rows if i == 0 and first_rows else _ocr_score_pair(context, image, use_roi_list[i])
            if len(rows) == 2:
                current_score = int("".join(filter(str.isdigit, rows[0].text)))
                max_score = int("".join(filter(str.isdigit, rows[1].text)))
                logger.debug(f"第{i + 1}列得分: {current_score} / {max_score}")
                score[["Vo", "Da", "Vi"][i]] = current_score
                score["max"] = max_score if score["max"] < max_score < 9999 else score["max"]
        try:
            logger.info(f"当前得分: Vo={score['Vo']}, Da={score['Da']}, Vi={score['Vi']}, Max={score['max']}")
            return score
        except ValueError:
            logger.warning("积分数据解析失败")
            return None

    def _get_available_options(self, context: Context, image) -> List[Dict[str, Any]]:
        """获取可用选项列表"""
        available_options = []
        available_options_name = ""

        for option_name, option_img in self.OPTIONS_CONFIG.items():
            reco_detail = context.run_recognition(
                "ProduceRecognitionWorkOptions",
                image,
                pipeline_override={"ProduceRecognitionWorkOptions": {"template": option_img, "focus": None}},
            )
            if reco_detail and reco_detail.hit:
                available_options.append({option_name: reco_detail.best_result.box})
                available_options_name += f"{option_name}, "

        logger.info(f"可用选项: {available_options_name.rstrip(', ')}")
        return available_options

    def _double_click(self, context: Context, x: int, y: int):
        """执行双击操作"""
        context.tasker.controller.post_click(x, y).wait()
        time.sleep(self.CLICK_DELAY)
        context.tasker.controller.post_click(x, y).wait()


@AgentServer.custom_action("ProduceChooseMirrorAuto")
class ProduceChooseMirrorAuto(CustomAction):
    """
    自动选择试镜挑战难度

    根据当前投票数在对应试镜的阈值列表中选择满足条件的最高难度：
    - 获取当前投票数和试镜各门槛分数及坐标
    - 选取满足 vote >= threshold 的最高门槛
    - 根据降档配置先跳过指定档数，再检测锁定，锁定则继续降档
    - 降档至 threshold=0 或找到未锁定选项为止
    """

    CLICK_DELAY = 0.5

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        image = context.tasker.controller.post_screencap().wait().get()
        vote = self._get_current_vote(context, image) or 1
        mirror = self._get_current_mirror(context, image)
        lowering_difficulty = self._get_lowering_difficulty(context, argv)

        # 门槛分数降序排列
        thresholds = sorted((int(k) for k in mirror.keys()), reverse=True)

        # 找到满足当前投票的最高门槛索引
        start_idx = 0
        for i, threshold in enumerate(thresholds):
            if vote >= threshold:
                start_idx = i
                break

        # 先降指定档数，再检查锁定，锁定则继续降档直至成功或为0档
        target_threshold = 0
        x = y = 0
        start = min(start_idx + lowering_difficulty, len(thresholds) - 1)
        for i in range(start, len(thresholds)):
            target_threshold = thresholds[i]
            target_box = mirror[str(target_threshold)]
            x = target_box[0] + target_box[2] // 2
            y = target_box[1] + target_box[3] // 2

            if target_threshold == 0:
                break
            if not self._check_lock(context, image, target_box):
                break

            logger.info(f"分数 {target_threshold:,} 已锁定，继续降档")

        # 在点击前判断当前是第几镜，若启用 focus 则点击后进入手动接管
        mirror_idx = self._get_focus_mirror_index(context, image)
        task, _ = _cultivation_option(context, argv)
        entry = _cultivation_entries.get(task)
        stage = None
        if entry is not None:
            stage = mirror_idx or self._read_mirror_index(context, image)

        logger.info(f"当前投票: {vote:,}, 目标分数: {target_threshold:,}, 点击坐标: ({x}, {y - 20})")
        self._double_click(context, x, y - 20)
        if stage and target_threshold in thresholds:
            entry.round_key = {1: 'mid1', 2: 'mid2', 3: 'final'}[stage]
            entry.audition_number = sorted(thresholds).index(target_threshold) + 1

        if not mirror_idx:
            return True

        logger.info(f"检测到 focus 配置，进入手动接管模式（第{mirror_idx}镜）")
        context.run_action(
            "ProduceMirrorFocus",
            pipeline_override={
                "ProduceMirrorFocus": {
                    "focus": {
                        "Node.ActionNode.Succeeded": {
                            "content": f"[color:LimeGreen]已暂停自动培育进入手动接管状态（第{mirror_idx}镜）[/color]\n"
                            '退出手动接管请点击下方"确定"按钮',
                            "display": ["log", "modal", "notification"],
                        }
                    }
                }
            },
        )

        return True

    @staticmethod
    def _get_focus_mirror_index(context: Context, image) -> int:
        """获取当前试镜对应的 focus 镜号。"""
        node_data = context.get_node_data("ProduceMirrorFlag")
        if not node_data:
            return 0
        attach = node_data.get("attach", {})
        logger.debug(f"focus 配置: {attach}")
        # 仅当有 focus_x 为 true 时才识别
        has_focus = any(attach.get(f"focus_{i}", False) for i in range(1, 4))
        if not has_focus:
            return 0
        index = ProduceChooseMirrorAuto._read_mirror_index(context, image)
        return index if attach.get(f"focus_{index}", False) else 0

    @staticmethod
    def _read_mirror_index(context: Context, image) -> int:
        for i in range(1, 4):
            reco_detail = context.run_recognition(
                f"ProduceMirrorFlag_{i}",
                image,
                pipeline_override={
                    f"ProduceMirrorFlag_{i}": {
                        "recognition": "TemplateMatch",
                        "roi": [12, 630, 696, 530],
                        "template": f"produce/NIA/mirror_{i}.png",
                        "threshold": 0.9,
                    }
                },
            )
            if reco_detail and reco_detail.hit:
                return i
        return 0

    @staticmethod
    def _check_lock(context: Context, image, target_box) -> bool:
        """检查目标分数附近是否有锁定图标"""
        roi = [
            target_box[0] + 330,
            target_box[1] - 80,
            100,
            100,
        ]
        reco_detail = context.run_recognition(
            "ProduceRecognitionLock",
            image,
            pipeline_override={
                "ProduceRecognitionLock": {
                    "recognition": "TemplateMatch",
                    "roi": roi,
                    "template": "produce/lock.png",
                    "green_mask": True,
                }
            },
        )
        return reco_detail is not None and reco_detail.hit

    @staticmethod
    def _get_lowering_difficulty(context: Context, argv) -> int:
        """检查是否启动降档"""
        params = json.loads(argv.custom_action_param)
        lowering_level = params.get("level") if isinstance(params, dict) else None
        if not lowering_level:
            return 0
        else:
            logger.info(f"检测到启动降档，降低{lowering_level}档")
            return int(lowering_level)

    @staticmethod
    def _get_current_mirror(context: Context, image):
        """获取当前试镜"""
        mirror = {"0": [360, 1050, 1, 1]}
        for result in read_ocr_node(context, image, "ProduceRecognitionMirror").rows:
            box = result.box
            text = "".join(filter(str.isdigit, result.text))
            mirror[text] = [box[0], box[1], box[2], box[3]]
        mirror = dict(sorted(mirror.items(), key=lambda x: int(x[0])))
        logger.info(f"当前试镜门槛分数: {list(mirror.keys())}")
        return mirror

    @staticmethod
    def _get_current_vote(context: Context, image) -> Optional[int]:
        """获取当前投票"""
        found = read_ocr_node(context, image, "ProduceRecognitionVote",
                              accept=lambda rows: rows[0].text.replace(",", "").isdecimal()).best
        if found:
            try:
                vote = int(found.text.replace(",", ""))
                return vote
            except ValueError:
                logger.warning("投票数据解析失败")
                return None

    def _double_click(self, context: Context, x: int, y: int):
        """执行双击操作"""
        context.tasker.controller.post_click(x, y).wait()
        time.sleep(self.CLICK_DELAY)
        context.tasker.controller.post_click(x, y).wait()


@AgentServer.custom_action("ProduceKeepDrinkAuto")
class ProduceKeepDrinkAuto(CustomAction):
    """
    处理保留饮料的窗口
    原本是在ProduceButton节点直接按保留按钮处理
    但是需要处理只有2瓶饮料时，一次性获得2瓶饮料时不会自动勾选3瓶的特殊情况
    所以转为独立处理
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        image = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("ProduceRecognitionUncheckedMark", image)
        if reco_detail.hit:
            for result in reco_detail.filtered_results:
                box = result.box
                context.tasker.controller.post_click(box[0] + int(box[2] / 2), box[1] + int(box[3] / 2)).wait()
                time.sleep(0.1)
            context.run_task("ProduceDrinkNoButton")
        return True
