# SPDX-License-Identifier: AGPL-3.0
# Source: miyu1019/MaaGakumasu fb8fd00cfcab031d9a968888ac2fa48b299b241b
# extensions/hif/agent/hif/action/battle_recognition.py
# 2026-11-06: extract the complete card UI state machine unchanged;
# policy, diagnostic and Maa transport bindings live in the caller.
"""Pinned upstream selection, submission, UI response and playable wait."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from utils import logger

if TYPE_CHECKING:
    from maa.context import Context


class BattleRecognitionMixin:
    TIME_OUT = 15.0

    SKIP_STABLE_COUNT = 3

    CARD_Y_MIN = 840

    CARD_Y_MAX = 1000

    CARD_HAND_STABLE_COUNT = 2

    CARD_HAND_X_TOLERANCE = 30

    CARD_HAND_POLL_INTERVAL = 0.2

    HIF_CARD_STATE_TIMEOUT = 3.0

    HIF_CARD_CLICK_ATTEMPTS = 2

    HIF_PREVIEW_X_TOLERANCE = 45

    HIF_OCCLUDED_X_TOLERANCE = 15

    HIF_RAISED_Y = 15

    HIF_DESELECT_POS = (5, 5)

    def _in_range(self, box: list) -> bool:
        """校验卡牌Y轴坐标，超出范围视为识别异常，不点击。"""
        return self.CARD_Y_MIN <= box[1] <= self.CARD_Y_MAX

    def _reset_hif_recognition(self) -> None:
        self._hif_hand_geometry = ()
        self._hif_hand_boxes = ()
        self._hif_missing_cards = []
        self._hif_detail_keys = {}
        self._hif_detail_upgraded = {}
        self._hif_detail_attempted = set()
        self._hif_probe_started_at = None
        self._hif_probe_expired = False
        self._hif_empty_started_at = None
        self._hif_empty_count = 0
        self._hif_empty_timeout_reported = False
        self._hif_skip_reason = None
        self._hif_recognition_turn = None
        self._hif_unusable_signature = None
        self._hif_unusable_count = 0
        self._hif_selection_index = None
        self._hif_unselected_count = 0
        self._hif_diagnostic_saved = False
        self._hif_selection_pair_saved = False
        self._reset_card_hand_stability()

    def _hif_card_boxes(self, results: list) -> list:
        """建议是手牌属性；牌框在 cards/suggestions 间切换时不丢掉该牌。"""
        candidates = [(r.box, r.score) for r in results
                      if r.label in ("cards", "suggestions") and r.score >= 0.5 and self._in_range(r.box)
                      and 0 <= r.box[0] < 720 and r.box[2] >= 90
                      and 180 <= r.box[3] <= 330 and r.box[1] + r.box[3] <= 1170]
        kept = []
        for box, score in sorted(candidates, key=lambda item: item[0][2] * item[0][3], reverse=True):
            if not any(self._hif_duplicate_box(box, other) for other, _ in kept):
                kept.append((box, score))
        return sorted(kept, key=lambda item: item[0][0] + item[0][2] // 2)

    def _hif_duplicate_box(self, box: list, other: list) -> bool:
        # 选中牌抬起后会遮住邻牌；只去掉至少半数落在另一框内的重复框/窄条。
        area, other_area = box[2] * box[3], other[2] * other[3]
        iou = self._iou(box, other)
        intersection = iou * (area + other_area) / (1 + iou)
        return intersection >= min(area, other_area) * 0.50

    @staticmethod
    def _hif_geometry(boxes: list) -> tuple:
        return tuple(box[0] + box[2] // 2 for box, _ in boxes)

    def _hif_same_geometry(self, boxes: list) -> bool:
        current = self._hif_geometry(boxes)
        return len(current) == len(self._hif_hand_geometry) and all(
            abs(old - new) <= self.CARD_HAND_X_TOLERANCE
            for old, new in zip(self._hif_hand_geometry, current))

    def _hif_bind_hand(self, boxes: list) -> None:
        current = self._hif_geometry(boxes)
        previous_boxes = list(self._hif_hand_boxes)
        previous_geometry = list(self._hif_hand_geometry)
        previous_keys = dict(self._hif_detail_keys)
        previous_upgraded = dict(self._hif_detail_upgraded)
        previous_attempted = set(self._hif_detail_attempted)
        # 同一手牌中的漏框只暂存身份；恢复到可见卡位后才能参与决策或点击。
        for box, key, attempted in self._hif_missing_cards:
            old = len(previous_boxes)
            previous_boxes.append(box)
            previous_geometry.append(box[0] + box[2] // 2)
            if key:
                previous_keys[old] = key
            if attempted:
                previous_attempted.add(old)
        mapping = {}
        pairs = sorted((abs(x - cx), index, old)
                       for index, cx in enumerate(current)
                       for old, x in enumerate(previous_geometry)
                       if abs(x - cx) <= self.HIF_PREVIEW_X_TOLERANCE)
        for _, index, old in pairs:
            if index not in mapping and old not in mapping.values():
                mapping[index] = old
        # 末张牌可能只剩右侧窄条，中心会偏移超过45像素；仅接受一对一的半框重叠。
        overlap = {index: [old for old, previous in enumerate(previous_boxes)
                           if old not in mapping.values() and self._hif_duplicate_box(box, previous)]
                   for index, (box, _) in enumerate(boxes) if index not in mapping}
        for index, matches in overlap.items():
            if len(matches) == 1 and sum(matches[0] in others for others in overlap.values()) == 1:
                mapping[index] = matches[0]
        self._hif_missing_cards = [(list(box), previous_keys.get(old), old in previous_attempted)
                                   for old, box in enumerate(previous_boxes) if old not in mapping.values()
                                   and (old in previous_keys or old in previous_attempted)]
        self._hif_detail_keys = {new: previous_keys[old] for new, old in mapping.items() if old in previous_keys}
        self._hif_detail_upgraded = {new: previous_upgraded[old] for new, old in mapping.items() if old in previous_upgraded}
        self._hif_detail_attempted = {new for new, old in mapping.items() if old in previous_attempted}
        self._hif_selection_index = next((new for new, old in mapping.items()
                                         if old == self._hif_selection_index), None)
        self._hif_hand_geometry = current
        self._hif_hand_boxes = tuple(list(box) for box, _ in boxes)


    @staticmethod
    def _hif_card_is_raised(box) -> bool:
        # A raised card ends above the regular hand row. The recommended
        # card halo may start high but still extends below the ordinary row.
        return box[1] <= 875 and box[1] + box[3] <= 1130

    @staticmethod
    def _hif_panel_top(image) -> Optional[int]:
        """详情框向上扩展；以浅色面板定位顶部，避开效果正文。"""
        if cv2 is None or image is None or image.shape[:2] != (1280, 720):
            return None
        region = image[240:850, 80:640]
        white = ((region.min(axis=2) > 218)
                 & (region.max(axis=2) - region.min(axis=2) < 30)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        panels = [(int(area), int(y) + 240) for x, y, w, h, area in stats[1:]
                  if 480 <= w <= 560 and h >= 160 and area >= 50000
                  and 0 <= x <= 35 and 810 <= y + h + 240 <= 845]
        return max(panels)[1] if panels else None

    @staticmethod
    def _hif_has_select(context: Context, image, box: list) -> bool:
        cx = box[0] + box[2] // 2
        # 两个局部裁剪覆盖九张实测样本，避免整条手牌 OCR 与邻卡名字连读。
        for left, top, width, height in ((cx - 70, 1106, 140, 40), (cx - 60, 1110, 120, 35)):
            reco = context.run_recognition("ProduceHIF__ProduceIdentitySelected", image, pipeline_override={
                "ProduceHIF__ProduceIdentitySelected": {"recognition": "OCR", "expected": "SELECT",
                                            "roi": [max(0, left), top, width, height]}})
            if reco and reco.hit:
                return True
        return False

    def _hif_selected_index(self, context: Context, image, boxes: list, *, pending: bool = True) -> int:
        selected = [index for index, (box, _) in enumerate(boxes)
                    if self._hif_has_select(context, image, box)]
        if len(selected) == 1:
            # A temporarily missing neighbor must not shift the selected card's identity/index.
            if self._hif_hand_geometry:
                box = boxes[selected[0]][0]
                center = box[0] + box[2] // 2
                positions = [i for i, old in enumerate(self._hif_hand_geometry)
                             if abs(old - center) <= self.HIF_PREVIEW_X_TOLERANCE]
                if len(positions) != 1:
                    return -2
                selected = positions
            self._hif_unselected_count = 0
            self._hif_selection_index = selected[0]
            return selected[0]
        panel = self._hif_panel_top(image)
        raised = any(self._hif_card_is_raised(box) for box, _ in boxes)
        if selected or panel is not None or raised:
            self._hif_unselected_count = 0
            return -2
        # 延迟选中失败后可能留下旧提示。两帧完整原手牌、无详情/SELECT/抬起牌才清理。
        if cv2 is not None and self._hif_same_geometry(boxes):
            self._hif_unselected_count += 1
            if pending and self._hif_unselected_count >= 2 and self._hif_selection_index is not None:
                logger.info("HIF点击前复核: 连续两帧未选中，清理上次选中记录")
                self._hif_selection_index = None
        else:
            self._hif_unselected_count = 0
        return -2 if pending and self._hif_selection_index is not None else -1

    def _hif_frame_boxes(self, context: Context, image) -> list:
        reco = context.run_recognition("ProduceHIF__ProduceRecognitionCards", image)
        return self._hif_card_boxes(reco.all_results if reco else [])

    def _hif_preview_box(self, boxes: list, index: int, *, context=None, image=None):
        """选中期间允许邻牌暂时漏框，但不接受新卡位或跨卡位关联。"""
        if index >= len(self._hif_hand_geometry):
            return None
        matches = {}
        extra = False
        for box, _ in boxes:
            cx = box[0] + box[2] // 2
            nearby = [i for i, x in enumerate(self._hif_hand_geometry)
                      if abs(x - cx) <= self.HIF_PREVIEW_X_TOLERANCE]
            if not nearby:
                nearby = [i for i, previous in enumerate(self._hif_hand_boxes)
                          if self._hif_duplicate_box(box, previous)]
            # 抬起的选中牌会遮住带建议光效的邻牌，使其在两侧卡位之间时有时无；
            # 这种中间漏框不改变已绑定卡位，仅在其余卡位全部可见时忽略；换手时卡位整体移动，仍会被拒绝。
            if not nearby and min(self._hif_hand_geometry) < cx < max(self._hif_hand_geometry):
                extra = True
                continue
            if len(nearby) != 1 or nearby[0] in matches:
                return None
            matches[nearby[0]] = box
        geometry = self._hif_hand_geometry
        steady = all(abs(box[0] + box[2] // 2 - geometry[i]) <= self.HIF_OCCLUDED_X_TOLERANCE for i, box in matches.items())
        if extra and (len(matches) != len(geometry) or not steady):
            return None
        target = matches.get(index)
        # 目标牌本身被抬起的邻牌遮挡漏框：其余卡位原地可见且相邻牌明显抬起时，按记录的卡位点选；
        # 选中后由 SELECT 与详情连续两帧确认，不会盲目出牌。
        if (target is None and not extra and steady and len(geometry) >= 3 and len(matches) == len(geometry) - 1
                and index < len(self._hif_hand_boxes)):
            previous = self._hif_hand_boxes[index]
            if any(i in matches and previous[1] - matches[i][1] >= self.HIF_RAISED_Y
                   for i in (index - 1, index + 1)):
                return list(previous)
        # 选中后模型可能只检测到详情面板；必须用当前帧 SELECT 和面板确认原卡位。
        if (target is None and context is not None and self._hif_selection_index == index
                and index < len(self._hif_hand_boxes) and self._hif_panel_top(image) is not None):
            previous = self._hif_hand_boxes[index]
            if self._hif_has_select(context, image, previous):
                return previous
        return target

    @staticmethod
    def _hif_click_card(context: Context, box: list, index: int, phase: str, attempt: int) -> bool:
        point = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        job_id = None
        try:
            job = context.tasker.controller.post_click(*point)
            job_id = job.job_id
            job.wait()
            if not job.succeeded:
                logger.warning(f"HIF{phase}点击执行未成功: job_id={job_id}，重新识别")
                return False
            retry = f" (第{attempt + 1}次)" if attempt else ""
            logger.info(f"HIF{phase}点击: 第{index + 1}张 point={point}{retry}，等待画面确认")
            return True
        except Exception as exc:
            logger.warning(f"HIF{phase}点击执行异常: job_id={job_id} error={exc!r}，重新识别")
            return False

    def _hif_select_card(self, context: Context, image, index: int):
        if context.tasker.stopping:
            return None
        image = context.tasker.controller.post_screencap().wait().get()
        boxes = self._hif_frame_boxes(context, image)
        if self._hif_preview_box(boxes, index) is None:
            self._hif_log_changed("select_skip", f"HIF选中: 第{index + 1}张卡位与当前帧不一致，本次不点击; "
                                  f"卡位={self._hif_hand_geometry}，当前={self._hif_geometry(boxes)}", signature=("mismatch", index))
            return None
        skip = context.run_recognition("ProduceHIF__ProduceRecognitionSkipRound", image)
        if not (skip and skip.hit):
            self._hif_log_changed("select_skip", f"HIF选中: 第{index + 1}张选中前未见SKIP，本次不点击")
            return None
        selected = self._hif_selected_index(context, image, boxes)
        if selected == -2:
            logger.warning("HIF选中卡位置未确认，本次不点击")
            return None
        for attempt in range(self.HIF_CARD_CLICK_ATTEMPTS):
            clicked = False
            before_image = image.copy() if self._hif_debug_enabled and hasattr(image, "copy") else image
            if selected != index:
                box = self._hif_preview_box(boxes, index)
                if box is None:
                    return None
                if not self._hif_click_card(context, box, index, "选中", attempt):
                    return None
                clicked = True
                self._hif_selection_index = index
                self._hif_unselected_count = 0
                time.sleep(0.4)
            else:
                logger.info(f"HIF选中复核: 第{index + 1}张已选中，本次未发出选中点击")
            deadline = time.time() + self.HIF_CARD_STATE_TIMEOUT
            stable = missed = 0
            while time.time() < deadline and not context.tasker.stopping:
                image = context.tasker.controller.post_screencap().wait().get()
                boxes = self._hif_frame_boxes(context, image)
                target = self._hif_preview_box(boxes, index, context=context, image=image)
                if target is None:
                    stable = missed = 0
                    time.sleep(self.CARD_HAND_POLL_INTERVAL)
                    continue
                target_selected = self._hif_has_select(context, image, target)
                stable = stable + 1 if target_selected and (cv2 is None or self._hif_panel_top(image) is not None) else 0
                if stable >= 2:
                    self._hif_selection_index = index
                    logger.info(f"HIF画面选中确认: 第{index + 1}张 SELECT及详情连续两帧，选中点击={clicked}")
                    return image
                # 保留满3秒展开时间；末尾两帧明确仍未选中才允许重试。
                if target_selected:
                    missed = 0
                else:
                    observed = self._hif_selected_index(context, image, boxes, pending=False)
                    playable = context.run_recognition("ProduceHIF__ProduceRecognitionSkipRound", image)
                    missed = missed + 1 if observed != -2 and playable and playable.hit else 0
                time.sleep(self.CARD_HAND_POLL_INTERVAL)
            if missed < 2 or context.tasker.stopping:
                break
            self._hif_selection_index = None
            if not clicked:
                logger.info("HIF选中复核: 本次未发出点击，原选中状态已消失，重新识别")
                return None
            selected = -1
            if attempt == 0:
                logger.warning("HIF选中点击控制器已执行，但画面仍未选中目标，重试一次")
                self._hif_log_selection_pair(before_image, image, index)
        logger.warning("HIF选中未确认，重新识别；状态不明确时不连续点击")
        return None

    def _hif_read_detail(self, context: Context, index: int) -> tuple:
        deadline = time.time() + self.HIF_CARD_STATE_TIMEOUT
        previous = ""
        count = 0
        while time.time() < deadline and not context.tasker.stopping:
            image = context.tasker.controller.post_screencap().wait().get()
            boxes = self._hif_frame_boxes(context, image)
            top = self._hif_panel_top(image)
            target = self._hif_preview_box(boxes, index, context=context, image=image)
            if target is None:
                count = 0
                time.sleep(self.CARD_HAND_POLL_INTERVAL)
                continue
            text = ""
            if top is not None and self._hif_has_select(context, image, target):
                reco = context.run_recognition("ProduceHIF__ProduceIdentityDetail", image, pipeline_override={
                    "ProduceHIF__ProduceIdentityDetail": {"recognition": "OCR", "expected": r"[^\n]+",
                                              "roi": [150, top + 8, 410, 40]}})
                if reco and reco.hit:
                    text = "".join(r.text or "" for r in sorted(reco.filtered_results, key=lambda r: r.box[0]))
            normalized = self._hif_card_name(text)
            count = count + 1 if normalized and normalized == previous else 1 if normalized else 0
            previous = normalized
            if count >= 2:
                return self._hif_name_key(text), text
            time.sleep(self.CARD_HAND_POLL_INTERVAL)
        return None, ""

    def _hif_selection_visible(self, context: Context, image, boxes: list) -> bool:
        raised = any(box[1] <= 875 and box[1] + box[3] <= 1130 for box, _ in boxes)
        return (raised or self._hif_panel_top(image) is not None
                or any(self._hif_has_select(context, image, box) for box, _ in boxes))

    def _hif_clear_selection(self, context: Context) -> bool:
        """取消选中并确认连续两帧无 SELECT、详情和抬起牌；画面本就未选中时不点击。"""
        # 第0轮只看画面：首帧仍选中才点左上角；之后每次点击等满状态超时再判断。
        for attempt in range(self.HIF_CARD_CLICK_ATTEMPTS + 1):
            if attempt:
                context.tasker.controller.post_click(*self.HIF_DESELECT_POS).wait()
                retry = f" (第{attempt}次)" if attempt > 1 else ""
                logger.info(f"HIF取消选中: 点左上角 @ {self.HIF_DESELECT_POS}{retry}")
                time.sleep(0.4)
            deadline = time.time() + self.HIF_CARD_STATE_TIMEOUT
            clear = 0
            while time.time() < deadline and not context.tasker.stopping:
                image = context.tasker.controller.post_screencap().wait().get()
                boxes = self._hif_frame_boxes(context, image)
                clear = clear + 1 if boxes and not self._hif_selection_visible(context, image, boxes) else 0
                if clear >= 2:
                    self._hif_selection_index = None
                    return True
                if not attempt and not clear:
                    break
                time.sleep(self.CARD_HAND_POLL_INTERVAL)
            if context.tasker.stopping:
                return False
        logger.warning("HIF取消选中未确认，暂不按当前手牌决策，重新识别")
        return False

    def _play_hif_card(self, context: Context, box: list) -> bool:
        image = context.tasker.controller.post_screencap().wait().get()
        boxes = self._hif_frame_boxes(context, image)
        cx = box[0] + box[2] // 2
        if not self._hif_hand_geometry:
            self._hif_log_changed("play_skip", "HIF出牌: 尚无已绑定卡位，重新识别")
            return False
        index = min(range(len(self._hif_hand_geometry)), key=lambda i: abs(self._hif_hand_geometry[i] - cx))
        if abs(self._hif_hand_geometry[index] - cx) > self.CARD_HAND_X_TOLERANCE:
            self._hif_log_changed("play_skip", f"HIF出牌: 目标卡位已偏移 cx={cx}，卡位={self._hif_hand_geometry}，重新识别",
                                  signature="offset")
            return False
        if not boxes or self._hif_preview_box(boxes, index) is None:
            self._hif_log_changed("play_skip", f"HIF出牌: 当前帧手牌与已绑定卡位不一致，重新识别; "
                                  f"卡位={self._hif_hand_geometry}，当前={self._hif_geometry(boxes)}", signature="mismatch")
            self._reset_card_hand_stability()
            if boxes:
                self._hif_bind_hand(boxes)
            return False
        image = self._hif_select_card(context, image, index)
        if image is None:
            return False
        boxes = self._hif_frame_boxes(context, image)
        target = self._hif_preview_box(boxes, index, context=context, image=image)
        if target is None:
            return False
        for attempt in range(self.HIF_CARD_CLICK_ATTEMPTS):
            if not self._hif_click_card(context, target, index, "确认出牌", attempt):
                return False
            deadline = time.time() + self.HIF_CARD_STATE_TIMEOUT
            transitioned = still_selected = 0
            while time.time() < deadline and not context.tasker.stopping:
                image = context.tasker.controller.post_screencap().wait().get()
                skip = context.run_recognition("ProduceHIF__ProduceRecognitionSkipRound", image)
                selected = self._hif_has_select(context, image, target)
                panel = self._hif_panel_top(image)
                changed = not (skip and skip.hit) or (panel is None and not selected)
                transitioned = transitioned + 1 if changed else 0
                still_selected = still_selected + 1 if selected and (cv2 is None or panel is not None) else 0
                if transitioned >= 2:
                    logger.info("HIF出牌: 确认点击后，已观察到详情关闭或动画转换")
                    self._hif_log_reset("play_skip", "select_skip")
                    self._reset_hif_recognition()
                    self._wait_until_playable(context, confirmation_count=self.SKIP_STABLE_COUNT)
                    return True
                time.sleep(self.CARD_HAND_POLL_INTERVAL)
            target = self._hif_preview_box(self._hif_frame_boxes(context, image), index, context=context, image=image)
            if still_selected < 2 or target is None or context.tasker.stopping:
                break
            if attempt == 0:
                logger.warning("HIF出牌点击未命中，已确认原卡仍选中，重试确认一次")
        logger.warning("HIF出牌点击后未确认画面转换，重新识别，不盲连点")
        return False

    def _reset_card_hand_stability(self) -> None:
        self._pending_card_hand = None
        self._pending_card_hand_count = 0
        self._pending_card_hand_started_at = None

    @staticmethod
    def _card_hand_snapshot(identified: list) -> tuple:
        """按横向顺序保存整手可出牌的身份与中心位置。"""
        return tuple(
            (card.get("key"), card["box"][0] + card["box"][2] // 2)
            for card in sorted(
                identified,
                key=lambda card: card["box"][0] + card["box"][2] // 2,
            )
        )

    def _same_card_hand(self, previous: tuple, current: tuple) -> bool:
        return len(previous) == len(current) and all(
            old_key == new_key
            and abs(old_x - new_x) <= self.CARD_HAND_X_TOLERANCE
            for (old_key, old_x), (new_key, new_x) in zip(previous, current)
        )

    def _confirm_card_hand(self, identified: list) -> str:
        """所有职业共用：整手牌连续稳定后才进入出牌决策。"""
        hand = self._card_hand_snapshot(identified)
        now = time.time()
        if self._pending_card_hand_started_at is None:
            self._pending_card_hand_started_at = now

        if self._pending_card_hand is not None and self._same_card_hand(
            self._pending_card_hand, hand
        ):
            self._pending_card_hand_count += 1
        else:
            self._pending_card_hand_count = 1
        self._pending_card_hand = hand

        if self._pending_card_hand_count >= self.CARD_HAND_STABLE_COUNT:
            # 两帧稳定的名字身份与详情身份共用本手缓存，邻牌被遮挡时不重复补认。
            self._hif_detail_keys.update({index: card["key"] for index, card in enumerate(identified)
                                          if card.get("key")})
            self._hif_detail_upgraded.update({index: card["upgraded"] for index, card in enumerate(identified)
                                              if card.get("upgraded") is not None})
            names = [self._card_label(key) for key, _ in hand]
            waited = now - self._pending_card_hand_started_at
            # 不逐帧记录待稳定手牌；等待明显偏长时说明期间识别在变化（漏框、遮挡等）。
            note = f"（等待{waited:.1f}秒才稳定）" if waited >= 2 else ""
            self._hif_log_changed("hand", f"整手牌识别已稳定: {names}{note}", signature=(self._last_turn_count, names))
            self._reset_card_hand_stability()
            return "stable"
        if now - self._pending_card_hand_started_at >= self.TIME_OUT:
            logger.warning(
                f"整手牌连续 {self.TIME_OUT:.0f} 秒未稳定，"
                f"采用最后一次识别: {[self._card_label(key) for key, _ in hand]}"
            )
            self._reset_card_hand_stability()
            return "stable"
        return "wait"

    @staticmethod
    def _iou(a: list, b: list) -> float:
        """两框交集/并集(IOU)。"""
        ax0, ay0, aw, ah = a
        bx0, by0, bw, bh = b
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _play_a_card(self, context: Context, box: list) -> bool:
        """
        选中并确认打出一张手牌；只有观察到画面转换才计为已出牌。

        Args:
            context: maa的Context类
            box: 卡牌范围，格式为[x, y, w, h]（x、y为左上角的坐标）

        Returns:
            bool: 已确认打出返回True；未确认时返回False，由主循环重新识别，不盲连点。
        """
        played = self._play_hif_card(context, box)
        if played:
            self._battle_confirmed_plays = getattr(self, '_battle_confirmed_plays', 0) + 1
        return played

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
        deadline = time.monotonic() + self.TIME_OUT
        while True:
            if context.tasker.stopping:
                return False
            if time.monotonic() >= deadline:
                logger.warning("HIF等待可出牌状态超过15秒，结束本次等待，交回状态识别")
                return False
            image = context.tasker.controller.post_screencap().wait().get()

            # 通过跳过回合按钮检测是否处于可出牌状态。
            # 特效动画期 skip 按钮会被特效遮盖/未稳定出现，故"连续"命中 N 次（非累计）
            # 才认为是真正的可出牌状态，从而滤掉出牌后特效期的动画帧（避免手牌漏框）。
            reco_detail = context.run_recognition("ProduceHIF__ProduceRecognitionSkipRound", image)
            if reco_detail and reco_detail.hit:
                count_playable += 1
                if count_playable >= confirmation_count:
                    return True
            else:
                count_playable = 0  # 未连续命中：一 miss 即清零，确保是连续稳定

            # 检测血条是否存在，如连续n次检查不到血条，则认为已退出出牌场景
            # 如果识别时间太长，2次就够了，主要避免CLEAR效果遮住血条的情况
            # 如果识别时间太短，导致提前出函数，CLEAR转PERFECT的那一回合出牌计时会差很远。如果出现这种情况，就设置为3次
            reco_detail = context.run_recognition("ProduceHIF__ProduceRecognitionHealthFlag", image)
            if not (reco_detail and reco_detail.hit):
                count_exit += 1
                if count_exit >= 2:
                    return False
            else:
                count_exit = 0

            if context.tasker.stopping:
                return False

            # 解决莫名其妙的误触问题
            reco_detail = context.run_recognition("ProduceHIF__ProduceButton", image)
            if reco_detail and reco_detail.hit:
                context.run_task("ProduceHIF__ProduceButton")

            # 处理移动卡片界面
            if self._handle_move_cards(context, image):
                count_playable = 0
                count_exit = 0

            # 检测任务中止的情况，防止卡死，检测成功时返回False
            if context.tasker.stopping:
                return False

            # 睡一会
            time.sleep(1)
