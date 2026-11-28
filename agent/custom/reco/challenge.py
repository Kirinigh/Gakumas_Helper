"""Arena pipeline recognition on Maa's current frame."""

import json

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from arena_winrate.cancellation import ArenaTaskCancelled
from arena_winrate.maa_challenge_actions import recognize_pending_outcome, recognize_result_communication

from agent.live_decision.simd_ocr import without_vulkan_cold_wait
from agent.live_decision.ocr_nodes import read_ocr_node


@AgentServer.custom_recognition("ArenaChallengeOutcomePage")
class ArenaChallengeOutcomePage(CustomRecognition):
    def analyze(self, context, argv):
        try:
            evidence = recognize_pending_outcome(context, argv.image)
        except ArenaTaskCancelled:
            return None
        if evidence is None:
            return None
        return CustomRecognition.AnalyzeResult(
            box=evidence["box"], detail={"capture_id": evidence["capture_id"]},
        )


@AgentServer.custom_recognition("ArenaPipelineText")
class ArenaPipelineText(CustomRecognition):
    @without_vulkan_cold_wait()
    def analyze(self, context, argv):
        override = {
            "expected": json.loads(argv.custom_recognition_param)["expected"],
            "roi": list(argv.roi),
            "order_by": "Horizontal",
        }
        observed = read_ocr_node(
            context, argv.image, "ArenaReaderOCR", override=override,
            accept=lambda rows: len(rows) == 1, prefer_gpu=True,
        )
        if observed.best is None:
            return None
        return self.AnalyzeResult(box=list(observed.best.box), detail={"text": observed.best.text})


@AgentServer.custom_recognition("ArenaResultCommunication")
class ArenaResultCommunication(CustomRecognition):
    @without_vulkan_cold_wait()
    def analyze(self, context, argv):
        box = recognize_result_communication(context, argv.image)
        if box is None:
            return None
        return self.AnalyzeResult(box=list(box), detail={})
