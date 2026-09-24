"""Recognize the bounded arena outcome continuation on Maa's current frame."""

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from arena_winrate.cancellation import ArenaTaskCancelled
from arena_winrate.maa_challenge_actions import recognize_pending_outcome


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
