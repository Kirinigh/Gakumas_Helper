"""HIF read-only scene recognition; never calls an action node."""

from route.observation import observe_hif_scene
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition


@AgentServer.custom_recognition("ProduceHifSceneObserve")
class ProduceHifSceneObserve(CustomRecognition):
    def analyze(self, context, argv):
        detail = observe_hif_scene(context, argv.image)
        # Return the stop reason too. The bridge decides whether a candidate is
        # usable; failing recognition must not fall through to generic clicks.
        return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail=detail)
