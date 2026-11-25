"""Run the shared OCR worker inside standard Maa pipeline recognition nodes."""

import json

from maa.pipeline import JOCR
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition

from agent.live_decision.simd_ocr import read_optional_simd_ocr
from agent.live_decision.maa_cpu_ocr import read_native_cpu_ocr


@AgentServer.custom_recognition("LiveSimdOCR")
class LiveSimdOCR(CustomRecognition):
    def analyze(self, context, argv):
        options = json.loads(argv.custom_recognition_param)
        x, y, width, height = argv.roi
        roi = [x, y, width, height]
        full_frame = (x, y, width, height) == (0, 0, argv.image.shape[1], argv.image.shape[0])
        index = options.get("index", 0)
        worker_params = {"order_by": "Horizontal", **options}
        worker_params.pop("index", None)
        if not full_frame:
            worker_params["roi"] = roi
        # Existing text predicates can reject a GPU misread on the same frame.
        # Free text and numeric readers keep the more accurate CPU path by default.
        rows = read_optional_simd_ocr(argv.image, worker_params, prefer_gpu=bool(options.get("expected")),
                                      index=index)
        if rows:
            row = rows[index]
            return self.AnalyzeResult(box=list(row.box), detail={
                "backend": "simd", "text": row.text, "score": row.score,
            })

        native = context.run_recognition_direct("OCR", JOCR(roi=roi, **options), argv.image)
        if native is not None:
            if native.hit:
                row = native.best_result
                return self.AnalyzeResult(box=list(native.box), detail={
                    "backend": "maa", "text": row.text, "score": row.score,
                })
            return None

        response = read_native_cpu_ocr(argv.image, {"roi": roi, **options})
        if response and (match := response["match"]):
            return self.AnalyzeResult(box=match["box"], detail={
                "backend": "maa-cpu", "text": match["text"], "score": match["score"],
            })
        return None
