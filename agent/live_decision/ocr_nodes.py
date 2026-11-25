"""Read Maa OCR nodes as ordered rows while preserving their native fallback."""

from typing import NamedTuple


class OcrRows(NamedTuple):
    rows: tuple
    best: object = None


def _simd_node_rows(context, image, node, override, *, accept, all_results, prefer_gpu):
    get_node_data = getattr(context, "get_node_data", None)
    if not callable(get_node_data):
        return None
    recognition = (get_node_data(node) or {}).get("recognition", {})
    if not isinstance(recognition, dict) or recognition.get("type") != "OCR":
        return None
    params = {**recognition.get("param", {}), **override}
    if params.pop("recognition", "OCR") != "OCR":
        return None
    if (params.pop("model", "") or params.pop("color_filter", "")
            or any(params.pop("roi_offset", ()))):
        return None
    index = params.pop("index", 0)
    roi = params.get("roi")
    if roi in (None, [0, 0, 0, 0], (0, 0, 0, 0)):
        params.pop("roi", None)
    elif (not isinstance(roi, (list, tuple)) or len(roi) != 4
          or any(type(value) is not int for value in roi) or min(roi[:2]) < 0 or min(roi[2:]) <= 0):
        # Relative and clipped ROIs are still evaluated by Maa itself.
        return None
    if all_results:
        # Current all-row callers ask for free text. Other queries would need
        # distinct filtered/all rows and an independently selected best result.
        if params.get("expected") not in (".+", [".+"]):
            return None
        params.update(expected=[], threshold=0)
    from agent.live_decision.simd_ocr import read_optional_simd_ocr

    rows = read_optional_simd_ocr(image, params, index=index, accept=accept, prefer_gpu=prefer_gpu)
    return OcrRows(rows, rows[index]) if rows else None


def _native_node_rows(context, image, node, override, all_results):
    options = {"pipeline_override": {node: override}} if override else {}
    detail = context.run_recognition(node, image, **options)
    if not detail or not detail.hit:
        return OcrRows(())
    filtered = getattr(detail, "filtered_results", None)
    raw = getattr(detail, "all_results", None)
    rows = (raw or filtered) if all_results else (filtered or raw)
    return OcrRows(tuple(rows or ()), getattr(detail, "best_result", None))


def read_ocr_node(context, image, node, *, override=None, accept=bool, all_results=False,
                  prefer_gpu=False, order="simd"):
    """Keep the original Maa parameters and return a single engine's rows for a frame."""
    if order == "native":
        return _native_node_rows(context, image, node, override, all_results)
    if order == "native-first":
        native = _native_node_rows(context, image, node, override, all_results)
        if accept(native.rows):
            return native
    simd = _simd_node_rows(context, image, node, override or {}, accept=accept,
                           all_results=all_results, prefer_gpu=prefer_gpu)
    if simd is not None:
        return simd
    return native if order == "native-first" else _native_node_rows(context, image, node, override, all_results)
