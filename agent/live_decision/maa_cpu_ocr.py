"""Resource-only Maa CPU fallback. The child never binds a game controller."""

import sys
import json
import base64
import logging
from pathlib import Path
from functools import cache


@cache
def _configured_worker():
    from agent.utils.runtime_paths import resolve_runtime_path
    from agent.live_decision.simd_ocr import OcrWorker

    ocr = resolve_runtime_path(Path('assets/resource/base/model/ocr'), Path('resource/base/model/ocr'))
    if not (ocr / 'rec.onnx').is_file():
        return None
    return OcrWorker((sys.executable, '-u', __file__, str(ocr.parents[1])))


def read_native_cpu_ocr(image, options):
    """Return the CPU response, or None if the private CPU resource cannot run."""
    if options.get('model') or options.get('color_filter'):
        return None
    worker = _configured_worker()
    if worker is None or not worker.available:
        return None
    try:
        return worker.request_pixels(image, options=options)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        worker.available = False
        logging.getLogger(__name__).warning('Maa CPU OCR worker unavailable: %s', error)
        return None


def _serve(bundle):
    import numpy as np
    from maa.tasker import Tasker
    from maa.library import Library
    from maa.pipeline import JOCR
    from maa.resource import Resource

    resource = Resource()
    if not resource.use_cpu() or not resource.post_bundle(bundle).wait().succeeded:
        raise RuntimeError('Could not load Maa CPU OCR resource')
    tasker = Tasker()
    if not Library.framework().MaaTaskerBindResource(tasker._handle, resource._handle):
        raise RuntimeError('Could not bind Maa CPU OCR resource')

    print('{"ready":true}')
    for message in sys.stdin:
        try:
            request = json.loads(message)
            pixels = np.frombuffer(base64.b64decode(request['pixels']), np.uint8)
            frame = pixels.reshape(request['height'], request['width'], 3)
            task = tasker.post_recognition('OCR', JOCR(**request['options']), frame).wait().get()
            detail = task.nodes[0].recognition if task and task.nodes else None
            row = detail.best_result if detail and detail.hit else None
            match = dict(box=list(detail.box), text=row.text, score=row.score) if row else None
            print(json.dumps({'match': match}, ensure_ascii=True))
        except (RuntimeError, ValueError, KeyError) as error:
            print(json.dumps({'error': str(error)}, ensure_ascii=True))


if __name__ == '__main__':
    _serve(Path(sys.argv[1]))
