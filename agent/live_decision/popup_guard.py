"""Serial Maa popup recognition, retaining the active resource images."""
import math
from copy import deepcopy

TEMPLATES = (
    'next.png', 'produce/round_close_button.png', 'produce/retry.png',
    'produce/yes.png', 'produce/start.png', 'produce/close.png',
    'produce/shop_exit.png', 'cancel.png',
)
ROI = (0, 780, 720, 500)


def _configuration(context):
    return popup_configuration(context.get_node_data('ProduceButton'))


def popup_configuration(node):
    """Validate one already-read node without another Context RPC."""
    if (not isinstance(node, dict) or node.get('enabled', True) is not True
            or node.get('inverse', False) is not False):
        raise ValueError('popup node is missing, disabled or inverted')
    recognition = node.get('recognition', {})
    if recognition.get('type') != 'TemplateMatch':
        raise ValueError('popup recognition is not the verified template method')
    params = deepcopy(recognition.get('param', {}))
    if (set(params) != {'template', 'roi', 'roi_offset', 'threshold', 'order_by',
                       'index', 'method', 'green_mask'}
            or tuple(params['template']) != TEMPLATES or tuple(params['roi']) != ROI
            or tuple(params['roi_offset']) != (0, 0, 0, 0)
            or params['method'] != 5 or params['green_mask'] is not False
            or params['order_by'] != 'Score' or params['index'] != 0
            or params['threshold'] not in ([.7], [.7]*len(TEMPLATES))):
        raise ValueError('popup configuration differs from the verified full-color slice')
    return params


def validate_popup_detail(detail, *, expected_template_count=None, expected_template_sizes=None):
    """A missing or incomplete negative is never proof that no popup exists."""
    if detail is None or type(getattr(detail, 'hit', None)) is not bool:
        raise ValueError('HIF popup fallback recognition did not run')
    if not detail.hit:
        rows = tuple(getattr(detail, 'all_results', ()) or ())
        if not rows:
            raise ValueError('HIF popup fallback has no measured evidence')
        if expected_template_sizes is not None:
            _validate_mirrored_negative(detail, rows, expected_template_count, expected_template_sizes)
        elif expected_template_count is not None and len(rows) != expected_template_count:
            raise ValueError('HIF popup fallback did not measure every template')
    return detail.hit


def _validate_mirrored_negative(detail, rows, expected_count, template_sizes):
    """Native NMS may return several sub-threshold boxes for one template.

    Only the controlled mirror supplies these active pixel dimensions. Unique
    dimensions identify every measured template without treating candidate
    count as template count or comparing native peaks with the pixel shortcut.
    """
    sizes = tuple(template_sizes)
    if (type(expected_count) is not int or len(sizes) != expected_count
            or any(not isinstance(size, tuple) or len(size) != 2
                   or any(type(value) is not int or value <= 0 for value in size) for size in sizes)
            or len(set(sizes)) != expected_count):
        raise ValueError('HIF popup mirror template dimensions are not unique')
    raw = getattr(detail, 'raw_detail', None)
    algorithm = getattr(detail, 'algorithm', None)
    if (getattr(algorithm, 'value', algorithm) != 'TemplateMatch'
            or not isinstance(raw, dict) or raw.get('filtered') != []
            or 'best' not in raw or raw['best'] is not None):
        raise ValueError('HIF popup native negative detail is inconsistent')
    covered = set()
    rx, ry, rw, rh = ROI
    for row in rows:
        box, score = getattr(row, 'box', None), getattr(row, 'score', None)
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or any(type(value) is not int for value in box)
                or type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= .7):
            raise ValueError('HIF popup native candidate is invalid')
        x, y, width, height = box
        size = (width, height)
        if size not in sizes or not (rx <= x <= rx+rw-width and ry <= y <= ry+rh-height):
            raise ValueError('HIF popup native candidate has no unique template owner')
        covered.add(size)
    if covered != set(sizes):
        raise ValueError('HIF popup fallback did not measure every template')


def parallel_hif_popup_hit(context, image):
    """Compatibility entry point: run the unchanged node once, serially.

    AgentServer context clones share a reverse-RPC response channel. Concurrent
    recognition can consume another call's response and deadlock the task. Do
    not clone or dispatch context methods from worker threads, including for
    read-only recognition. The historical function name is kept for callers.
    """
    # The original node owns all active parameters and resource/image overrides.
    # None/error is an observation failure, never a negative vote.
    detail = context.run_recognition('ProduceButton', image)
    expected_count = None
    if detail is not None and not detail.hit:
        try:
            _configuration(context)
        except (ValueError, TypeError, KeyError, AttributeError):
            pass  # The original node owns a changed, unsupported configuration.
        else:
            expected_count = len(TEMPLATES)
    return validate_popup_detail(detail, expected_template_count=expected_count)
