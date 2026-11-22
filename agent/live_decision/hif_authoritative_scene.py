"""Batch the existing scene recognitions through serial Context RPCs."""
import math
import time
from copy import deepcopy

from .popup_guard import TEMPLATES, popup_configuration, validate_popup_detail
from .hif_popup_parallel import HifPopupMatcher

HEALTH = 'ProduceRecognitionHealthFlag'
SKIP = 'ProduceRecognitionSkipRound'
CHOICE = 'ProduceRecognitionChooseMoveCards'
POPUP = 'ProduceButton'


def prepare_scene_batch(context, individual_hit):
    """Check node attributes once, before the authoritative capture clocks."""
    try:
        nodes = {name: context.get_node_data(name) for name in (HEALTH, SKIP, CHOICE, POPUP)}
        if any(not isinstance(node, dict) or node.get('enabled', True) is not True
               or node.get('inverse', False) is not False
               or node.get('recognition', {}).get('type') != 'TemplateMatch'
               for node in nodes.values()):
            return None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    try:
        popup_params = popup_configuration(nodes[POPUP])
        popup_count = len(TEMPLATES)
    except (TypeError, ValueError, KeyError):
        popup_params = None
        popup_count = None  # Preserve the original active configuration's rule.
    mirror = getattr(context, 'gkh_hif_popup_mirror', None)
    if not isinstance(mirror, HifPopupMatcher) or popup_params is None:
        mirror = None
    return SceneBatch(context, individual_hit, popup_count, popup_params, mirror)


class SceneBatch:
    def __init__(self, context, individual_hit, popup_count, popup_params=None, popup_mirror=None):
        self.context = context
        self.individual_hit = individual_hit
        self.popup_count = popup_count
        self.popup_params = popup_params
        self.popup_mirror = popup_mirror
        self.popup_parts_supported = popup_params is not None
        self.popup_prefix_elapsed_ms = None
        self.popup_suffix_elapsed_ms = None
        self._popup_prefix = None
        self.events = []

    def _native_popup_subset(self, image, measured, indices):
        """Keep proven low peaks; native still owns every uncertain matrix.

        This is the same per-template partition as the existing four-template
        path. No native matrix, peak suppression or threshold is approximated.
        The native raw result is explicitly scoped to its template subset.
        """
        indices = tuple(indices)
        rows = measured.get('rows', ())
        if (len(rows) != len(indices) or any(
                row.get('template') != TEMPLATES[index]
                or type(row.get('score')) not in (int, float)
                or not math.isfinite(row['score']) or not -1 <= row['score'] <= 1
                or row.get('threshold') != .7
                for index, row in zip(indices, rows, strict=True))):
            return None  # An incomplete computation keeps the original full RPC.
        pending = [index for index, row in zip(indices, rows, strict=True)
                   if not 0 < row['score'] < .5]
        if not pending or len(pending) == len(indices):
            return None
        params = deepcopy(self.popup_params)
        params['template'] = [TEMPLATES[index] for index in pending]
        if len(params['threshold']) == len(TEMPLATES):
            params['threshold'] = [params['threshold'][index] for index in pending]
        node = 'Task275HifPopupSubset'
        started = time.perf_counter_ns()
        detail = self.context.run_recognition(node, image, pipeline_override={node: {
            'recognition': 'TemplateMatch', **params,
        }})
        algorithm = getattr(detail, 'algorithm', None)
        if getattr(algorithm, 'value', algorithm) != 'TemplateMatch':
            raise ValueError('HIF popup subset has another recognition algorithm')
        hit = validate_popup_detail(detail, expected_template_count=len(pending),
            expected_template_sizes=tuple(self.popup_mirror.template_sizes[index] for index in pending))
        return dict(name=node, templates=params['template'], hit=hit,
            elapsed_ms=(time.perf_counter_ns()-started)/1_000_000., raw_detail=detail.raw_detail)

    def popup_hit(self, image):
        if self.popup_mirror is not None:
            measured = self.popup_mirror.match(image)
            event = dict(node=POPUP, kind='PopupPixels', mode='parallel_pixels',
                         measurement=measured, fallback_nodes=[])
            if not measured['needs_native']:
                event['votes'] = {POPUP: measured['hit']}
                self.events.append(event)
                return measured['hit']
            subset = self._native_popup_subset(image, measured, range(8))
            if subset is not None:
                event.update(mode='parallel_pixels_native_subset', native_subset=subset,
                    fallback_nodes=[subset['name']], votes={POPUP: subset['hit']})
                self.events.append(event)
                return subset['hit']
            event['fallback_nodes'].append(POPUP)
            self.events.append(event)
        detail = self.context.run_recognition(POPUP, image)
        hit = validate_popup_detail(detail, expected_template_count=self.popup_count,
            expected_template_sizes=self.popup_mirror.template_sizes if self.popup_mirror is not None else None)
        if self.popup_mirror is not None:
            event['votes'] = {POPUP: hit}
            event['raw_detail'] = detail.raw_detail
        return hit

    def _popup_part(self, image, part):
        params = deepcopy(self.popup_params)
        params['template'] = list(TEMPLATES[part*4:(part+1)*4])
        if len(params['threshold']) == len(TEMPLATES):
            params['threshold'] = params['threshold'][part*4:(part+1)*4]
        node = 'Task275HifPopupPart' + str(part)
        started = time.perf_counter_ns()
        measured = self.popup_mirror.match(image, part=part) if self.popup_mirror is not None else None
        if measured is not None and not measured['needs_native']:
            return dict(name=node, templates=params['template'], hit=measured['hit'],
                        elapsed_ms=(time.perf_counter_ns()-started)/1_000_000.,
                        mode='parallel_pixels', measurement=measured)
        if measured is not None:
            subset = self._native_popup_subset(image, measured, range(part*4,(part+1)*4))
            if subset is not None:
                return dict(name=node, templates=params['template'], hit=subset['hit'],
                    elapsed_ms=(time.perf_counter_ns()-started)/1_000_000.,
                    mode='parallel_pixels_native_subset', measurement=measured, native_subset=subset)
        detail = self.context.run_recognition(node, image, pipeline_override={node: {
            'recognition': 'TemplateMatch', **params,
        }})
        elapsed_ms = (time.perf_counter_ns()-started)/1_000_000.
        if getattr(detail, 'algorithm', None) not in ('TemplateMatch',):
            if getattr(getattr(detail, 'algorithm', None), 'value', None) != 'TemplateMatch':
                raise ValueError('HIF popup partition has another recognition algorithm')
        hit = validate_popup_detail(detail, expected_template_count=4,
            expected_template_sizes=self.popup_mirror.template_sizes[part*4:(part+1)*4]
                if self.popup_mirror is not None else None)
        result = dict(name=node, templates=params['template'], hit=hit,
                      elapsed_ms=elapsed_ms, raw_detail=detail.raw_detail)
        if measured is not None:
            result.update(mode='native_fallback', measurement=measured)
        return result

    def start_popup(self, image):
        """Measure four original templates without emitting a negative vote."""
        if not self.popup_parts_supported or self._popup_prefix is not None:
            raise ValueError('HIF popup partition is unavailable or already pending')
        first = self._popup_part(image, 0)
        self._popup_prefix = (image, first)

    def finish_popup(self, image):
        """Only both complete parts on the same saved frame yield one vote."""
        if self._popup_prefix is None or self._popup_prefix[0] is not image:
            raise ValueError('HIF popup partition belongs to another frame')
        first = self._popup_prefix[1]
        second = self._popup_part(image, 1)
        self._popup_prefix = None
        self.popup_prefix_elapsed_ms = first['elapsed_ms']
        self.popup_suffix_elapsed_ms = second['elapsed_ms']
        hit = first['hit'] or second['hit']
        self.events.append(dict(node=POPUP, kind='PopupParts',
            mode='parallel_pixel_parts' if self.popup_mirror is not None else 'serial_parts',
            sub_results=[first, second], fallback_nodes=[], votes={POPUP: hit}))
        return hit

    def observe_with_popup_parts(self, image):
        """The first full frame measures its actual prefix cost for scheduling."""
        health, playable = self._group('And', (HEALTH, SKIP), image)
        choice = self._individual(CHOICE, image)
        self.start_popup(image)
        popup = self.finish_popup(image)
        return health, playable, choice, popup

    def _individual(self, name, image):
        return self.popup_hit(image) if name == POPUP else self.individual_hit(self.context, name, image)

    def _group(self, kind, names, image):
        node = 'Task275HifScene' + kind
        event = dict(node=node, kind=kind, mode='serial_batch', sub_results=[], fallback_nodes=[])
        values = {}
        try:
            detail = self.context.run_recognition(node, image, pipeline_override={node: {
                'recognition': kind, 'all_of' if kind == 'And' else 'any_of': list(names),
            }})
            if (detail is None or type(getattr(detail, 'hit', None)) is not bool
                    or getattr(getattr(detail, 'algorithm', None), 'value', detail.algorithm) != kind):
                raise ValueError('scene batch recognition is missing or has another algorithm')
            groups = detail.all_results
            if not isinstance(groups, (tuple, list)) or len(groups) != 1:
                raise ValueError('scene batch lacks one complete result group')
            children = groups[0].sub_results
            if not isinstance(children, (tuple, list)) or not 1 <= len(children) <= len(names):
                raise ValueError('scene batch child list is incomplete')
            for expected, child in zip(names, children):
                if (child.name != expected or type(child.hit) is not bool
                        or getattr(child.algorithm, 'value', child.algorithm) != 'TemplateMatch'):
                    raise ValueError('scene batch child identity or result is invalid')
                if expected == POPUP:
                    validate_popup_detail(child, expected_template_count=self.popup_count)
                values[expected] = child.hit
                event['sub_results'].append(dict(name=child.name, hit=child.hit,
                    algorithm='TemplateMatch', raw_detail=child.raw_detail))
            aggregate = all(values.values()) if kind == 'And' else any(values.values())
            short_circuit = (kind == 'And' and not aggregate) or (kind == 'Or' and aggregate)
            if detail.hit != aggregate or len(children) < len(names) and not short_circuit:
                raise ValueError('scene batch short-circuit evidence is inconsistent')
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as error:
            values = {}
            event.update(mode='individual_fallback', reason=str(error), sub_results=[])
        for name in names:
            if name not in values:
                event['fallback_nodes'].append(name)
                values[name] = self._individual(name, image)
        event['votes'] = dict(values)
        self.events.append(event)
        return tuple(values[name] for name in names)

    def observe(self, image, *, include_popup):
        health, playable = self._group('And', (HEALTH, SKIP), image)
        if include_popup and self.popup_mirror is not None:
            choice, popup = self._individual(CHOICE, image), self.popup_hit(image)
        elif include_popup:
            choice, popup = self._group('Or', (CHOICE, POPUP), image)
        else:
            choice, popup = self._individual(CHOICE, image), False
        return health, playable, choice, popup
