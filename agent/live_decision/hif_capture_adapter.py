"""Use Hiro's logical screen coordinates over the existing Maa controller.

This wraps the existing Maa controller; it never supplies a new input endpoint.
Endpoint/elevation and frame-age checks remain the caller's responsibility.
"""

import time
import weakref
from copy import deepcopy

MAA_CAPTURE_ERROR_PREFIX = 'Maa 截图失败：'
_MARKER = '_gkh_hif_capture_adapter'


class HifCaptureAdapterError(ValueError):
    """The current HIF capture context cannot safely map this operation."""


class _Forwarder:
    def __init__(self, wrapped):
        object.__setattr__(self, '_hif_wrapped', wrapped)

    def __getattr__(self, name):
        return getattr(self._hif_wrapped, name)

    def __setattr__(self, name, value):
        if name.startswith('_hif_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._hif_wrapped, name, value)

    def __delattr__(self, name):
        if name.startswith('_hif_'):
            object.__delattr__(self, name)
        else:
            delattr(self._hif_wrapped, name)


class _CaptureJob(_Forwarder):
    def __init__(self, wrapped, adapter, requested_at):
        super().__init__(wrapped)
        self._hif_adapter = adapter
        self._hif_requested_at = requested_at
        self._hif_captured_at = None
        self._hif_image = None

    def wait(self):
        self._hif_wrapped.wait()
        if self._hif_captured_at is None:
            # Preserve the Maa completion time, before copying/resizing pixels.
            self._hif_captured_at = time.perf_counter()
        return self

    def get(self, wait=False):
        if wait:
            self.wait()
        if self._hif_image is not None:
            return self._hif_image
        if self._hif_adapter._hif_latest is not self:
            raise HifCaptureAdapterError('HIF capture job was superseded before reading its image')
        if not self._hif_wrapped.succeeded:
            raise HifCaptureAdapterError('HIF capture job has not succeeded')
        if self._hif_captured_at is None:
            # Polling without wait has no exact completion timestamp. The request
            # timestamp is conservative; reading an old buffer must not freshen it.
            self._hif_captured_at = self._hif_requested_at
        raw = self._hif_wrapped.get()
        self._hif_image = self._hif_adapter._normalize(raw, self._hif_captured_at)
        return self._hif_image


class HifCaptureAdapter(_Forwarder):
    """Track frame provenance and map input through the latest Maa capture."""

    def __init__(self, controller, input_guard=None):
        super().__init__(controller)
        self._hif_screen = None
        self._hif_error = None
        self._hif_latest = None
        self._hif_frames = {}
        self._hif_capture_count = 0
        self._hif_last_metadata = None
        self._hif_input_guard = None
        if input_guard is not None:
            self.bind_input_guard(input_guard)

    def bind_input_guard(self, guard):
        if (not callable(guard) or (self._hif_input_guard is not None
                                   and self._hif_input_guard is not guard)):
            raise HifCaptureAdapterError('HIF input ownership guard cannot be replaced')
        self._hif_input_guard = guard

    def _authorise_input(self):
        if self._hif_input_guard is not None and self._hif_input_guard() is not True:
            raise HifCaptureAdapterError('HIF input ownership is not active')

    @property
    def _gkh_hif_capture_adapter(self):
        return self

    def _normalize(self, image, captured_at):
        import numpy as np

        from .hiro_runtime import hiro_module

        geometry = hiro_module('devices.base')
        if self._hif_screen is None:
            self._hif_screen = geometry.LogicalScreen()
        try:
            if image is not None and (not isinstance(image, np.ndarray) or image.dtype != np.uint8):
                raise geometry.DeviceUnavailable('控制器未返回 8 位图像')
            normalized = self._hif_screen.frame(image)
        except geometry.DeviceUnavailable as error:
            self._hif_error = f'{MAA_CAPTURE_ERROR_PREFIX}{error}'
            raise HifCaptureAdapterError(self._hif_error) from error
        self._hif_error = None
        metadata = dict(captured_at=captured_at, raw_shape=list(image.shape),
                        effective_shape=list(normalized.shape), normalized=normalized is not image)
        key = id(normalized)
        self._hif_frames[key] = (
            weakref.ref(normalized, lambda _ref: self._hif_frames.pop(key, None)), metadata)
        self._hif_last_metadata = metadata
        self._hif_capture_count += 1
        return normalized

    def frame_metadata(self, image):
        record = self._hif_frames.get(id(image))
        if record is None or record[0]() is not image:
            raise HifCaptureAdapterError('Frame was not captured through this HIF context')
        return deepcopy(record[1])

    def audit(self):
        last_frame = deepcopy(self._hif_last_metadata)
        return dict(
            raw_shape=last_frame['raw_shape'] if last_frame else None,
            effective_shape=last_frame['effective_shape'] if last_frame else None,
            capture_count=self._hif_capture_count,
            last_frame=last_frame,
            invalid_reason=self._hif_error,
        )

    def post_screencap(self, *args, **kwargs):
        requested_at = time.perf_counter()
        job = _CaptureJob(self._hif_wrapped.post_screencap(*args, **kwargs), self, requested_at)
        self._hif_latest = job
        return job

    def capture_with_guard(self, guard, *args, **kwargs):
        """Keep normalization/provenance around either trusted capture path."""
        if not callable(guard):
            raise HifCaptureAdapterError('HIF capture guard must be callable')
        requested_at = time.perf_counter()
        capture = getattr(self._hif_wrapped, 'capture_with_guard', None)
        if callable(capture):
            raw = capture(guard, *args, **kwargs)
        else:
            if guard() is not True:
                raise HifCaptureAdapterError('HIF capture budget guard was revoked')
            raw = self._hif_wrapped.post_screencap(*args, **kwargs)
        job = _CaptureJob(raw, self, requested_at)
        self._hif_latest = job
        return job

    def capture_with_record_guard(self, guard, *args, **kwargs):
        """Preserve original pixels/time around the explicit native record guard."""
        capture = getattr(self._hif_wrapped, 'capture_with_record_guard', None)
        if not callable(guard) or not callable(capture):
            raise HifCaptureAdapterError('HIF record capture guard is unavailable')
        requested_at = time.perf_counter()
        job = _CaptureJob(capture(guard, *args, **kwargs), self, requested_at)
        self._hif_latest = job
        return job

    @property
    def cached_image(self):
        if self._hif_latest is None:
            raise HifCaptureAdapterError('No HIF capture was requested in this context')
        return self._hif_latest.get()

    def _check_input(self):
        if self._hif_latest is None or self._hif_latest._hif_image is None:
            raise HifCaptureAdapterError(self._hif_error or 'HIF input requires the latest validated capture')
        return self._hif_latest._hif_image

    def _point(self, x, y):
        height, width = self._check_input().shape[:2]
        if type(x) is not int or type(y) is not int or not (0 <= x < width and 0 <= y < height):
            raise HifCaptureAdapterError(f'HIF point outside canonical frame: {(x, y)}')
        return self._hif_screen.to_native(x, y)

    def _rect(self, rect):
        if (not isinstance(rect, (tuple, list)) or len(rect) != 4
                or any(type(value) is not int for value in rect)):
            raise HifCaptureAdapterError('HIF action requires an explicit integer rectangle')
        x, y, width, height = rect
        if width < 1 or height < 1:
            raise HifCaptureAdapterError('HIF action rectangle is empty')
        left, top = self._point(x, y)
        right, bottom = self._point(x + width - 1, y + height - 1)
        return [left, top, right - left + 1, bottom - top + 1]

    def post_click(self, x, y, *args, **kwargs):
        point = self._point(x, y)
        self._authorise_input()
        return self._hif_wrapped.post_click(*point, *args, **kwargs)

    def post_click_with_guard(self, x, y, *, final_guard):
        """Map/authorise first, then keep the caller's guard at the SDK edge."""
        point = self._point(x, y)
        self._authorise_input()
        if not callable(final_guard):
            raise HifCaptureAdapterError('HIF final input guard must be callable')
        dispatch = getattr(self._hif_wrapped, 'post_click_with_guard', None)
        if callable(dispatch):
            return dispatch(*point, final_guard=final_guard)
        if final_guard() is not True:
            raise HifCaptureAdapterError('HIF final input guard was revoked')
        return self._hif_wrapped.post_click(*point)

    @property
    def post_click_with_endpoint_guard(self):
        """Expose the owning controller's fresh endpoint receipt when supported."""
        dispatch = getattr(type(self._hif_wrapped), 'post_click_with_endpoint_guard', None)
        if not callable(dispatch):
            return None

        def guarded(x, y, *, final_guard):
            point = self._point(x, y)
            self._authorise_input()
            if not callable(final_guard):
                raise HifCaptureAdapterError('HIF final input guard must be callable')
            return dispatch(self._hif_wrapped, *point, final_guard=final_guard)

        return guarded

    def post_swipe(self, x1, y1, x2, y2, duration, *args, **kwargs):
        begin = self._point(x1, y1)
        end = self._point(x2, y2)
        self._authorise_input()
        return self._hif_wrapped.post_swipe(*begin, *end, duration, *args, **kwargs)

    def _unsupported_coordinates(self, *args, **kwargs):
        raise HifCaptureAdapterError('HIF adapter supports coordinate input only through Click/Swipe')

    post_touch_down = _unsupported_coordinates
    post_touch_move = _unsupported_coordinates
    post_relative_move = _unsupported_coordinates
    post_touch_up = _unsupported_coordinates
    post_multi_swipe = _unsupported_coordinates
    post_scroll = _unsupported_coordinates
    post_press_key = _unsupported_coordinates
    post_key_down = _unsupported_coordinates
    post_key_up = _unsupported_coordinates
    post_input_text = _unsupported_coordinates


def _unsupported_pipeline(*args, **kwargs):
    raise HifCaptureAdapterError('HIF adapter requires an explicit single-node Click/Swipe action')


class _TaskerProxy(_Forwarder):
    def __init__(self, wrapped, controller):
        super().__init__(wrapped)
        self._hif_controller = controller

    @property
    def controller(self):
        return self._hif_controller

    # Pipeline execution can resolve implicit coordinates inside Maa. The HIF
    # observer does not use it; do not silently bypass normalization here.
    post_task = staticmethod(_unsupported_pipeline)


class _ContextProxy(_Forwarder):
    def __init__(self, wrapped, tasker, adapter):
        super().__init__(wrapped)
        self._hif_tasker = _TaskerProxy(tasker, adapter)
        self._hif_adapter = adapter

    @property
    def tasker(self):
        return self._hif_tasker

    @property
    def gkh_hif_capture_adapter(self):
        return self._hif_adapter

    def run_action(self, entry, box=(0, 0, 0, 0), reco_detail='', pipeline_override=None):
        if (tuple(box) != (0, 0, 0, 0) or reco_detail
                or not isinstance(pipeline_override, dict) or set(pipeline_override) != {entry}):
            return _unsupported_pipeline()
        node = pipeline_override[entry]
        if not isinstance(node, dict):
            return _unsupported_pipeline()
        node = deepcopy(node)
        if node.get('action') == 'Click' and set(node) == {'action', 'target'}:
            node['target'] = self._hif_adapter._rect(node['target'])
        elif node.get('action') == 'Swipe' and set(node) == {'action', 'begin', 'end', 'duration'}:
            if type(node['duration']) is not int or node['duration'] <= 0:
                return _unsupported_pipeline()
            node['begin'] = self._hif_adapter._rect(node['begin'])
            node['end'] = self._hif_adapter._rect(node['end'])
        else:
            return _unsupported_pipeline()
        self._hif_adapter._authorise_input()
        return self._hif_wrapped.run_action(entry, pipeline_override={entry: node})

    run_action_direct = staticmethod(_unsupported_pipeline)
    run_task = staticmethod(_unsupported_pipeline)


def _existing_adapter(controller):
    existing = getattr(controller, _MARKER, None)
    # Require the adapter's self-marker, not merely an attribute synthesized by
    # a general-purpose proxy/test double. This also works across import aliases.
    return existing if existing is not None and getattr(existing, _MARKER, None) is existing else None


def adapt_hif_context(context, *, input_guard=None):
    """Return a forwarding context; repeated or externally wrapped calls are safe.

    Wrappers must delegate unknown attributes (as the existing HIF host does).
    """
    tasker = context.tasker
    controller = tasker.controller
    existing = _existing_adapter(controller)
    if existing is not None:
        if input_guard is not None:
            existing.bind_input_guard(input_guard)
        return context
    return _ContextProxy(context, tasker, HifCaptureAdapter(controller, input_guard))


def captured_at(controller, image, fallback=None):
    """Use the original Maa capture time, never the time a cached frame is read.

    Unadapted callers retain their existing timestamp or clock behavior. An
    adapted controller must know this exact frame; fallback cannot freshen an
    unknown/copy/cached frame or hide missing capture provenance.
    """
    adapter = _existing_adapter(controller)
    if adapter is not None:
        return adapter.frame_metadata(image)['captured_at']
    return time.perf_counter() if fallback is None else fallback
