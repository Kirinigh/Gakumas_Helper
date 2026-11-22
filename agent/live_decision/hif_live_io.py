"""Existing audited Maa dispatch and diagnostic frame persistence services."""
import time


def _history_read_only(*_args, **_kwargs):
    raise ValueError('HIF durable history is read-only inside the record')


class _HistoryDict(dict):
    __init__ = __setitem__ = __delitem__ = __ior__ = _history_read_only
    clear = pop = popitem = setdefault = update = _history_read_only

    def __deepcopy__(self, memo):
        from copy import deepcopy

        copied = {}
        memo[id(self)] = copied
        copied.update((key, deepcopy(value, memo)) for key, value in self.items())
        return copied


class _HistoryList(list):
    __init__ = __setitem__ = __delitem__ = __iadd__ = __imul__ = _history_read_only
    append = clear = extend = insert = pop = remove = reverse = sort = _history_read_only

    def __deepcopy__(self, memo):
        from copy import deepcopy

        copied = []
        memo[id(self)] = copied
        copied.extend(deepcopy(value, memo) for value in self)
        return copied


def freeze_record_history(value):
    """Copy JSON history into private read-only containers, preserving its values.

    The Host calls this after JSON encoding has already rejected cycles and
    invalid numbers. These objects never enter the on-disk format; ordinary
    deep copies remain mutable for a later record/window's normal appenders.
    """
    copied, active = {}, set()

    def freeze(current):
        if isinstance(current, (_HistoryDict, _HistoryList)):
            return current
        if type(current) in (dict, list, tuple):
            key = id(current)
            if key in active:
                raise ValueError('HIF durable history contains a cycle')
            if key in copied:
                return copied[key]
            active.add(key)
            if type(current) is dict:
                frozen = dict.__new__(_HistoryDict)
                dict.update(frozen, ((name, freeze(child)) for name, child in current.items()))
            elif type(current) is list:
                frozen = list.__new__(_HistoryList)
                list.extend(frozen, (freeze(child) for child in current))
            else:
                frozen = tuple(freeze(child) for child in current)
            active.remove(key)
            copied[key] = frozen
            return frozen
        if type(current) not in (str, int, float, bool, type(None)):
            raise ValueError('HIF durable history contains a non-JSON value')
        return current

    # Deepcopy already preserves shared prefixes. Preserve those same links
    # when freezing; expanding them again makes every final budget check walk
    # the entire serialized history. No value or validation result is cached.
    return freeze(value)


def post_live_click_with_deadline(controller, x, y, *, result, persist,
                                  verify_endpoints, before_dispatch,
                                  input_deadline, authorization_floor,
                                  not_dispatched_error, preparation_key='native_input_preparation'):
    """Persist uncertain intent, then recheck the original deadline at Maa dispatch."""
    import math

    preparation = dict(x=x, y=y, input_deadline=input_deadline,
                       authorization_floor=authorization_floor,
                       dispatch_state='prepared_unconfirmed')
    result[preparation_key] = preparation
    preparation_started = time.perf_counter()
    dispatch_attempted = False
    endpoint_dispatch = getattr(controller, 'post_click_with_endpoint_guard', None)
    endpoints = None

    def final_guard(current_endpoints=None):
        nonlocal dispatch_attempted, endpoints
        guard_started = time.perf_counter()
        preparation['dispatch_guard_elapsed_ms'] = (guard_started - persistence_finished) * 1000
        if dispatch_attempted:
            raise ValueError('native_input_guard_already_consumed')
        if callable(endpoint_dispatch):
            # Root has just checked the current task, owner and every actual
            # input process. Consume that same receipt at the SDK edge rather
            # than doing the same OS work twice around the durable write.
            game = current_endpoints.get('game', {}) if isinstance(current_endpoints, dict) else {}
            if (not isinstance(current_endpoints, dict)
                    or not {'game', 'agent', 'maa'} <= current_endpoints.keys()
                    or game.get('pid') != result['endpoints']['game']['pid']
                    or tuple(game.get('observed_client_size', ()))
                    != tuple(result['endpoints']['game']['client_size'])
                    or any(not isinstance(row, dict) or type(row.get('pid')) is not int
                        or row['pid'] <= 0 or row.get('integrity_rid') != 0x3000
                        for row in current_endpoints.values())):
                raise not_dispatched_error('native_final_input_endpoints_changed')
            endpoints = current_endpoints
            preparation['endpoints'] = endpoints
            preparation['endpoint_source'] = 'root_final_dispatch'
        try:
            checked_at = before_dispatch()
        finally:
            guard_finished = time.perf_counter()
            preparation['before_dispatch_elapsed_ms'] = (guard_finished - guard_started) * 1000
            preparation['total_pre_dispatch_elapsed_ms'] = (guard_finished - preparation_started) * 1000
        # Includes the adapter's ownership/coordinate checks and the Root
        # controller's current record and endpoint verification, not just Host.
        final_checked_at = time.perf_counter()
        within_deadline = (
            input_deadline is None and preparation_key == 'native_navigation_preparation'
            or type(input_deadline) in (int, float) and math.isfinite(input_deadline)
            and final_checked_at < input_deadline
        )
        if (not within_deadline or not all(type(value) in (int, float) and math.isfinite(value)
                for value in (authorization_floor, checked_at, final_checked_at))
                or not authorization_floor < checked_at <= final_checked_at):
            raise not_dispatched_error('snapshot_expired_after_input_preparation')
        preparation.update(dispatch_state='dispatch_attempted', authorization_checked_at=final_checked_at)
        result['native_input_dispatches'].append(dict(x=x, y=y, endpoints=endpoints))
        result['input_count'] = len(result['native_input_dispatches'])
        dispatch_attempted = True
        return True

    try:
        if not callable(endpoint_dispatch):
            endpoints = verify_endpoints(controller.info,
                expected_client_size=tuple(result['endpoints']['game']['client_size']))
            preparation['endpoints'] = endpoints
        endpoints_finished = time.perf_counter()
        preparation['endpoint_elapsed_ms'] = (endpoints_finished - preparation_started) * 1000
        persistence_stages = persist()
        persistence_finished = time.perf_counter()
        if isinstance(persistence_stages, dict):
            preparation['persistence_stages_ms'] = persistence_stages
        preparation['persistence_elapsed_ms'] = (persistence_finished - endpoints_finished) * 1000
        preparation['pre_guard_elapsed_ms'] = (persistence_finished - preparation_started) * 1000
        if callable(endpoint_dispatch):
            return endpoint_dispatch(x, y, final_guard=final_guard)
        dispatch = getattr(controller, 'post_click_with_guard', None)
        if callable(dispatch):
            return dispatch(x, y, final_guard=final_guard)
        final_guard()
        return controller.post_click(x, y)
    except Exception as error:
        if dispatch_attempted:
            # The SDK may have accepted the command before raising. Preserve
            # its dispatch count and let the existing caller record uncertainty.
            raise
        preparation.update(dispatch_state='rejected_before_dispatch',
                           reason=str(error), input_may_have_been_dispatched=False)
        try:
            persist()
        except Exception as persistence_error:
            preparation['rejection_persistence_error'] = type(persistence_error).__name__
        raise not_dispatched_error(str(error)) from error


def persist_deferred_frames(store, session, frames, *, attempt_id, encode, capture_kind='screenshot'):
    """Use the existing store's bounded atomic batches, preserving every record."""
    started = time.perf_counter()
    report = dict(frame_count=len(frames), written_count=0, failed_count=0, batch_count=0)
    for offset in range(0, len(frames), 64):
        images, records = [], []
        for label, frame, record in frames[offset:offset + 64]:
            try:
                ok, encoded = encode('.png', frame)
                if not ok:
                    raise ValueError('image encode failed')
                images.append((encoded.tobytes(), {'label': label, 'attempt': attempt_id}))
                records.append(record)
            except Exception as error:
                record['image_not_retained'] = str(error)
                report['failed_count'] += 1
        if not images:
            continue
        report['batch_count'] += 1
        try:
            category = 'failure' if capture_kind == 'failure_scene' else 'normal'
            paths = store.write_images(category, session, images, capture_kind=capture_kind, suffix='.png')
            if len(paths) != len(records):
                raise ValueError('capture store returned an incomplete batch path list')
        except Exception as error:
            # The store owns rollback and suspension. Do not retry the same
            # failed batch as single writes or claim any image was retained.
            for record in records:
                record['image_not_retained'] = str(error)
            report['failed_count'] += len(records)
            for _, _, record in frames[offset + 64:]:
                record['image_not_retained'] = str(error)
                report['failed_count'] += 1
            break
        else:
            for record, path in zip(records, paths, strict=True):
                record['path'] = str(path)
            report['written_count'] += len(records)
    report['elapsed_ms'] = (time.perf_counter() - started) * 1000
    return report
