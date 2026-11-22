"""Reuse successful immutable-object validation within one offline search only."""
from functools import wraps
from contextlib import contextmanager
from contextvars import ContextVar

_VALIDATED = ContextVar("high_score_validated_objects", default=None)
_CHECK_BUDGET = ContextVar("high_score_kernel_budget", default=None)
_TRANSFORMS = ContextVar("high_score_state_transforms", default=None)


@contextmanager
def validation_session(check_budget=None, *, reuse_parent=False):
    """Optionally share an enclosing search's immutable catalogs/state caches.

    Reuse is explicit: independent decisions and ordinary nested sessions keep
    fresh caches. A reused child retains both its own and its parent's budget.
    """
    validated = _VALIDATED.get() if reuse_parent else None
    transforms = _TRANSFORMS.get() if reuse_parent else None
    if validated is not None and transforms is not None:
        parent_check = _CHECK_BUDGET.get()
        if parent_check is not None:
            child_check = check_budget

            def check_budget():
                parent_check()
                if child_check is not None:
                    child_check()
    else:
        validated, transforms = {}, {}
    token = _VALIDATED.set(validated)
    budget_token = _CHECK_BUDGET.set(check_budget)
    transform_token = _TRANSFORMS.set(transforms)
    try:
        yield
    finally:
        _VALIDATED.reset(token)
        _CHECK_BUDGET.reset(budget_token)
        _TRANSFORMS.reset(transform_token)


def check_search_budget():
    check = _CHECK_BUDGET.get()
    if check is not None:
        check()


def immutable_validation(function):
    @wraps(function)
    def validate(instance):
        cache = _VALIDATED.get()
        if cache is not None and cache.get(id(instance)) is instance:
            return
        function(instance)
        if cache is not None:
            # Strong references make id reuse impossible. Bound retained state
            # and never carry validation across observations or decisions.
            if len(cache) >= 8192:
                cache.clear()
            cache[id(instance)] = instance
    return validate


def immutable_state_transform(function):
    """Memoize an idempotent normalization within one decision.

    Callers must satisfy f(f(x)) == f(x); normalized outputs are also cached.
    Both keys keep strong references, so object-id reuse cannot alias a state.
    """
    @wraps(function)
    def transform(state):
        cache = _TRANSFORMS.get()
        key = (function, id(state))
        entry = cache.get(key) if cache is not None else None
        if entry is not None and entry[0] is state:
            return entry[1]
        result = function(state)
        if cache is not None:
            if len(cache) >= 8192:
                cache.clear()
            cache[key] = (state, result)
            cache[(function, id(result))] = (result, result)
        return result
    return transform


def immutable_projection(function):
    """Cache a pure immutable-object projection; unlike transforms, not idempotent."""
    @wraps(function)
    def project(instance):
        cache = _TRANSFORMS.get()
        key = (function, id(instance))
        entry = cache.get(key) if cache is not None else None
        if entry is not None and entry[0] is instance:
            return entry[1]
        result = function(instance)
        if cache is not None:
            if len(cache) >= 8192:
                cache.clear()
            cache[key] = (instance, result)
        return result
    return project


def registered_state_validation(function):
    """Cache a successful kernel/state validation only within one decision.

    Kernel catalogs are fixed during a search. A new decision gets a fresh
    cache, so updates to definitions or observations are always revalidated.
    """
    @wraps(function)
    def validate(kernel, state, *, plan_type):
        cache = _TRANSFORMS.get()
        key = (function, id(kernel), id(state), plan_type)
        entry = cache.get(key) if cache is not None else None
        if entry is not None and entry[0] is state and entry[1] is kernel:
            return
        function(kernel, state, plan_type=plan_type)
        if cache is not None:
            if len(cache) >= 8192:
                cache.clear()
            cache[key] = (state, kernel)
    return validate
