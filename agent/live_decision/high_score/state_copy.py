"""Copy explicitly audited immutable records without rebuilding every field.

Registered exact types have generated constructors and no post-init hooks.
Validation remains explicit at the same kernel boundaries. Unknown types and
subclasses keep dataclasses.replace, including their constructor behavior.
"""
from dataclasses import fields
from dataclasses import replace as dataclass_replace

from .contracts import DeckBelief, StrategyState, ScheduledEffectRef, EffectTriggerContext

_COPY_FIELDS = {kind: frozenset(field.name for field in fields(kind) if field.init)
                for kind in (StrategyState, DeckBelief, ScheduledEffectRef, EffectTriggerContext)}


def register_copy_record(kind):
    """Explicit opt-in for audited generated records, applied after dataclass."""
    _COPY_FIELDS[kind] = frozenset(field.name for field in fields(kind) if field.init)
    return kind


def replace(obj, /, **changes):
    kind = type(obj)
    allowed = _COPY_FIELDS.get(kind)
    if allowed is None or hasattr(kind, "__post_init__"):
        return dataclass_replace(obj, **changes)
    if not changes.keys() <= allowed:
        return dataclass_replace(obj, **changes)
    result = object.__new__(kind)
    object.__setattr__(result, "__dict__", obj.__dict__ | changes)
    return result
