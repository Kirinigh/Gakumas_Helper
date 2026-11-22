"""Retain the two owned drinks for the rest of the authorized 0030 first battle."""
from copy import copy
from dataclasses import fields, dataclass

from .kernel import DrinkDefinition

LIVE_RUN_ID = 'hif-recovered-8dc7d21c70944554a1768f863a711207'
_RETAINED = frozenset(('4', '18'))


def retains_round1_drinks(run_id, revision, state):
    return (run_id == LIVE_RUN_ID and revision == 1 and state.turn + state.turns_remaining == 10
            and len(state.drinks) == 2 and frozenset(state.drinks) == _RETAINED)


@dataclass(frozen=True)
class _ReservedDrink(DrinkDefinition):
    # Stored in the existing complete-search definitions and copied to worker kernels.
    reserved_for_later: bool = True


def reserve_drinks_for_planning(kernel):
    """Keep the observed effects and inventory, including triggers from consumed drinks."""
    search_kernel = copy(kernel)
    search_kernel.drinks = dict(kernel.drinks)
    for key in _RETAINED:
        original = kernel.drinks[key]
        search_kernel.drinks[key] = _ReservedDrink(**{
            field.name: getattr(original, field.name) for field in fields(DrinkDefinition)})
    return search_kernel
