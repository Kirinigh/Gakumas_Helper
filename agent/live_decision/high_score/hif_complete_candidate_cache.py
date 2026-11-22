"""One validated transaction for the strong candidate's business computation."""
from .hif_mcts_cache import json_value, require_candidate_tree_versions
from .hif_strong_tail_cache import export_tail_link, restore_tail_link
from .hif_strong_policy_cache import export_candidate_state, prepare_restore_candidate_state
from .hif_frozen_policy_continuation import export_link, restore_link

VERSION = 'hif-complete-candidate-cache-v1'


def export_complete(owner, request):
    return json_value(dict(version=VERSION,
        computation=export_candidate_state(owner, request),
        choice=owner._export_cache_choice(request),
        frozen_policy_continuation=export_link(owner._frozen_policy_continuation),
        actual_tail=export_tail_link(owner)))


def restore_complete(owner, payload, request):
    require_candidate_tree_versions(payload, owner.search, complete_policy=True)
    payload = json_value(payload)
    if (set(payload) != {'version', 'computation', 'choice', 'frozen_policy_continuation', 'actual_tail'}
            or payload['version'] != VERSION
            or owner._choice_continuation is not None
            or owner._frozen_policy_continuation is not None
            or owner._actual_tail_link is not None):
        raise ValueError('strong_cache_complete_envelope_invalid')
    # Rules, algorithm configuration, model and all tables validate before any
    # memo is rebound to the current kernel, and before any field is installed.
    apply = prepare_restore_candidate_state(owner, payload['computation'], request)
    count = payload['computation']['plan_count']
    choice = owner._restore_cache_choice(payload['choice'], request, count)
    if not owner.deterministic_policy_continuation and payload['frozen_policy_continuation'] is not None:
        raise ValueError('strong_cache_frozen_continuation_not_enabled')
    frozen = restore_link(payload['frozen_policy_continuation'], owner.kernel, request,
        context=owner.search._context(request), plan_count=count, deadline=owner.deadline)
    if not owner.actual_tail_session_reuse and payload['actual_tail'] is not None:
        raise ValueError('strong_cache_actual_tail_not_enabled')
    tail = restore_tail_link(owner, payload['actual_tail'], request, count)
    receipt = apply()
    owner._choice_continuation = choice
    owner._frozen_policy_continuation = frozen
    owner._actual_tail_link = tail
    return dict(receipt, restored_choice=choice is not None,
                restored_frozen_continuation=frozen is not None,
                restored_tail_entries=tail['session'].entry_count if tail else 0)
