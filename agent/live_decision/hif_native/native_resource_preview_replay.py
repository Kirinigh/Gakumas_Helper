"""Reconcile a stopped B with an observed, pinned current-play GENKI tier."""
from types import SimpleNamespace


def verify_selected_resource_model(checkpoint, action):
    from ..high_score.live_policy import BoundHifOpeningPolicy

    row, bundle = checkpoint['row'], checkpoint['bundle']
    failure = row['live_events'][-1]['failure_events'][0]
    resource = failure['resource_transaction']
    slot = failure['expected_selected_slot']
    if (action.kind.value != 'PLAY_CARD' or action.source_slot != slot
            or action.card_instance_id != bundle.state.deck.hand[slot].instance_id):
        raise ValueError('HIF resource model recovery changed its original selected action')
    policy = BoundHifOpeningPolicy(bundle, action_order=checkpoint['event']['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    ordinary = policy.selected_resource_preview(slot)
    if resource.get('card_effect_preview') != ordinary:
        raise ValueError('HIF stopped resource model differs from its original ordinary forecast')
    policy.bind_selected_single_plus_estimate(row['gray_selection_journal']['single_plus_estimate'])
    if policy._current_genki_support_definition() is None:
        raise ValueError('HIF stopped resource model has no observed genki-only support tier')
    preview = policy.selected_resource_preview(slot)
    observations = resource['observations']
    absolutes, costs = observations['absolutes'], observations['effective_cost']
    fields = (('stamina', bundle.state.stamina), ('shield', bundle.state.genki))
    if (absolutes.get('accepted') is not True or len(absolutes.get('observations', ())) != 3
            or len(costs) != 3 or any(c.get('accepted') is not True
                or type(c.get('value')) is not int
                or c['value'] != preview.get('displayed_card_cost', preview['effective_cost']) for c in costs)):
        raise ValueError('HIF repaired resource model lacks its three actual costs and absolutes')
    for name, before in fields:
        expected = preview['preview_' + name]
        delta = observations[name + '_delta']
        # Zero deltas can be proved by the existing clean-zero pixel evidence;
        # the normal observer owns that receipt, just as for nonzero OCR deltas.
        if (resource.get('preview_' + name) != expected or absolutes.get(name) != expected
                or any(v.get(name, {}).get('accepted') is not True
                    or v[name].get('value') != expected for v in absolutes['observations'])
                or delta.get('accepted') is not True or type(delta.get('value')) is not int
                or before + delta['value'] != expected or len(delta.get('observations', ())) != 3):
            raise ValueError('HIF repaired resource model still differs from its observed preview')
    # This is only proof that fresh reobservation can use the repaired model.
    # The source stays failed and no input/settlement receipt is manufactured.
    return preview
