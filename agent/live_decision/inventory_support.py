"""Separate complete current Support text from already verified permanent keys."""
import math
import unicodedata
from copy import deepcopy

if __package__.startswith('agent.'):
    from agent.utils.effect_text import normalize_score_effect_text
else:
    from utils.effect_text import normalize_score_effect_text

from .opening_observer import valid_opening_frame_ids

PERMANENT = (('14', 1), ('54', 1))
BODY = ('破竹の勢い++好調が8ターン以上の場合、使用可集中+10'
        '以降の4ターンの間、ターン開始時、スコア+7好調2ターン試験・ステージ中1回')
SPOTLIGHT_PERMANENT = (('3', 1), ('52', 1))
SPOTLIGHT_BODY = ('スポットライト++体力消費3元気+9好調8ターン'
    '次のターン、スキルカードを2枚引く2ターン後、スキルカードを引くスコア+6')
SPOTLIGHT_SOURCE = 'hif_inventory_295_support_v1'


def _text(value):
    return ''.join(unicodedata.normalize('NFKC', value).split())


def resolve_inventory_detail(catalog, candidate, observations, frame_ids, *, bundle=None,
                             badge_observer=None):
    """Check a known complete support body before parsing permanent counts."""
    from .inventory_customization import resolve_detail_customizations

    detail = {key: candidate[key] for key in ('zone', 'index', 'card_id')}
    detail.update(rows=[[dict(text=row.text, score=row.score, box=row.box) for row in group]
                        for group in observations], frame_ids=list(frame_ids))
    if bundle is not None and detail['card_id'] == '295' and any(
            '++' in _text(row['text']) for group in detail['rows'] for row in group
            if 105 <= row['box'][1] <= 135):
        return separate_inventory_support(bundle, detail)
    try:
        detail['customizations'] = resolve_detail_customizations(catalog, detail['card_id'], observations,
            allow_bound_support_projection=bundle is not None)
    except ValueError as error:
        # A complete body can omit effects such as upgradeHand. Only that
        # finite-count ambiguity may request the same detail's badge evidence;
        # weak text, wrong identities and unknown effects keep their rejection.
        if (badge_observer is None or str(error) !=
                f"clicked card {int(detail['card_id'])} must resolve exactly one positive customization count"):
            raise
        from .inventory_detail_badge import verify_inventory_detail_badge_count

        evidence = badge_observer()
        count = verify_inventory_detail_badge_count(evidence, frame_ids)
        detail['customizations'] = resolve_detail_customizations(catalog, detail['card_id'], observations,
            allow_bound_support_projection=bundle is not None, expected_count=count)
        detail['badge_count_evidence'] = evidence
    return separate_inventory_support(bundle, detail) if bundle is not None else detail


def _separate_spotlight_support(bundle, detail, titles):
    from .high_score.skill_support import supported_definition

    rows, frame_ids = detail.get('rows', ()), detail.get('frame_ids', ())
    if (bundle is None or detail.get('zone') != '手札' or type(detail.get('index')) is not int
            or detail['index'] < 0 or not valid_opening_frame_ids(frame_ids) or len(rows) != len(frame_ids)
            or any(len(group) != 1 or _text(group[0]['text']) != 'スポットライト++' for group in titles)):
        raise ValueError('295 support lacks complete current hand detail')
    # Two 295 variants may coexist. Select only the independently bound key
    # whose complete supported definition matches this body, never the first
    # canonical-ID match and never a newly inferred permanent customization.
    candidates = [key for key in bundle.inventory_keys if key[1] == '295'
                  and tuple((str(k), v) for k, v in key[2]) == SPOTLIGHT_PERMANENT]
    if len(candidates) != 1:
        raise ValueError('295 support lacks one verified permanent variant')
    target = [ref for ref in bundle.state.deck.all_cards if ref.instance_id == candidates[0][0]]
    if len(target) != 1 or target[0].card_id != '295@3=1,52=1':
        raise ValueError('295 support permanent instance changed')
    supported_definition(bundle.kernel.cards[target[0].card_id], enhancement_count=2)
    for group in rows:
        if any(type(row.get('score')) not in (int, float) or not .95 <= row['score'] <= 1
                or len(row.get('box', ())) != 4 or any(type(v) is not int for v in row['box'])
                or row['box'][0] < 190 or row['box'][1] < 105 or min(row['box'][2:]) <= 0
                or row['box'][0]+row['box'][2] > 700 or row['box'][1]+row['box'][3] > 382 for row in group):
            raise ValueError('295 support detail confidence or region is unresolved')
        body = _text(''.join(row['text'] for row in sorted(group, key=lambda row: (row['box'][1], row['box'][0]))))
        if normalize_score_effect_text(body) != SPOTLIGHT_BODY:
            raise ValueError('295 support complete effects differ from the observed tier')
    result = deepcopy(detail)
    result['customizations'] = dict(SPOTLIGHT_PERMANENT)
    result['temporary_support'] = dict(source=SPOTLIGHT_SOURCE,
        run_id=bundle.run_id, revision=bundle.revision, card_instance_id=candidates[0][0],
        permanent_customizations=dict(SPOTLIGHT_PERMANENT), observations=deepcopy(detail),
        enhancement_count=2, effects_verified=False, submission_authorized=False)
    return result


def separate_inventory_support(bundle, detail):
    """Revalidate raw text; never infer a new permanent level from support text.

    This preserves permanent identity and archives the complete current body.
    A separate current-instance verifier may bind its temporary effects;
    actual selected-title, payment and input checks remain mandatory.
    """
    rows = detail.get('rows', ())
    titles = [[row for row in group if 105 <= row['box'][1] <= 135]
              for group in rows]
    if not any('++' in _text(row['text']) for group in titles for row in group):
        return deepcopy(detail)
    if detail.get('card_id') == '295':
        return _separate_spotlight_support(bundle, detail, titles)
    frame_ids = detail.get('frame_ids', ())
    if (bundle is None or detail.get('zone') != '手札' or detail.get('card_id') != '597'
            or not valid_opening_frame_ids(frame_ids) or len(rows) != len(frame_ids)
            or any(len(group) != 1 or _text(group[0]['text']) != '破竹の勢い++' for group in titles)):
        raise ValueError('inventory temporary enhancement lacks a supported permanent binding')
    candidates = [row for row in bundle.inventory_keys if row[1] == '597']
    if (len(candidates) != 1 or tuple((str(k),v) for k,v in candidates[0][2]) != PERMANENT
            or detail.get('customizations') != {'14':1, '54':2}):
        raise ValueError('597 support differs from the verified permanent customization')
    for group in rows:
        if any(type(row.get('score')) not in (int,float) or not math.isfinite(row['score'])
                or not .95 <= row['score'] <= 1 or len(row['box']) != 4
                or any(type(v) is not int for v in row['box'])
                or row['box'][0] < 190 or row['box'][1] < 105
                or row['box'][2] <= 0 or row['box'][3] <= 0
                or row['box'][0]+row['box'][2] > 700 or row['box'][1]+row['box'][3] > 382 for row in group):
            raise ValueError('597 support detail confidence is unresolved')
        text = _text(''.join(row['text'] for row in sorted(group,key=lambda r:(r['box'][1],r['box'][0]))))
        if normalize_score_effect_text(text) != BODY:
            raise ValueError('597 support complete effects differ from the observed tier')
    result = deepcopy(detail)
    result['customizations'] = dict(PERMANENT)
    result['temporary_support'] = dict(source='hif_inventory_597_support_v1',
        run_id=bundle.run_id, revision=bundle.revision, card_instance_id=candidates[0][0],
        observed_effective_customizations=deepcopy(detail['customizations']),
        permanent_customizations=dict(PERMANENT), observations=deepcopy(detail),
        enhancement_count=2, effects_verified=False, submission_authorized=False)
    return result


def verify_inventory_support_row(bundle, row):
    proof = row.get('temporary_support')
    if proof is None:
        return True
    try:
        expected = separate_inventory_support(bundle, proof['observations'])
        return (proof == expected['temporary_support']
            and all(row.get(key) == expected[key] for key in ('zone','index','card_id','customizations')))
    except (ValueError,KeyError,TypeError,AttributeError,IndexError):
        return False


def verified_inventory_hand_support_fact(bundle, row):
    """Project the complete inventory body for its currently held instance only.

    The caller supplies the verified complete inventory's state and epoch.
    Original identity receipts remain unchanged and confer no input authority.
    """
    from .high_score.contracts import VerifiedHandSupport
    from .high_score.skill_support import supported_definition
    from .high_score.support_schedule import CARD_ID

    proof = row.get('temporary_support')
    if (not isinstance(proof, dict) or proof.get('source') not in ('hif_inventory_597_support_v1', SPOTLIGHT_SOURCE)
            or not verify_inventory_support_row(bundle, row)
            or (proof.get('run_id'), proof.get('revision')) != (bundle.run_id, bundle.revision)
            or bundle.state.terminal or bundle.state.pending_choice or bundle.state.resolution_stack):
        raise ValueError('current inventory support lacks its complete original body and epoch')
    held = [ref for ref in bundle.state.deck.hand if ref.instance_id == proof['card_instance_id']]
    card_id = '295@3=1,52=1' if proof['source'] == SPOTLIGHT_SOURCE else CARD_ID
    if len(held) != 1 or held[0].card_id != card_id:
        raise ValueError('current inventory support target is not the same held permanent instance')
    supported_definition(bundle.kernel.cards[held[0].card_id], enhancement_count=2)
    return VerifiedHandSupport(held[0].instance_id, held[0].card_id, bundle.state.turn, 2)


def _ordinary_plus_detail_body(definition):
    """Render only fully covered, registered semantics; unknown payloads reject.

    The catalog's zero-customization matcher alone does not cover schedule
    duration, usability or the once-per-stage footer. Compare their complete
    registered shape as well, without a card-name/body whitelist.
    """
    from .high_score.kernel import EffectKind, EffectSpec, ConditionKind, EffectCondition
    from .high_score.contracts import SchedulePhase

    condition = definition.condition
    if (condition is None or type(condition.threshold) is not int or condition.threshold < 0
            or condition != EffectCondition(ConditionKind.GOOD_CONDITION_MIN, threshold=condition.threshold)
            or definition.required_stance is not None or definition.persistent_effects
            or definition.limited is not True or not definition.effects):
        raise ValueError('ordinary plus detail has unsupported complete condition or lifecycle')
    body = f'{definition.name}好調が{condition.threshold}ターン以上の場合、使用可'
    for effect in definition.effects:
        if effect.kind is EffectKind.CONCENTRATION:
            amount = effect.amount
            if (type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0
                    or int(amount) != amount or effect != EffectSpec(effect.kind, amount)):
                raise ValueError('ordinary plus detail has unsupported scalar payload')
            body += f'集中+{int(amount)}'
        elif effect.kind is EffectKind.SCHEDULE:
            if len(effect.scheduled_effects) != 1:
                raise ValueError('ordinary plus detail has unsupported schedule body')
            child, duration = effect.scheduled_effects[0], effect.schedule_limit
            amount = child.amount
            if (type(duration) is not int or duration < 1
                    or type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0
                    or int(amount) != amount or child != EffectSpec(EffectKind.SCORE, amount)
                    or effect != EffectSpec(EffectKind.SCHEDULE, scheduled_effects=(child,),
                        schedule_phase=SchedulePhase.START_OF_TURN, schedule_limit=duration)):
                raise ValueError('ordinary plus detail has unsupported schedule payload')
            body += f'以降の{duration}ターンの間、ターン開始時、スコア+{int(amount)}'
        else:
            raise ValueError('ordinary plus detail has unsupported effect')
    return _text(body + '試験・ステージ中1回')


def observe_inventory_ordinary_plus(bundle, detail, *, catalog):
    """Bind complete current detail readings to one held base instance.

    This observation proves a registered single-plus body, not its origin or
    future recurrence. Normal cost is not present in these text rows: it must
    be unchanged by the projection and still requires the usual input checks.
    """
    from .high_score.kernel import CardCostKind
    from .high_score.hif_support_planning import ordinary_plus_definition

    rows, frame_ids = detail.get('rows', ()), detail.get('frame_ids', ())
    if (detail.get('zone') != '手札' or type(detail.get('index')) is not int or detail['index'] < 0
            or not valid_opening_frame_ids(frame_ids) or len(rows) != len(frame_ids)
            or (detail.get('run_id'), detail.get('revision'), detail.get('observed_turn'))
                != (bundle.run_id, bundle.revision, bundle.state.turn)
            or bundle.state.terminal or bundle.state.pending_choice or bundle.state.resolution_stack):
        raise ValueError('ordinary plus detail lacks a complete current hand epoch')
    card_id = detail.get('card_id')
    keys = [key for key in bundle.inventory_keys if key[1] == card_id]
    refs = [ref for ref in bundle.state.deck.all_cards if ref.card_id == card_id]
    if (type(card_id) is not str or not card_id.isdecimal() or len(keys) != 1 or keys[0][2]
            or len(refs) != 1 or refs[0].instance_id != keys[0][0] or refs[0].upgraded
            or [ref for ref in bundle.state.deck.hand if ref.instance_id == keys[0][0]] != refs):
        raise ValueError('ordinary plus detail lacks a unique uncustomized held base instance')
    ordinary = bundle.kernel.cards.get(card_id)
    projected = ordinary_plus_definition(bundle.kernel.cards, ordinary)
    visible = bundle.kernel.cards[ordinary.upgrade_card_id]
    if (ordinary.base_card_id != card_id or visible.name != ordinary.name + '+'
            or catalog.skill_card_title(int(card_id)) != ordinary.name
            or catalog.skill_card_title(int(visible.card_id)) != visible.name
            or ordinary.cost_kind is not CardCostKind.NORMAL or projected.cost_kind is not CardCostKind.NORMAL
            or ordinary.full_power_cost or projected.full_power_cost
            or ordinary.stamina_cost != projected.stamina_cost):
        raise ValueError('ordinary plus detail catalog or unobserved cost changed')
    expected = _ordinary_plus_detail_body(projected)
    for group in rows:
        if not group or any(
                type(row.get('text')) is not str or type(row.get('score')) not in (int, float)
                or not math.isfinite(row['score']) or not .95 <= row['score'] <= 1
                or len(row.get('box', ())) != 4 or any(type(v) is not int for v in row['box'])
                or row['box'][0] < 190 or row['box'][1] < 105 or min(row['box'][2:]) <= 0
                or row['box'][0]+row['box'][2] > 700 or row['box'][1]+row['box'][3] > 382 for row in group):
            raise ValueError('ordinary plus detail confidence or complete region is unresolved')
        titles = [row for row in group if 105 <= row['box'][1] <= 135]
        if len(titles) != 1 or _text(titles[0]['text']) != _text(visible.name):
            raise ValueError('ordinary plus detail complete title changed')
        text = _text(''.join(row['text'] for row in sorted(group, key=lambda row: (row['box'][1], row['box'][0]))))
        if normalize_score_effect_text(text) != expected:
            raise ValueError('ordinary plus detail complete registered effects changed')
        catalog.confirm_clicked_skill_card(text, predicted_card_id=int(visible.card_id))
        if catalog.resolve_effective_customizations(int(visible.card_id), text,
                expected_count=0, allow_positive_completion=False) != {}:
            raise ValueError('ordinary plus detail does not match zero customization')
    return dict(source='hif_inventory_ordinary_plus_v1', run_id=bundle.run_id,
        revision=bundle.revision, observed_turn=bundle.state.turn, card_instance_id=refs[0].instance_id,
        card_id=card_id, visible_card_id=visible.card_id, enhancement_count=1,
        projection_kind='ordinary_plus_estimate', effects_verified=True, cost_observed=False,
        submission_authorized=False, observations=deepcopy(detail))


def verify_inventory_ordinary_plus(bundle, receipt, *, catalog):
    """Rebuild the full receipt, returning only a current-hand planning fact."""
    from .high_score.contracts import VerifiedHandSupport

    expected = observe_inventory_ordinary_plus(bundle, receipt.get('observations', {}), catalog=catalog)
    if receipt != expected:
        raise ValueError('ordinary plus detail receipt changed')
    return VerifiedHandSupport(expected['card_instance_id'], expected['card_id'],
        expected['observed_turn'], 1, projection_kind='ordinary_plus_estimate')
