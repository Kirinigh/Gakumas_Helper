"""Check the public AS-B38CARVV opening against the unchanged local rule catalog."""
from __future__ import annotations

import re
import csv
import json
import argparse
from pathlib import Path

from tools.task280_gkbench_hosted import hosted_request, planning_upgrade_level
from agent.live_decision.high_score.contracts import PlanType
from agent.live_decision.high_score.catalog_v2 import (
    compile_item_catalog,
    compile_drink_catalog,
    compile_skill_card_catalog,
)

PROBLEM = 'AS-B38CARVV'
ITEMS = {'pitem_01-3-157-1': '285'}


def _master(path, name):
    return json.loads((path / (name + '.json')).read_text(encoding='utf-8'))


def _local_names(path):
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row['name'], []).append(row['id'])
    return result


def _identity_by_name(name, names, compiled):
    choices = names.get(name, ())
    if len(choices) != 1:
        return None, 'no_unique_catalog_name'
    item_id = choices[0]
    return item_id, 'catalog_present' if item_id in compiled else 'effect_not_compiled'


def _form_gaps(observation, known_forms, card_master, names, compiled):
    forms = {(card['id'], card['u']) for zone in ('hand', 'deck', 'grave', 'hold', 'lost')
             for card in observation[zone]}
    for form in sorted(forms - known_forms):
        planning_level = planning_upgrade_level(form[1])
        row = card_master.get((form[0], planning_level))
        local_id, status = ((None, 'master_form_unknown') if row is None
                            else _identity_by_name(row['name'], names, compiled))
        yield dict(id=form[0], upgrade=form[1], planning_upgrade=planning_level, local_id=local_id, status=status)


def check_opening(observation, loadout, *, assets, master, known_card_forms=(), known_drink_ids=()):
    if (observation['v'], observation['exam']['plan'], observation['i']) != ('uiobs.v2', 2, 0):
        raise ValueError('as_b38_verified_opening_required')
    skill_path, drinks_path = assets / 'skill_cards.csv', assets / 'p_drinks.csv'
    cards = compile_skill_card_catalog(skill_path, plan_type=PlanType.SENSE, include_free_card_ids=('585',))
    drinks = compile_drink_catalog(drinks_path, plan_type=PlanType.SENSE)
    items = compile_item_catalog(assets / 'p_items.csv', plan_type=PlanType.SENSE)
    card_names, drink_names = _local_names(skill_path), _local_names(drinks_path)
    card_master = {(row['id'], row.get('upgradeCount', 0)): row for row in _master(master, 'ProduceCard')}
    drink_master = {row['id']: row for row in _master(master, 'ProduceDrink')}
    card_results = list(_form_gaps(observation, set(known_card_forms), card_master, card_names, cards.cards))
    drink_results = []
    for raw_id in sorted(set(observation['drinks']) - set(known_drink_ids)):
        row = drink_master.get(raw_id)
        local_id, state = ((None, 'master_drink_unknown') if row is None
                           else _identity_by_name(row['name'], drink_names, drinks.drinks))
        drink_results.append(dict(id=raw_id, local_id=local_id, status=state))
    item_results = [dict(id=raw_id, local_id=local_id, status='effect_not_compiled')
                    for raw_id, local_id in ITEMS.items()
                    if raw_id in loadout['produce_item_ids'] and local_id not in items.items]
    return dict(problem=PROBLEM, slot=0, rule=65, action_submissions=0,
                card_mapping_candidates=card_results, drink_mapping_candidates=drink_results,
                item_gaps=item_results, support_count=len(loadout['support_cards']),
                observed_enchants=len(observation['ench']),
                ready_for_strong=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation', required=True, help='Your completed UINet 2.1 benchmark evaluation ID')
    parser.add_argument('--assets', type=Path, required=True, help='Local benchmark CSV directory')
    parser.add_argument('--master', type=Path, required=True, help='Local master-data directory')
    parser.add_argument('--known-mappings', type=Path,
                        help='Local JSON with known_card_forms and known_drink_ids')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('a new evidence path is required')
    if not re.fullmatch(r'[a-f0-9]{12}', args.evaluation):
        parser.error('invalid evaluation identifier')
    known = {}
    if args.known_mappings:
        data = json.loads(args.known_mappings.read_text(encoding='utf-8'))
        forms, drinks = data['known_card_forms'], data['known_drink_ids']
        if (not isinstance(forms, list) or any(not isinstance(form, list) or len(form) != 2
                or not isinstance(form[0], str) or not form[0]
                or type(form[1]) is not int or form[1] < 0 for form in forms)
                or not isinstance(drinks, list) or any(not isinstance(drink, str) or not drink for drink in drinks)):
            parser.error('invalid known mapping shapes')
        known = dict(known_card_forms=tuple(map(tuple, forms)), known_drink_ids=tuple(drinks))
    request = hosted_request(args.token_file)
    if request('/api/me')['engine']['rules_version'] != 65:
        raise ValueError('referee_rule_changed')
    evaluation = request('/api/evals/' + args.evaluation)
    run = request('/api/runs/' + evaluation['run'])
    if run['engine']['rules_version'] != 65:
        raise ValueError('comparison_rule_changed')
    reference = next(row for row in evaluation['done'] if row['id'] == PROBLEM and row['slot'] == 0)
    result = request('/api/runs/' + evaluation['run'] + '/episodes/' + PROBLEM + '/0')
    if (result['id'] != PROBLEM or result['seed_index'] != 0 or result['seed_split'] != 'template'
            or result['final']['finished'] is not True or result['final']['score'] != reference['score']):
        raise ValueError('comparison_receipt_mismatch')
    problem = next(row for row in request('/api/sets/hif/problems') if row['id'] == PROBLEM and row['core'])
    report = check_opening(result['steps'][0]['obs'], problem['archive']['loadout'],
                           assets=args.assets, master=args.master, **known)
    report.update(comparison_evaluation=args.evaluation, uinet21_final=reference['score'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
