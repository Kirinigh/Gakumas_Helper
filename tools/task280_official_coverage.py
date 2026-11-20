"""Read-only inventory and rule-catalog coverage of the official HIF set."""

from __future__ import annotations

import re
import csv
import json
import argparse
from pathlib import Path
from collections import Counter

from tools.task280_shared_rules import shared_card_candidates
from tools.task280_gkbench_hosted import hosted_request, planning_upgrade_level
from agent.live_decision.high_score.contracts import PlanType
from agent.live_decision.high_score.catalog_v2 import (
    compile_item_catalog,
    compile_drink_catalog,
    compile_skill_card_catalog,
)

_PLAN_TYPES = {'PLAN1': PlanType.SENSE, 'PLAN2': PlanType.LOGIC, 'PLAN3': PlanType.ANOMALY}


def card_form(entry):
    """Read identity only; keep the original full loadout with the referee."""
    if isinstance(entry, str):
        head = entry.split('/', 1)[0]
        match = re.fullmatch(r'(p_card-[\w-]+)(?:\+(\d+))?', head)
        if not match:
            raise ValueError('invalid_compact_card_form')
        return match[1], int(match[2] or 0), '/' in entry
    return (entry['produce_card_id'], entry.get('current_upgrade_count', 0),
            bool(entry.get('customizes', ())))


def _csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _rows_by_name(rows):
    names = {}
    for row in rows:
        names.setdefault(row['name'], []).append(row['id'])
    return names


def _master_rows(master, name):
    return json.loads((master / (name + '.json')).read_text(encoding='utf-8'))


def _gap(identity, master, local, compiled):
    if identity not in master:
        return 'master_missing'
    matches = local.get(master[identity], ())
    if len(matches) != 1:
        return 'local_identity_unresolved'
    if matches[0] not in compiled:
        return 'local_effect_not_compiled'
    return None


def verified_produce_only_items(public_items, ris_items):
    """An exact, unique title match must prove the RIS produce-only scope."""
    by_name = {}
    public_counts = Counter(row['name'] for row in public_items.values())
    for row in ris_items:
        by_name.setdefault(row['name'], []).append(row)
    verified = set()
    for item, raw in public_items.items():
        matches = by_name.get(raw['name'], ())
        if public_counts[raw['name']] == 1 and len(matches) == 1:
            candidate, = matches
            if (candidate['mode'] == 'produce' and not candidate['effects'].strip()
                    and candidate['rarity'] == raw.get('rarity')):
                verified.add(item)
    return verified


def audit(rows, *, assets, master, produce_only_items=()):
    card_rows = _csv_rows(assets / 'skill_cards.csv')
    master_cards = _master_rows(master, 'ProduceCard')
    shared_cards = shared_card_candidates(master_cards, card_rows)
    item_rows = _master_rows(master, 'ProduceItem')
    item_ids = {row['id'] for row in item_rows}
    excluded_produce = set(produce_only_items) - item_ids
    metadata = {
        'card': ({(row['id'], row.get('upgradeCount', 0)): row['name']
                  for row in master_cards}, _rows_by_name(card_rows)),
        'drink': ({row['id']: row['name'] for row in _master_rows(master, 'ProduceDrink')},
                  _rows_by_name(_csv_rows(assets / 'p_drinks.csv'))),
        'item': ({row['id']: row['name'] for row in item_rows if row['isExamEffect']},
                 _rows_by_name(_csv_rows(assets / 'p_items.csv'))),
    }
    include_free = tuple(row['id'] for row in card_rows if row['plan'] == 'free')
    compiled = {
        scene: {
            'card': compile_skill_card_catalog(assets / 'skill_cards.csv', plan_type=plan,
                                               include_free_card_ids=include_free).cards,
            'drink': compile_drink_catalog(assets / 'p_drinks.csv', plan_type=plan).drinks,
            'item': compile_item_catalog(assets / 'p_items.csv', plan_type=plan).items,
        }
        for scene, plan in _PLAN_TYPES.items()
    }
    rows_out = []
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('duplicate_official_problem')
        seen.add(row['id'])
        archive = row['archive']
        loadout = archive['loadout']
        entry = dict(id=row['id'], core=bool(row['core']), plan=row['scene']['plan'])
        if family := archive.get('meta', {}).get('family'):
            entry['family'] = family
        if 'seed' in loadout or archive.get('prefix'):
            raise ValueError('unexpected_seed_or_prefix_on_template')
        if loadout['schema'] != 'gkms.hif_loadout.v1':
            raise ValueError('unexpected_loadout_schema')
        problems, candidates, extras = [], [], Counter()
        items = set(loadout.get('produce_item_ids', ()))
        groups = {'card': [], 'drink': loadout.get('drink_ids', ()),
                  'item': items & metadata['item'][0].keys()}
        for spec in loadout['deck_cards']:
            raw_id, tier, customized = card_form(spec)
            groups['card'].append((raw_id, planning_upgrade_level(tier)))
            extras['customized_cards'] += int(customized)
            extras['high_tier_cards'] += int(tier > 1)
        for kind, identities in groups.items():
            for identity in sorted(set(identities)):
                if reason := _gap(identity, *metadata[kind], compiled[entry['plan']][kind]):
                    problems.append(dict(kind=kind, identity=identity, reason=reason))
                    if kind == 'card' and identity in shared_cards and reason == 'local_effect_not_compiled':
                        candidates.append(dict(identity=identity, local_id=shared_cards[identity].card_id))
        unknown_items = items - item_ids - excluded_produce
        problems += [dict(kind='item', identity=item, reason='master_missing') for item in sorted(unknown_items)]
        extras['support_cards'] = len(loadout.get('support_cards', ()))
        extras['opening_enchants'] = sum(len(loadout.get(field, ())) for field in
                                         ('main_item_enchant_ids', 'lesson_start_enchant_ids', 'memory_ability_ids'))
        knowledge_status = 'catalog_gap' if problems else 'catalog_identity_only'
        status = knowledge_status if entry['plan'] == 'PLAN1' else 'current_strong_sense_only'
        rows_out.append(entry | dict(status=status, knowledge_status=knowledge_status,
                                     gaps=problems, produce_only_beyond_master=sorted(items & excluded_produce),
                                     shared_rule_candidates=candidates, features=dict(extras)))
    return dict(set='hif', scope='official_public_template_coverage_only', submitted_actions=0,
                verified_strong_openings=0, rows=rows_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True, help='Local benchmark CSV directory')
    parser.add_argument('--master', type=Path, required=True, help='Local master-data directory')
    parser.add_argument('--ris-items', type=Path, help='P-item table from the fixed RIS reference')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('a new evidence path is required')
    request = hosted_request(args.token_file)
    engine = request('/api/me')['engine']
    collection = next(row for row in request('/api/sets') if row['set'] == 'hif')
    produce_only = (verified_produce_only_items(request('/api/catalog')['items'], _csv_rows(args.ris_items))
                    if args.ris_items else ())
    report = audit(request('/api/sets/hif/problems'), assets=args.assets, master=args.master,
                   produce_only_items=produce_only)
    rows = report['rows']
    core = [row for row in rows if row['core']]
    if (len(rows) != collection['problems'] or {row['id'] for row in core} != set(collection['core'])):
        raise ValueError('official_collection_changed_during_read')
    report.update(engine=engine, problems=len(rows), core=len(core), core_seeds=collection['core_seeds'],
                  shared_candidate_problems=sum(bool(row['shared_rule_candidates']) for row in rows),
                  shared_candidate_core=sum(bool(row['shared_rule_candidates']) for row in core),
                  catalog_gaps_by_plan={plan: sum(row['plan'] == plan and row['knowledge_status'] == 'catalog_gap'
                                                  for row in rows) for plan in _PLAN_TYPES},
                  produce_only_beyond_master_problems=sum(bool(row['produce_only_beyond_master']) for row in rows),
                  all_plans=dict(Counter(row['plan'] for row in rows)),
                  core_status=dict(Counter(row['status'] for row in core)),
                  all_status=dict(Counter(row['status'] for row in rows)))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
