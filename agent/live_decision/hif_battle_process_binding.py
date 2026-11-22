"""Separate original battle accounting identity from the current game process."""
from copy import deepcopy


def migrated_identity(original, migrations):
    """Read an already published process chain without rewriting old grants."""
    if not isinstance(migrations, list):
        raise ValueError('battle process migrations must retain their ordered history')
    current = dict(original)
    for item in migrations:
        handoff = item.get('handoff', {})
        before, after = item.get('from_game', {}), item.get('to_game', {})
        proof = item.get('process_recovery', {})
        expected_owner = dict(owner='task275', phase='battle', game_pid=current['game_pid'],
            revision=current['ownership_revision'])
        if (item.get('schema_version') != 1 or item.get('run_id') != current['route_run_id']
                or item.get('from_ownership') != expected_owner
                or before.get('pid') != current['game_pid'] or before.get('pid') == after.get('pid')
                or type(after.get('pid')) is not int or after['pid'] <= 0
                or type(after.get('created_at_100ns')) is not int or after['created_at_100ns'] <= 0
                or item.get('current_task', {}).get('game') != after
                or any(handoff.get(key) != current[key] for key in (
                    'handoff_id', 'ledger_revision', 'policy_profile'))
                or proof.get('old_pid') != before['pid'] or proof.get('new_pid') != after['pid']
                or proof.get('scope') not in ('pending_card_restart_rule_history_v1',
                    'pending_card_restart_turn_history_v1')
                or proof.get('zero_new_business_inputs') is not True
                or proof.get('logical_context') != {key: value for key, value in
                    handoff.get('live_context', {}).items() if key != 'round_number'}):
            raise ValueError('battle process migration differs from the original handoff')
        current.update(game_pid=after['pid'], ownership_revision=current['ownership_revision'] + 1)
    return current


def battle_process_migrations(record):
    """Return this handoff's entries from the existing owned budget history."""
    from route.host import windows_route_clock
    from route.cultivation import _budget_container
    from route.continuation import execution_process_migrations

    budget = (_budget_container(record['handoff']) or {}).get('automatic_cultivation')
    if not isinstance(budget, dict) or not budget.get('process_migrations'):
        return []
    migrations = execution_process_migrations(budget, now=windows_route_clock().now())
    return [item for item in migrations
        if item['handoff']['handoff_id'] == record['handoff']['handoff_id']]


def accounting_identity(record, current):
    migrations = battle_process_migrations(record)
    if not migrations:
        return current
    first = migrations[0]
    original = dict(current, game_pid=first['from_game']['pid'],
        ownership_revision=first['from_ownership']['revision'])
    if migrated_identity(original, migrations) != current:
        raise ValueError('battle process chain does not reach the current owner')
    return original


def process_fields(record):
    migrations = battle_process_migrations(record)
    return {'process_migrations': deepcopy(migrations)} if migrations else {}
