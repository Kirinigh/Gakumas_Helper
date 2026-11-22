"""Cursor for hiro's acquisition-ordered qualifier memory search."""

MEMORY_SEARCH_CONTRACTS = {
    'start_memory_change': ('start_memory_mismatch', ('start_memory_open_sort',)),
    'start_memory_open_sort': ('start_memory_open_sort', ('start_memory_choose_sort',)),
    'start_memory_choose_sort': ('start_memory_choose_sort', ('start_memory_apply_sort',)),
    'start_memory_apply_sort': ('start_memory_apply_sort', ('start_memory_select_cell',)),
    'start_memory_select_cell': ('start_memory_select_cell',
        ('start_memory_select_cell', 'start_memory_scroll', 'start_memory_close')),
    'start_memory_scroll': ('start_memory_scroll', ('start_memory_select_cell',)),
    'start_memory_ok': ('start_memory_close', ('start_confirmation', 'start_memory_search_stopped')),
}
MEMORY_SEARCH_SCENES = frozenset({source for source, _ in MEMORY_SEARCH_CONTRACTS.values()}
    | {'start_memory_search_stopped'})
MEMORY_SETTLE_SECONDS = {
    'start_memory_change': 1.5, 'start_memory_open_sort': 1.0,
    'start_memory_choose_sort': .5, 'start_memory_apply_sort': 1.2,
    'start_memory_select_cell': .6, 'start_memory_scroll': 1.0, 'start_memory_ok': 2.0,
}


def search_target_key(kind, page):
    if kind in ('start_memory_select_cell', 'start_memory_scroll', 'start_memory_ok'):
        return f'{kind}:{page["memory_page"]}:{page["memory_cell"]}'
    return kind


def _cursor(target):
    kind, page, cell = target.key.split(':')
    page, cell = int(page), int(cell)
    if kind != target.kind or not 0 <= page < 12 or not 0 <= cell < 9:
        raise ValueError('invalid saved memory search position')
    return page, cell


def search_position(host):
    """Read the source action from its original gate even after process replacement."""
    gate = host.hooks.transition_gate
    target = gate.pending.target if gate.pending else None
    if target is not None and target.kind in MEMORY_SEARCH_CONTRACTS:
        if target.kind == 'start_memory_select_cell':
            page, cell = _cursor(target)
            return dict(stage='inspect_cell', page=page, cell=cell)
        if target.kind == 'start_memory_scroll':
            page, _ = _cursor(target)
            return dict(stage='select_cell', page=page + 1, cell=0)
        stage = {'start_memory_change': 'open_sort', 'start_memory_open_sort': 'choose_sort',
            'start_memory_choose_sort': 'apply_sort', 'start_memory_apply_sort': 'select_cell'}.get(target.kind)
        if stage is not None:
            return dict(stage=stage, page=0, cell=0)
    return (host.handoff or {}).get('start_preparation', {}).get('memory_search')


def search_closed(host):
    pending = host.hooks.transition_gate.pending
    if pending is not None and pending.target.kind == 'start_memory_ok':
        return True
    return (host.handoff or {}).get('start_preparation', {}).get('memory_search_closed') is True


def remember_search_page(host, page):
    state = host.handoff['start_preparation']
    if page['phase'] in MEMORY_SEARCH_SCENES - {'start_memory_mismatch', 'start_memory_search_stopped'}:
        # Persist the position being *read*. An interrupted inspection must re-read
        # the selected card before it can issue the proposed next cell or OK.
        state['memory_search'] = search_position(host)
    if page['phase'] == 'start_memory_search_stopped' and search_closed(host):
        state['memory_search_closed'] = True
    elif page['phase'] == 'start_confirmation':
        state.pop('memory_search', None)
        state.pop('memory_search_closed', None)


def search_destination(pending, observation, page):
    kind = pending.target.kind
    if kind not in MEMORY_SEARCH_CONTRACTS:
        return True
    if page is None or page.get('phase') != observation.scene or observation.scene not in pending.expected_scenes:
        return False
    if kind == 'start_memory_ok':
        return observation.scene in ('start_confirmation', 'start_memory_search_stopped')
    if kind in ('start_memory_select_cell', 'start_memory_scroll'):
        source_page, source_cell = _cursor(pending.target)
        where = page.get('memory_page'), page.get('memory_cell')
        if kind == 'start_memory_scroll':
            return where == (source_page + 1, 0)
        if observation.scene == 'start_memory_select_cell':
            return source_cell < 8 and where == (source_page, source_cell + 1)
        return where == (source_page, source_cell)
    return True
