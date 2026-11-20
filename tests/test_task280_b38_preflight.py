from tools.task280_b38_preflight import _form_gaps, _identity_by_name


def test_b38_inventory_keeps_actual_tier_and_resolves_ordinary_plus_for_planning():
    observation = {
        'hand': [{'id': 'spotlight', 'u': 2}],
        'deck': [{'id': 'spotlight', 'u': 1}, {'id': 'counter', 'u': 1}],
        'grave': [], 'hold': [], 'lost': [],
    }
    master = {('spotlight', 1): {'name': 'spotlight+'}, ('spotlight', 2): {'name': 'spotlight++'},
              ('counter', 1): {'name': 'counter+'}}
    actual = list(_form_gaps(observation, {('spotlight', 1)}, master,
                             {'spotlight+': ['762'], 'counter+': ['585']}, {'762'}))
    assert actual == [
        {'id': 'counter', 'upgrade': 1, 'planning_upgrade': 1, 'local_id': '585', 'status': 'effect_not_compiled'},
        {'id': 'spotlight', 'upgrade': 2, 'planning_upgrade': 1, 'local_id': '762', 'status': 'catalog_present'},
    ]


def test_partial_catalog_match_is_not_reported_as_unique():
    assert _identity_by_name('shared', {'shared': ['1', '2']}, {'1'}) == (None, 'no_unique_catalog_name')
    assert _identity_by_name('known', {'known': ['1']}, {'1'}) == ('1', 'catalog_present')
