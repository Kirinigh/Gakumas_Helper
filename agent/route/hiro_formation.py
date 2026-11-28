"""Hiro's inventory, formation planner and writer on the active Maa task."""

from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import cache
from contextlib import ExitStack, contextmanager
from importlib.util import module_from_spec, spec_from_file_location

from agent.live_decision.hiro_runtime import hiro_module, hiro_runtime_root

from .hiro_cultivation import _replace, _hiro_memory_indexes


def auto_formation_selected(context):
    node = context.get_node_data('ProduceHifAutoFormationSettings') or {}
    return (node.get('attach') or {}).get('auto') is True


def _formation_file(relative):
    return hiro_runtime_root() / relative


@cache
def _support_inventory_tool():
    path = _formation_file(Path('tools/support_inventory.py'))
    if not path.is_file():
        raise ValueError('自动编成缺少固定 Hiro 支援卡库存扫描脚本。')
    spec = spec_from_file_location('_gkh_pinned_hiro_support_inventory', path)
    tool = module_from_spec(spec)
    spec.loader.exec_module(tool)
    tool.print = lambda *_args, **_kwargs: None
    return tool


@contextmanager
def formation_assets():
    _support_inventory_tool()
    assets = hiro_module('vision.assets')
    support = hiro_module('vision.support')
    index = _formation_file(Path('assets/pack/support/index.json'))
    pack = index.parents[1]
    if not index.is_file() or not any((pack / 'support').glob('*.png')):
        raise ValueError('自动编成缺少 Hiro 支援卡图，可改用游戏内保存的套组。')
    with ExitStack() as scope:
        scope.enter_context(_replace(assets, 'support_dir', lambda: pack / 'support'))
        scratch = scope.enter_context(TemporaryDirectory(prefix='gkh-hiro-formation-'))
        # The packaged Hiro reader can resolve memory icons from prebuilt
        # indexes even when the original card-art directory is absent.
        scope.enter_context(_hiro_memory_indexes())
        support._library.cache_clear()
        try:
            yield Path(scratch)
        finally:
            support._library.cache_clear()


def _wait_support_page(dev, read):
    menu = hiro_module('produce.formation_menu')
    for _ in range(menu.WAIT_TRIES):
        page = read(dev.shoot())
        if page is not None:
            return page
        dev.sleep(menu.WAIT_GAP)
    raise ValueError('未到达预期的支援卡库存页。')


def scan_support_inventory(dev):
    """Bind the upstream read-only scan to the active Maa screen and clock."""
    screens = hiro_module('vision.support_screens')
    first = _wait_support_page(dev, screens.read_support_list)
    if first['page'] != 1:
        first = hiro_module('produce.formation_writer')._list_to_top(dev, screens.read_support_list)
    if first['page'] != 1:
        raise ValueError('支援卡库存未回到第 1 页。')
    tool = _support_inventory_tool()
    device = SimpleNamespace(screenshot=dev.shoot, tap=dev.click, swipe=dev.swipe)
    with (
        _replace(hiro_module('tool_device'), 'open_tool_device', lambda *_a, **_kw: device),
        _replace(tool, 'time', SimpleNamespace(sleep=dev.sleep)),
    ):
        inventory = tool.aggregate_list(tool.list_scan())
    # The planner defaults unknown levels to 1. Use only observed levels.
    known = {}
    for card, row in inventory['cards'].items():
        if type(row['level']) is not int or row['level'] < 1:
            continue
        known[card] = row
        rank = row['breaks']
        if type(rank) is int and 1 <= rank <= 4:
            row['limit_rank'] = f'SupportCardLevelLimitRank__{rank}'
    return dict(inventory, cards=known)


def _qualifier_slots(dev, rt, read_formation, consent, *, mode='hif', previous_slots=None):
    kinds = tuple(kind for kind in ('support', 'memory') if kind not in consent)
    slots = dict(consent)
    blank_only = []
    if kinds:
        survey = hiro_module('produce.formation_survey').survey(dev, mode, kinds=kinds,
            read_formation=read_formation, stop_at_blank=True)
        for kind in kinds:
            empty = [row['set'] for row in survey['sets'] if row['kind'] == kind and row['filled'] == 0]
            if empty:
                slots[kind] = min(empty)
                blank_only.append(kind)
                continue
            if previous_slots is not None:
                rows = [row for row in survey['sets'] if row['filled'] not in (None, 0)]
                written = hiro_module('cli.cmd_run').own_full_slots(previous_slots, mode, kind, rows)
                if written:
                    slots[kind] = written[0]
                    continue
            hiro_module('runtime.nav').goto_home(rt)
            raise ValueError(f'{kind} 编成没有可用空位。请腾出空位或在任务中指定允许覆盖的编号。')
    return slots, tuple(blank_only)


def build_qualifier_formation(rt, skin, formation, report, directory, *, consent=None,
                             mode='hif', produce=None, reuse_written=False):
    """Scan from the formation menu, write the chosen presets and return home."""
    from utils.local_data import DEFAULT_RUNTIME_ROOT

    menu_module = hiro_module('produce.formation_menu')
    writer = hiro_module('produce.formation_writer')
    picker = hiro_module('produce.memory_picker')
    screens = hiro_module('vision.support_screens')
    nav = hiro_module('runtime.nav')
    read_ocr = hiro_module('produce.ops').ocr_boxes
    shoot = lambda: rt.screen.shot().img
    click = rt.device.tap

    def swipe(x1, y1, x2, y2, duration=None, *, hold=0.):
        rt.device.swipe(x1, y1, x2, y2, duration=duration or .3, hold=hold)

    dev = writer.Device(shoot=shoot, click=click, swipe=swipe, sleep=rt.sleep, ocr=read_ocr)

    def make_picker():
        return picker.Picker(shoot, click,
            lambda *points: swipe(*points, duration=picker.SWIPE_DURATION, hold=picker.SWIPE_HOLD),
            ocr=read_ocr, clock=rt.clock, sleep=rt.sleep)

    menu = menu_module.Menu(dev)
    nav.goto_home(rt)
    slot_root = DEFAULT_RUNTIME_ROOT / 'hiro-formation'
    slots, kinds = _qualifier_slots(dev, rt, screens.read_support_formation, consent or {}, mode=mode,
                                    previous_slots=slot_root if reuse_written else None)
    menu.open()
    menu.ensure_mode(mode)
    menu.ensure_tab('support')
    menu.ensure_set(slots['support'])
    x, y = screens.FORMATION_TILES[0]
    click(x + screens.TILE_W // 2, y + 59)
    supports = scan_support_inventory(dev)
    menu.ensure_tab('memory')
    menu.ensure_set(slots['memory'])
    click(*menu_module.MEMORY_TILES[0])
    chosen = make_picker()
    for _ in range(menu_module.WAIT_TRIES):
        if picker.picker_open(read_ocr(shoot())):
            break
        rt.sleep(menu_module.WAIT_GAP)
    memories = picker.scan(chosen, directory / 'memory-inventory.json')
    report['formation_inventory'] = {'supports': len(supports['cards']), 'memories': len(memories['entries']),
                                     'memory_complete': memories['complete']}
    menu.wait_menu(why='回忆扫描后返回编成')
    if not memories['complete']:
        raise ValueError('回忆库存扫描未完成，未提交自动编成。')
    planner = hiro_module('brain.formation_plan')
    plan = planner.build(skin, support_inventory=supports,
        memory_inventory=memories,
        support_set=slots['support'], memory_set=slots['memory'],
        main_attr=formation['main_attr'], fixed_main=True, mode=produce)
    for kind in ('support', 'memory'):
        if plan[kind].get('error'):
            raise ValueError(f"无法生成自动编成：{plan[kind]['error']}")
    report['formation_plan'] = plan
    changes = menu_module.run(dev, plan, mode, make_picker=make_picker,
        read_formation=screens.read_support_formation, read_list=screens.read_support_list, blank_only=kinds)
    if written := {kind: slots[kind] for kind in slots if changes.get(kind)}:
        planner.record_slots(slot_root, mode, written)
    report['formation_changes'] = changes
    nav.goto_home(rt)
    return plan
