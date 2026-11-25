"""Hiro's inventory, formation planner and writer on the active Maa task."""

from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import cache
from contextlib import ExitStack, contextmanager
from importlib.util import module_from_spec, spec_from_file_location

from agent.live_decision.hiro_runtime import hiro_module, hiro_runtime_root

from .hiro_cultivation import _replace


def auto_formation_selected(context):
    node = context.get_node_data('ProduceHifAutoFormationSettings') or {}
    return (node.get('attach') or {}).get('auto') is True


@cache
def _support_inventory_tool():
    path = hiro_runtime_root() / 'tools/support_inventory.py'
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
    memory = hiro_module('vision.memory_icons')
    index = hiro_runtime_root() / 'assets/pack/support/index.json'
    pack = index.parents[1]
    if (not index.is_file() or not any((pack / 'support').glob('*.png'))
            or not any((pack / 'cards').glob('*.png'))):
        raise ValueError('自动编成缺少 Hiro 支援卡或回忆卡图，可改用游戏内保存的套组。')
    with ExitStack() as scope:
        scope.enter_context(_replace(assets, 'support_dir', lambda: pack / 'support'))
        scope.enter_context(_replace(assets, 'card_art_dir', lambda: pack / 'cards'))
        # Keep the renderer's optional cache outside the installed program.
        scratch = scope.enter_context(TemporaryDirectory(prefix='gkh-hiro-formation-'))
        scope.enter_context(_replace(memory, 'ART_CACHE', str(Path(scratch) / 'memory-icons.npz')))
        support._library.cache_clear()
        memory._art_index.cache_clear()
        try:
            yield Path(scratch)
        finally:
            support._library.cache_clear()
            memory._art_index.cache_clear()


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


def build_qualifier_formation(rt, skin, formation, report, directory):
    """Scan from the formation menu, write the chosen presets and return home."""
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
    menu.open()
    menu.ensure_mode('hif')
    menu.ensure_tab('support')
    menu.ensure_set(formation['support_set'])
    x, y = screens.FORMATION_TILES[0]
    click(x + screens.TILE_W // 2, y + 59)
    supports = scan_support_inventory(dev)
    menu.ensure_tab('memory')
    menu.ensure_set(formation['memory_set'])
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
    plan = hiro_module('brain.formation_plan').build(skin, support_inventory=supports,
        memory_inventory=memories,
        support_set=formation['support_set'], memory_set=formation['memory_set'],
        main_attr=formation['main_attr'], fixed_main=True)
    for kind in ('support', 'memory'):
        if plan[kind].get('error'):
            raise ValueError(f"无法生成自动编成：{plan[kind]['error']}")
    report['formation_plan'] = plan
    report['formation_changes'] = menu_module.run(dev, plan, 'hif', make_picker=make_picker,
        read_formation=screens.read_support_formation, read_list=screens.read_support_list)
    nav.goto_home(rt)
    return plan
