"""Attach Maa's actual AP dispatch receipt to Hiro's original mode opening."""

import re
import unicodedata
from contextlib import ExitStack

from agent.live_decision.hiro_runtime import hiro_module

from .hiro_cultivation import _replace


def start_ap_quote(frame, cost):
    prices = set()
    for row in frame.texts:
        if not 1035 <= row.rect.center.y <= 1190:
            continue
        text = ''.join(unicodedata.normalize('NFKC', row.text).split())
        if match := re.fullmatch(r'(\d{1,3})→(\d{1,3})', text):
            prices.add(tuple(map(int, match.groups())))
    if len(prices) != 1:
        raise ValueError('Hiro opening AP quote is unreadable')
    before, after = prices.pop()
    if before - after != cost:
        raise ValueError('Hiro opening AP quote differs from the selected mode')
    return dict(before=before, after=after, cost=cost)


class HiroOpening:
    """One charged start per segment, with the observed quote and Maa job ID."""

    def __init__(self, *, costs, persist, refill):
        self.costs, self.persist, self.refill = costs, persist, refill
        self.starts = {}

    def enter(self, rt, mode, *, skin, difficulty=None, memory=(), formation=None, setup=None):
        opening = hiro_module('runtime.opening')
        nav = hiro_module('runtime.nav')
        blocked = hiro_module('runtime.loop').ResumeBlocked
        point = rt.raw_device.tap
        find, finals_button = nav.find, opening._finals_button
        charge_target = None

        def find_button(frame, name, **kwargs):
            nonlocal charge_target
            match = find(frame, name, **kwargs)
            if match is not None and name == 'Produce.Step4.ButtonProduceStart':
                charge_target = tuple(match.rect.center)
            return match

        def find_finals_button(frame):
            nonlocal charge_target
            charge_target = finals_button(frame)
            return charge_target

        def tap(x, y):
            nonlocal charge_target
            if (x, y) != charge_target:
                return point(x, y)
            current = rt.screen.frame
            finals = mode == 'main' and opening._finals_start(current)
            match = None if finals else find(current, 'Produce.Step4.ButtonProduceStart')
            target = finals_button(current) if finals else (tuple(match.rect.center) if match else None)
            if (x, y) != target:
                raise blocked('The AP start button changed before dispatch')
            if mode in self.starts:
                raise blocked('The charged start was already submitted')
            try:
                quote = start_ap_quote(current, self.costs[mode])
            except ValueError as error:
                raise blocked(str(error)) from error
            receipt = dict(quoted_ap=quote, sdk_success=None, job_id=None)
            self.starts[mode] = receipt
            self.persist(self.starts)
            receipt.update(job_id=point(x, y), sdk_success=True)
            self.persist(self.starts)
            charge_target = None
            return receipt['job_id']

        def stop_refill(*args, **kwargs):
            raise blocked('AP drink is disabled in the Produce task')

        prepare = dict(formation or {}, idol=skin)
        with ExitStack() as scope:
            scope.enter_context(_replace(rt.raw_device, 'tap', tap))
            scope.enter_context(_replace(nav, 'find', find_button))
            scope.enter_context(_replace(opening, '_finals_button', find_finals_button))
            if not self.refill:
                scope.enter_context(_replace(hiro_module('loop.ap_refill'), 'refill', stop_refill))
            if setup is not None:
                scope.enter_context(setup(rt))
            if mode == 'main':
                opening.new_produce(rt, mode, expect_memory=memory, **prepare)
            elif mode == 'qualify':
                opening.new_produce(rt, mode, **prepare)
            else:
                hiro_module('runtime.' + ('hajime' if mode == 'hajime' else 'nia_sample')).new_produce(
                    rt, difficulty, **prepare)
