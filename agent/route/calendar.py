"""Observe the actual HIF countdown before choosing an ordinary page family."""
import time

from .hif import day_from_countdown

FAMILIES = {1: 'lesson', 2: 'training', 3: 'outing', 4: 'lesson', 5: 'training', 6: 'consult_entry'}


class ScheduleCountdownChanged(ValueError):
    """The calendar is transitioning; return to recognition without input."""


def read_calendar(rows):
    from .production import _text
    _text(rows, r'H[.]I[.]F本戦まで', (20, 20, 190, 80))
    return read_calendar_countdown(rows)


def read_calendar_countdown(rows):
    from .production import _text
    countdown = _text(rows, r'[0-6]日', (40, 60, 145, 100))
    day = day_from_countdown(int(countdown[0]))
    if day not in FAMILIES:
        raise ValueError('ordinary calendar has no bound page family for its observed day')
    return day, FAMILIES[day]


def refresh_schedule_home(parent, image, rows, frame_id, *, day):
    """Use the first date only; read actionable buttons from miyu's next capture."""
    from .production import _current_rows

    if any(row['text'] in ('SELECT', 'SELEC') and row['box'][1] >= 1000 for row in rows):
        return None
    try:
        first_day, _ = read_calendar_countdown(rows)
    except ValueError:
        if not any(row['text'] == 'H.I.F本戦まで' for row in rows):
            return None
        first_day = None
    metadata = dict(frame_id=frame_id, phase='schedule_countdown', day=first_day, rows=rows)
    if parent.evidence_sink:
        parent.evidence_sink(image, metadata)
    parent.events.append(metadata)
    if first_day != day:
        time.sleep(.5)
        raise ScheduleCountdownChanged('schedule countdown differs from the bound day')
    time.sleep(.25)
    if parent.context.tasker.stopping:
        raise ValueError('task stopping before schedule countdown recheck')
    started = parent.clock.now()
    capture = parent.context.tasker.controller.post_screencap().wait()
    if not capture.succeeded:
        raise ValueError('schedule countdown recheck capture failed')
    image = capture.get()
    rows = _current_rows(parent.context, image)
    try:
        second_calendar = read_calendar_countdown(rows)
    except ValueError:
        second_calendar = None
    second_day = second_calendar[0] if second_calendar is not None else None
    if second_day != first_day:
        metadata = dict(frame_id=f'route-frame-{parent.game_pid}-{started}', phase='schedule_countdown_changed',
            day=second_day, first_frame_id=frame_id, first_day=first_day, rows=rows)
        if parent.evidence_sink:
            parent.evidence_sink(image, metadata)
        parent.events.append(metadata)
        time.sleep(.5)
        raise ScheduleCountdownChanged('schedule countdown changed before choosing the current button')
    return started, image, rows, second_calendar


def read_route_resources(rows):
    from .production import _text
    matched = next((row for row in rows if row.get('source') == 'observed_submitted_course_fraction'), None)
    fraction = matched['text'] if matched is not None else _text(rows, r'\d{1,3}/\d{1,3}', (285, 35, 185, 85))
    current, maximum = map(int, fraction.split('/'))
    currency = int(_text(rows, r'\d{1,6}', (350, 80, 120, 85)))
    if not 0 <= current <= maximum <= 100 or maximum == 0:
        raise ValueError('ordinary route resources are contradictory')
    return dict(stamina=current, max_stamina=maximum, currency=currency)
