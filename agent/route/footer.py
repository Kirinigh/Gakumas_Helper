"""Locate the cultivation menu and the two observed nonbattle HUD layouts."""
from .prestart_memory import _match


def _menu(image):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('unsupported HIF cultivation image')
    # The 141 capture identifies the ordinary menu actually opened in-game.
    return _match(image, 'hif_cultivation_menu', (600, 1150, 110, 100))


def cultivation_menu_target(image):
    target = _menu(image)
    if target is None:
        raise ValueError('current cultivation menu icon is absent')
    return target


def read_footer_offset(image, *, context=None):
    target = _menu(image)
    # In 157/173 the menu image starts at y=1182; in 170 it is 13px lower.
    if target is not None:
        for offset in (0, 13):
            if abs(target[1] - 1182 - offset) <= 1:
                return offset, dict(source='cultivation_menu_template', box=target)
    if context is not None:
        from .production import read_nonbattle_drink_controls
        # A covered menu still leaves the actual slot controls available. Locate
        # those at the two supported insets without inferring any drink identity.
        matches = []
        for offset in (0, 13):
            filled, empty = read_nonbattle_drink_controls(context, image, footer_offset=offset)
            slots = [index for index, pair in enumerate(zip(filled, empty, strict=True)) if any(pair)]
            if slots:
                matches.append((offset, dict(source='observed_drink_controls', slots=slots)))
        if len(matches) == 1:
            return matches[0]
    raise ValueError('current drink footer layout is unresolved')
