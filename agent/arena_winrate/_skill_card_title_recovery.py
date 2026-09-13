"""One-character title lookup; callers own popup geometry and confirmation."""

from collections import defaultdict


def _one_edit(first: str, second: str) -> bool:
    if first == second or abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        return sum(a != b for a, b in zip(first, second)) == 1
    short, long = sorted((first, second), key=len)
    index = next((i for i, char in enumerate(short) if char != long[i]), len(short))
    return short[index:] == long[index + 1:]


class SkillCardTitleRecoveryIndex:
    def __init__(self, titles: dict[str, tuple[int, ...]]) -> None:
        self.titles = titles
        deleted: defaultdict[str, set[str]] = defaultdict(set)
        for title in titles:
            # A plus sign is identity information, never an editable character.
            for index, char in enumerate(title):
                if char != "+":
                    deleted[title[:index] + title[index + 1:]].add(title)
        self.deleted = dict(deleted)

    def candidates(self, observed: str) -> tuple[int, ...]:
        if (
            not observed or observed in self.titles
            or "+" in observed.removesuffix("+")
            # A trailing unknown glyph could be a misread upgrade mark.
            # Do not delete it and silently accept the normal sibling.
            or (not observed.endswith("+") and observed[:-1] + "+" in self.titles)
        ):
            return ()
        possible = set(self.deleted.get(observed, ()))
        for index, char in enumerate(observed):
            if char == "+":
                continue
            removed = observed[:index] + observed[index + 1:]
            if removed in self.titles:
                possible.add(removed)
            possible.update(self.deleted.get(removed, ()))
        return tuple(sorted({
            card_id
            for title in possible
            if title.endswith("+") == observed.endswith("+")
            and _one_edit(observed.removesuffix("+"), title.removesuffix("+"))
            for card_id in self.titles[title]
        }))
