"""Refresh the installed arena selector without changing saved option indices."""

from __future__ import annotations

import os
import json
import uuid
import threading
from pathlib import Path
from collections.abc import Callable

from .stages import ContestStageCatalog

PERIOD_OPTION = "竞技场期数"
PREVIEW_DESCRIPTION = "$竞技场预览期说明"
ComponentIdentity = tuple[str, str, str]


def period_case_label(
    season: int, *, latest: bool, preview: bool, recent_formal: bool,
) -> tuple[str, dict[str, str]]:
    """Describe RIS catalog flags without claiming which season the game is running."""

    if latest:
        key = "竞技场最新预览期格式" if preview else "竞技场最新期格式"
    elif preview:
        key = "竞技场固定预览期格式"
    elif recent_formal:
        key = "竞技场最近正式期格式"
    else:
        key = "竞技场固定期格式"
    return f"${key}", {"season": str(season)}


def update_season_option(interface: dict, catalog: ContestStageCatalog) -> None:
    """Append new fixed seasons; the GUI may sort their presentation separately."""

    definitions = {season: catalog.resolve(season) for season in catalog.seasons}
    latest = catalog.resolve("latest")
    recent_formal = max(
        (season for season, definition in definitions.items() if not definition.preview),
        default=None,
    ) if latest.preview else None
    options = interface.get("option")
    period = options.get(PERIOD_OPTION) if isinstance(options, dict) else None
    if not isinstance(period, dict) or period.get("type") != "select":
        raise ValueError("installed arena season selector is missing")
    cases = period.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("installed arena season cases are missing")
    represented: set[int] = set()
    names: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("installed arena season case is invalid")
        name = case.get("name")
        if not isinstance(name, str) or name in names:
            raise ValueError("installed arena season case names are invalid")
        names.add(name)
        override = case.get("pipeline_override")
        config = override.get("ChallengeSeasonConfig") if isinstance(override, dict) else None
        attach = config.get("attach") if isinstance(config, dict) else None
        selection = attach.get("season") if isinstance(attach, dict) else None
        if index == 0:
            if name != "latest" or selection != "latest":
                raise ValueError("installed arena latest must retain its symbolic first case")
            definition = latest
        else:
            if type(selection) is not int or selection < 1 or name != f"season-{selection}":
                raise ValueError("installed arena fixed season identity is invalid")
            represented.add(selection)
            definition = definitions.get(selection)
        if definition is not None:
            case["label"], case["label_args"] = period_case_label(
                definition.season, latest=index == 0, preview=definition.preview,
                recent_formal=definition.season == recent_formal,
            )
            if definition.preview:
                case["description"] = PREVIEW_DESCRIPTION
            elif case.get("description") == PREVIEW_DESCRIPTION:
                case.pop("description")
    for season in sorted(definitions.keys() - represented):
        label, label_args = period_case_label(
            season, latest=False, preview=definitions[season].preview,
            recent_formal=season == recent_formal,
        )
        case = {
            "name": f"season-{season}",
            "label": label,
            "label_args": label_args,
            "pipeline_override": {"ChallengeSeasonConfig": {"attach": {"season": season}}},
        }
        if definitions[season].preview:
            case["description"] = PREVIEW_DESCRIPTION
        cases.append(case)
    replacement = f"season-{latest.season}"
    cases[0]["replacement_case"] = replacement
    period["default_case"] = replacement
    period["dynamic_cases"] = True


class ArenaSeasonUiSynchronizer:
    """Try once per component/task; successful identities need no further file I/O."""

    def __init__(self, interface_path: Path) -> None:
        self.interface_path = interface_path
        self._successful_identity: ComponentIdentity | None = None
        self._failed_identity: ComponentIdentity | None = None
        self._lock = threading.Lock()

    def reset_failed(self) -> None:
        with self._lock:
            self._failed_identity = None

    def sync(
        self,
        catalog: ContestStageCatalog,
        identity: ComponentIdentity,
        *,
        on_error: Callable[[Exception], None],
    ) -> bool:
        with self._lock:
            if identity == self._successful_identity:
                return True
            if identity == self._failed_identity:
                return False
            temporary: Path | None = None
            try:
                original = self.interface_path.read_text(encoding="utf-8-sig")
                interface = json.loads(original)
                if not isinstance(interface, dict):
                    raise ValueError("installed interface root must be an object")
                before = json.dumps(interface, ensure_ascii=False)
                update_season_option(interface, catalog)
                if json.dumps(interface, ensure_ascii=False) != before:
                    temporary = self.interface_path.with_name(f".interface-season-{uuid.uuid4().hex}.tmp")
                    temporary.write_text(
                        json.dumps(interface, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    )
                    os.replace(temporary, self.interface_path)
                self._successful_identity = identity
                self._failed_identity = None
                return True
            except FileNotFoundError as error:
                self._failed_identity = identity
                # Source worktrees and custom diagnostic bundles need no installed UI.
                if temporary is not None:
                    self._report(on_error, error)
                return False
            except Exception as error:
                self._failed_identity = identity
                self._report(on_error, error)
                return False
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError as error:
                        self._report(on_error, error)

    @staticmethod
    def _report(on_error: Callable[[Exception], None], error: Exception) -> None:
        try:
            on_error(error)
        except Exception:
            pass
