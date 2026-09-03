"""One-time migration from free-text arena seasons to the fixed selector."""

from __future__ import annotations

import json
import argparse
from uuid import uuid4
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping, MutableMapping, MutableSequence

PERIOD_OPTION_NAME = "竞技场期数"
PERIOD_CONFIG_NODE = "ChallengeSeasonConfig"
LEGACY_SEASON_FIELDS = ("arena_win_rate_season", "arena_own_score_season")


class ArenaPeriodMigrationError(RuntimeError):
    """Raised when an instance cannot be migrated without guessing."""


def _period_case_selection(case: Mapping[str, Any]) -> object:
    pipeline_override = case.get("pipeline_override")
    if not isinstance(pipeline_override, Mapping):
        raise ArenaPeriodMigrationError("fixed arena period case has no pipeline override")

    selections: list[object] = []
    for node_name, parameter_name in (
        (PERIOD_CONFIG_NODE, "attach"),
        ("ChallengeChoose", "custom_action_param"),
    ):
        node = pipeline_override.get(node_name)
        parameters = node.get(parameter_name) if isinstance(node, Mapping) else None
        if isinstance(parameters, Mapping) and "season" in parameters:
            selections.append(parameters["season"])
    if not selections:
        raise ArenaPeriodMigrationError("fixed arena period case has no season value")
    if any(selection != selections[0] for selection in selections[1:]):
        raise ArenaPeriodMigrationError("fixed arena period case has conflicting season values")
    return selections[0]


@dataclass(frozen=True)
class ArenaPeriodSelectionCatalog:
    index_by_value: Mapping[str | int, int]
    value_by_index: tuple[str | int, ...]
    default_index: int

    @classmethod
    def from_interface(cls, value: Mapping[str, Any]) -> "ArenaPeriodSelectionCatalog":
        options = value.get("option")
        if not isinstance(options, Mapping):
            raise ArenaPeriodMigrationError("interface option catalog is missing")
        period = options.get(PERIOD_OPTION_NAME)
        if not isinstance(period, Mapping) or period.get("type") != "select":
            raise ArenaPeriodMigrationError("fixed arena period selector is missing")
        cases = period.get("cases")
        default_case = period.get("default_case")
        if not isinstance(cases, list) or not cases or not isinstance(default_case, str):
            raise ArenaPeriodMigrationError("fixed arena period selector is invalid")

        index_by_value: dict[str | int, int] = {}
        default_index: int | None = None
        for index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                raise ArenaPeriodMigrationError("fixed arena period case is invalid")
            if case.get("name") == default_case:
                default_index = index
            selection = _period_case_selection(case)
            if (
                selection != "latest"
                and (isinstance(selection, bool) or not isinstance(selection, int) or selection < 1)
            ):
                raise ArenaPeriodMigrationError("fixed arena period case has an invalid season value")
            if selection in index_by_value:
                raise ArenaPeriodMigrationError("fixed arena period selector contains duplicate values")
            index_by_value[selection] = index
        if default_index is None:
            raise ArenaPeriodMigrationError("fixed arena period default case is absent")
        return cls(
            index_by_value=index_by_value,
            value_by_index=tuple(index_by_value),
            default_index=default_index,
        )

    def legacy_index(self, value: object) -> int | None:
        if value == "latest":
            return self.index_by_value.get("latest")
        if isinstance(value, str) and value.isascii() and value.isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
        return self.index_by_value.get(value)


@dataclass
class ArenaPeriodMigrationResult:
    changed: bool = False
    legacy_fields_removed: int = 0
    selectors_added: int = 0
    selectors_repaired: int = 0
    selectors_remapped: int = 0
    warnings: list[str] = field(default_factory=list)


def _property(value: Mapping[str, Any], name: str) -> str | None:
    folded = name.casefold()
    return next((key for key in value if isinstance(key, str) and key.casefold() == folded), None)


def _migrate_option_list(
    options: MutableSequence[Any],
    *,
    catalog: ArenaPeriodSelectionCatalog,
    previous_catalog: ArenaPeriodSelectionCatalog | None,
    result: ArenaPeriodMigrationResult,
) -> None:
    legacy_values: list[object] = []
    existing_selector: MutableMapping[str, Any] | None = None

    for option in tuple(options):
        if not isinstance(option, MutableMapping):
            continue
        name_key = _property(option, "name")
        if name_key is not None and option.get(name_key) == PERIOD_OPTION_NAME:
            existing_selector = option

        data_key = _property(option, "data")
        data = option.get(data_key) if data_key is not None else None
        if isinstance(data, MutableMapping):
            for legacy_field in LEGACY_SEASON_FIELDS:
                if legacy_field in data:
                    legacy_values.append(data.pop(legacy_field))
                    result.changed = True
                    result.legacy_fields_removed += 1

        sub_options_key = _property(option, "sub_options")
        sub_options = option.get(sub_options_key) if sub_options_key is not None else None
        if isinstance(sub_options, MutableSequence):
            _migrate_option_list(
                sub_options,
                catalog=catalog,
                previous_catalog=previous_catalog,
                result=result,
            )

    if existing_selector is None and not legacy_values:
        return

    migrated_indices = [catalog.legacy_index(value) for value in legacy_values]
    valid_indices = {index for index in migrated_indices if index is not None}
    if not legacy_values:
        migrated_index = catalog.default_index
    elif any(index is None for index in migrated_indices) or len(valid_indices) != 1:
        migrated_index = catalog.default_index
        result.warnings.append(
            f"旧期数值 {legacy_values!r} 无法唯一映射到固定目录，已回退到目录最新"
        )
    else:
        migrated_index = valid_indices.pop()

    if existing_selector is None:
        options.insert(0, {"name": PERIOD_OPTION_NAME, "index": migrated_index})
        result.changed = True
        result.selectors_added += 1
        return

    index_key = _property(existing_selector, "index") or "index"
    current_index = existing_selector.get(index_key)
    if previous_catalog is not None:
        if (
            isinstance(current_index, bool)
            or not isinstance(current_index, int)
            or not 0 <= current_index < len(previous_catalog.value_by_index)
        ):
            existing_selector[index_key] = migrated_index
            result.changed = True
            result.selectors_repaired += 1
            result.warnings.append("旧竞技场期数选择索引无效，已修复")
            return
        previous_value = previous_catalog.value_by_index[current_index]
        remapped_index = catalog.index_by_value.get(previous_value)
        if remapped_index is None:
            remapped_index = catalog.default_index
            result.warnings.append(
                f"旧竞技场期数 {previous_value!r} 已不受支持，已回退到目录最新"
            )
        if remapped_index != current_index:
            existing_selector[index_key] = remapped_index
            result.changed = True
            result.selectors_remapped += 1
        return

    if (
        isinstance(current_index, bool)
        or not isinstance(current_index, int)
        or not 0 <= current_index < len(catalog.value_by_index)
    ):
        existing_selector[index_key] = migrated_index
        result.changed = True
        result.selectors_repaired += 1
        result.warnings.append("已用旧期数修复无效的竞技场期数选择索引")


def migrate_instance_value(
    value: MutableMapping[str, Any],
    *,
    catalog: ArenaPeriodSelectionCatalog,
    previous_catalog: ArenaPeriodSelectionCatalog | None = None,
) -> ArenaPeriodMigrationResult:
    """Migrate every nested Maa option list while preserving unrelated settings."""

    result = ArenaPeriodMigrationResult()

    def visit(node: object) -> None:
        if isinstance(node, MutableMapping):
            for key, child in tuple(node.items()):
                if isinstance(child, MutableSequence) and key.casefold() in {"option", "sub_options"}:
                    _migrate_option_list(
                        child,
                        catalog=catalog,
                        previous_catalog=previous_catalog,
                        result=result,
                    )
                elif isinstance(child, (MutableMapping, MutableSequence)):
                    visit(child)
        elif isinstance(node, MutableSequence):
            for child in tuple(node):
                visit(child)

    visit(value)
    return result


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def migrate_instance_file(
    path: str | Path,
    *,
    catalog: ArenaPeriodSelectionCatalog,
    previous_catalog: ArenaPeriodSelectionCatalog | None = None,
) -> ArenaPeriodMigrationResult:
    instance_path = Path(path)
    try:
        value = json.loads(instance_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArenaPeriodMigrationError(f"实例配置无法读取或不是有效 JSON：{instance_path}") from error
    if not isinstance(value, MutableMapping):
        raise ArenaPeriodMigrationError(f"实例配置根结构不是对象：{instance_path}")
    result = migrate_instance_value(
        value,
        catalog=catalog,
        previous_catalog=previous_catalog,
    )
    if result.changed:
        try:
            _atomic_write_json(instance_path, value)
        except OSError as error:
            raise ArenaPeriodMigrationError(f"实例配置迁移结果无法写入：{instance_path}") from error
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", required=True, type=Path)
    parser.add_argument("--interface", required=True, type=Path)
    parser.add_argument("--previous-interface", type=Path)
    args = parser.parse_args()

    try:
        interface = json.loads(args.interface.read_text(encoding="utf-8-sig"))
        if not isinstance(interface, Mapping):
            raise ArenaPeriodMigrationError("interface root is not an object")
        catalog = ArenaPeriodSelectionCatalog.from_interface(interface)
        previous_catalog = None
        if args.previous_interface is not None:
            previous_interface = json.loads(
                args.previous_interface.read_text(encoding="utf-8-sig")
            )
            if not isinstance(previous_interface, Mapping):
                raise ArenaPeriodMigrationError("previous interface root is not an object")
            previous_catalog = ArenaPeriodSelectionCatalog.from_interface(previous_interface)
        reports = []
        if args.instances_dir.is_dir():
            for path in sorted(args.instances_dir.glob("*.json")):
                result = migrate_instance_file(
                    path,
                    catalog=catalog,
                    previous_catalog=previous_catalog,
                )
                reports.append(
                    {
                        "path": path.name,
                        "changed": result.changed,
                        "legacy_fields_removed": result.legacy_fields_removed,
                        "selectors_added": result.selectors_added,
                        "selectors_repaired": result.selectors_repaired,
                        "selectors_remapped": result.selectors_remapped,
                        "warnings": result.warnings,
                    }
                )
        print(json.dumps({"schema_version": 1, "instances": reports}, ensure_ascii=False))
    except (OSError, json.JSONDecodeError, ArenaPeriodMigrationError) as error:
        parser.exit(2, f"竞技场期数配置迁移失败：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
