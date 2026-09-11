"""Resolve the paired native/Python framework pin without network access."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def resolve_framework_version(root: Path) -> str:
    declarations = []
    for line in (root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        requirement = line.partition("#")[0].strip()
        if re.match(r"maafw(?:\s|[=<>!~;\[]|$)", requirement, re.IGNORECASE):
            declarations.append(requirement)
    if len(declarations) != 1:
        raise ValueError("requirements.txt must contain exactly one maafw pin")
    match = re.fullmatch(r"maafw\s*==\s*(\d+\.\d+\.\d+)", declarations[0], re.IGNORECASE)
    if match is None:
        raise ValueError("maafw must use an exact stable version")
    version = match.group(1)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project_pins = [item for item in project["project"]["dependencies"] if item.lower().startswith("maafw")]
    if project_pins != [f"maafw=={version}"]:
        raise ValueError("pyproject.toml and requirements.txt maafw pins differ")
    return version


if __name__ == "__main__":
    paired_version = resolve_framework_version(Path(__file__).resolve().parents[2])
    print(f"tag=v{paired_version}")
    print(f"maafw_version={paired_version}")
