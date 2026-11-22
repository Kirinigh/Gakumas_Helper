"""Resolve the existing repository/package layouts from this source tree.

The precedence and missing-file behavior match the Live observer's existing
resolver. Working directory, task parameters and saved runtime paths do not
select a different code/data source.
"""
from pathlib import Path


def resolve_runtime_path(source_relative: Path, installed_relative: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = root / source_relative, root / installed_relative
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])
