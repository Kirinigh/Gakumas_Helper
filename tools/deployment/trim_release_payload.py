"""Project qualified build files into a slimmer Windows release payload."""

import json
import shutil
import argparse
from pathlib import Path

_MODEL = "resource/base/model"
_CARD_BASE = f"{_MODEL}/classify/card_embedding/preserved_base"
_ARENA_BASE = f"{_MODEL}/embedding/arena_card/preserved_base"
_BADGE_BASE = f"{_MODEL}/embedding/arena_badge_reference/preserved_base"

# Keep the build evidence available during component qualification. The remaining
# ancestor classification and arena manifest, gallery and class table stay in
# the payload until the build and shipping layouts are independently qualified.
TRIMMED_PATHS = frozenset({
    f"{_CARD_BASE}/README.md",
    f"{_CARD_BASE}/card_embedding_classes.json",
    f"{_CARD_BASE}/card_embedding_gallery.npz",
    f"{_CARD_BASE}/card_embedding_model.onnx",
    f"{_CARD_BASE}/manifest.json",
    f"{_BADGE_BASE}/badge_reference_gallery.npz",
    f"{_BADGE_BASE}/hard_negative_evaluation.json",
    f"{_BADGE_BASE}/manifest.json",
    f"{_BADGE_BASE}/skill_card_classification.json",
    f"{_ARENA_BASE}/README.md",
    f"{_ARENA_BASE}/arena_card_embedding_model.onnx",
    f"{_ARENA_BASE}/skill_card_classification.json",
    f"{_ARENA_BASE}/preserved_base/arena_card_embedding_classes.json",
    f"{_ARENA_BASE}/preserved_base/arena_card_embedding_gallery.npz",
    f"{_ARENA_BASE}/preserved_base/arena_card_embedding_model.onnx",
    f"{_ARENA_BASE}/preserved_base/manifest.json",
    "docs/img/button_example.png",
    "docs/img/button_F1_curve.png",
    "docs/img/button_PR_curve.png",
    "docs/img/button_val_batch0_pred.jpg",
})

_UPDATE_STATE = "GAKUMAS_HELPER_UPDATE.json"


def copy_trimmed_payload(source: Path, destination: Path) -> tuple[str, ...]:
    """Copy an already qualified build into a new directory for manifest generation."""
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("The shipping payload must be a new directory outside the qualified build")

    build = json.loads((source / "GAKUMAS_HELPER_BUILD.json").read_text(encoding="utf-8-sig"))
    protected = TRIMMED_PATHS.intersection(build["critical_files"])
    if protected:
        raise ValueError(f"Slimming would remove required program files: {sorted(protected)}")

    removed = tuple(sorted(path for path in TRIMMED_PATHS if (source / path).is_file()))
    excluded = {*removed, _UPDATE_STATE}

    def omit(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(source)
        return {name for name in names if (relative / name).as_posix() in excluded}

    shutil.copytree(source, destination, symlinks=True, ignore=omit)
    return removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Qualified build with its component evidence")
    parser.add_argument("destination", type=Path, help="New shipping tree for update manifest generation")
    args = parser.parse_args()
    for path in copy_trimmed_payload(args.source, args.destination):
        print(path)
