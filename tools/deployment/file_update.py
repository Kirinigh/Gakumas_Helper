"""File-level release inventories and adjacent delta assets (no binary patches)."""
from __future__ import annotations

import re
import json
import hashlib
import zipfile
from pathlib import Path

STATE = "GAKUMAS_HELPER_UPDATE.json"
PROTOCOL = 1
MUTABLE = frozenset({".local", "config", "logs", "log", "cache", "captures", "screenshots",
                     "profiles", "userdata", "user_data", "runtime-data", "temp", "tmp",
                     "temp_res", "temp_mfa", "temp_maafw", "backup", "debug"})
MUTABLE_FILES = frozenset({"config.json", "appsettings.json", "maa.log", "gui.log"})


def protected(relative: str) -> bool:
    parts = relative.lower().split("/")
    return (parts[0] in MUTABLE or parts[-1] in MUTABLE_FILES
            or "__pycache__" in parts or parts[-1].endswith((".pyc", ".backupmfa")))


def safe_path(relative: str) -> str:
    if (not relative or "\\" in relative or ":" in relative
            or any(part in {"", ".", ".."} or part.endswith((".", " ")) for part in relative.split("/"))
            or protected(relative)):
        raise ValueError(f"Invalid or mutable update path: {relative}")
    return relative


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files = {}
    seen = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Update payload cannot contain links")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == STATE:
            continue
        safe_path(relative)
        if relative.casefold() in seen:
            raise ValueError("Update payload contains case-colliding paths")
        seen.add(relative.casefold())
        files[relative] = digest(path)
    return files


def version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value)
    if not match:
        raise ValueError("File updates require a stable semantic version")
    return tuple(map(int, match.groups()))


def write_state(root: Path, release_version: str) -> dict:
    version(release_version)
    build = json.loads((root / "GAKUMAS_HELPER_BUILD.json").read_text(encoding="utf-8-sig"))
    # Runtime/framework/layout changes require a full package; the GUI/Core may change in a delta.
    files = inventory(root)
    state = {"schema_version": PROTOCOL, "version": release_version,
             "compatibility": {"layout": "gkh-portable-v1", "platform": "win-x86_64",
                               "framework": build.get("framework"),
                               "python_packages": build.get("python_packages", {}),
                               "runtime_files": {p: h for p, h in files.items()
                                                 if (p.startswith(("python/", "runtimes/", "libs/System.", "libs/Microsoft."))
                                                     or p.endswith(".runtimeconfig.json")
                                                     or p.rsplit("/", 1)[-1].lower() in {"coreclr.dll", "hostfxr.dll", "hostpolicy.dll", "clrjit.dll"})}},
             "files": files}
    (root / STATE).write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return state


def load_state(root: Path) -> dict:
    state = json.loads((root / STATE).read_text(encoding="utf-8-sig"))
    if type(state.get("schema_version")) is not int or state["schema_version"] != PROTOCOL or state.get("files") != inventory(root):
        raise ValueError("Release inventory does not match actual package bytes")
    version(state["version"])
    return state


def build_adjacent(candidate: Path, previous: Path | None, output: Path, *, require_full: bool = False) -> dict:
    target = load_state(candidate)
    result = {"protocol": PROTOCOL, "target": target,
              "state_sha256": digest(candidate / STATE), "delta": None}
    if previous is None or not (previous / STATE).is_file():
        return result  # Bootstrap is always a complete update.
    base = load_state(previous)
    left, right = version(base["version"]), version(target["version"])
    if left >= right:
        raise ValueError("Previous formal release must precede target")
    if require_full or left[:2] != right[:2] or base["compatibility"] != target["compatibility"]:
        return result
    old, new = base["files"], target["files"]
    added = sorted(new.keys() - old.keys())
    deleted = sorted(old.keys() - new.keys())
    modified = sorted(p for p in new.keys() & old.keys() if new[p] != old[p])
    changes = {"added": added + [STATE], "modified": modified, "deleted": deleted,
               "added_dir": [], "deleted_dir": []}
    # Deliberately no OS/architecture token: every historical selector ranks the full ZIP higher.
    name = f"GakumasHelper-delta-{base['version']}-to-{target['version']}.gkhdelta"
    destination = output / name
    if destination.exists():
        raise ValueError("Delta output already exists")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        payloads = {p: (candidate / p).read_bytes() for p in added + modified + [STATE]}
        payloads["changes.json"] = (json.dumps(changes, sort_keys=True) + "\n").encode()
        for relative, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(relative, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    result["delta"] = {"from": base["version"], "to": target["version"],
                       "base_state_sha256": digest(previous / STATE),
                       "base": base, "asset": name, "bytes": destination.stat().st_size,
                       "sha256": digest(destination)}
    return result
