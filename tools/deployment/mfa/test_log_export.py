"""Compile the real patched MFA log exporter and verify its ZIP output offline.

Only the GUI/storage-picker seam is replaced. No MFA installation is read or
modified, no controller is created, and no dependency restore is performed.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

SOURCE_SHA256 = "76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A"
EXPORTER_BLOB = "689a4f592e24dd85b95ff3fc198aa38803c7306a"
HERE = Path(__file__).resolve().parent


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    archive = args.source_zip.resolve()
    dotnet = args.dotnet.resolve()
    work = args.work_dir.resolve()
    if work.exists():
        raise ValueError("work-dir must be a fresh directory")
    if hashlib.sha256(archive.read_bytes()).hexdigest().upper() != SOURCE_SHA256:
        raise ValueError("The source ZIP is not the pinned MFAAvalonia v2.15.2 archive")
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source_zip:
        source_zip.extractall(work)
    source_root = work / "MFAAvalonia-2.15.2"
    exporter = source_root / "MFAAvalonia/Helper/FileLogExporter.cs"
    original = exporter.read_bytes()
    if hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest() != EXPORTER_BLOB:
        raise ValueError("The pinned FileLogExporter Git object does not match")
    env = os.environ.copy()
    env.update({
        "GIT_CEILING_DIRECTORIES": str(work),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_GENERATE_ASPNET_CERTIFICATE": "false",
        "DOTNET_ADD_GLOBAL_TOOLS_TO_PATH": "false",
        "DOTNET_CLI_HOME": str(work / "dotnet-home"),
    })
    patch = HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
    for options in (["--check"], []):
        subprocess.run(["git", "apply", *options, "--", str(patch)], cwd=source_root, env=env, check=True)
    if "ExportLogResult.Partial" not in exporter.read_text(encoding="utf-8-sig"):
        raise ValueError("The log exporter patch was skipped")
    sdk_root = dotnet.parent
    sdk_dirs = sorted((sdk_root / "sdk").glob("10.*"))
    ref_dirs = sorted((sdk_root / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))
    runtime_dirs = sorted((sdk_root / "shared/Microsoft.NETCore.App").glob("10.*"))
    if not sdk_dirs or not ref_dirs or not runtime_dirs:
        raise ValueError("A complete, already installed .NET 10 SDK is required")
    output = work / "LogExportHarness.dll"
    response = work / "compile.rsp"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable",
        f'/out:"{output}"',
        *(f'/reference:"{item}"' for item in sorted(ref_dirs[-1].glob("*.dll"))),
        f'"{exporter}"',
        f'"{HERE / "log_export_harness.cs"}"',
    ]), encoding="utf-8")
    (work / "LogExportHarness.runtimeconfig.json").write_text(json.dumps({
        "runtimeOptions": {"tfm": "net10.0", "framework": {
            "name": "Microsoft.NETCore.App", "version": runtime_dirs[-1].name,
        }},
    }), encoding="utf-8")
    compile_result = subprocess.run([
        str(dotnet), str(sdk_dirs[-1] / "Roslyn/bincore/csc.dll"), "/noconfig", f"@{response}",
    ], cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (work / "compile.log").write_text(compile_result.stdout + compile_result.stderr, encoding="utf-8")
    compile_result.check_returncode()
    result = subprocess.run([str(dotnet), str(output), str(work / "cases")], cwd=work, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    (work / "export.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    evidence = {
        "source_sha256": SOURCE_SHA256,
        "original_exporter_blob": EXPORTER_BLOB,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest().upper(),
        "exporter_sha256": hashlib.sha256(exporter.read_bytes()).hexdigest().upper(),
        "implementation": "complete production FileLogExporter.cs; only UI and picker interfaces stubbed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "result": result.stdout.strip(),
        "network": "none; no restore",
        "archives": [
            {"path": str(p.relative_to(work)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest().upper()}
            for p in sorted((work / "cases").rglob("*.zip")) if p.name == "output.zip"
        ],
    }
    (work / "result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--dotnet", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
