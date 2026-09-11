"""Compile the actual focus path diagnostic against an offline logging adapter."""
from __future__ import annotations

import os
import json
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

from test_dynamic_option_cases import SOURCE_SHA256, extract_method

HERE = Path(__file__).resolve().parent


def run(args):
    archive, dotnet, work = args.source_zip.resolve(), args.dotnet.resolve(), args.work_dir.resolve()
    if work.exists():
        raise ValueError("work-dir must be new")
    if hashlib.sha256(archive.read_bytes()).hexdigest().upper() != SOURCE_SHA256:
        raise ValueError("Unexpected pinned source archive")
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(work)
    source = work / "MFAAvalonia-2.15.2"
    patch = args.patch.resolve()
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(work), DOTNET_CLI_TELEMETRY_OPTOUT="1")
    for options in (["--check"], []):
        subprocess.run(["git", "apply", *options, "--", str(patch)], cwd=source, env=env, check=True)
    focus = (source / "MFAAvalonia/Extensions/MaaFW/FocusHandler.cs").read_text(encoding="utf-8-sig")
    methods = [extract_method(focus, signature) for signature in (
        "    private static void LogUnresolvedFocusPath(",
        "    private static bool LooksLikeFilePath(",
    )]
    production = work / "ProductionMethods.cs"
    production.write_text("using System; using System.IO; using System.Text.RegularExpressions;\n"
                          + "partial class FocusHandler {\n" + "\n".join(methods) + "\n}", encoding="utf-8")
    sdk = sorted((dotnet.parent / "sdk").glob("10.*"))[-1]
    refs = sorted((dotnet.parent / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))[-1]
    runtime = sorted((dotnet.parent / "shared/Microsoft.NETCore.App").glob("10.*"))[-1]
    output = work / "FocusHarness.dll"
    response = work / "compile.rsp"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable",
        f'/out:"{output}"', *(f'/reference:"{item}"' for item in sorted(refs.glob("*.dll"))),
        f'"{production}"', f'"{HERE / "focus_content_harness.cs"}"',
    ]), encoding="utf-8")
    (work / "FocusHarness.runtimeconfig.json").write_text(json.dumps({
        "runtimeOptions": {"tfm": "net10.0", "framework": {"name": "Microsoft.NETCore.App", "version": runtime.name}},
    }), encoding="utf-8")
    for name, command in (
        ("compile", [str(dotnet), str(sdk / "Roslyn/bincore/csc.dll"), "/noconfig", f"@{response}"]),
        ("sequence", [str(dotnet), str(output), str(work)]),
    ):
        result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        (work / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        result.check_returncode()
    evidence = {"status": "PASS", "result": result.stdout.strip(), "source_sha256": SOURCE_SHA256,
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest().upper(),
                "production_methods_sha256": hashlib.sha256(production.read_bytes()).hexdigest().upper(),
                "network": "none; no restore", "scope": "actual diagnostic methods; no GUI or content resolver changes"}
    (work / "result.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--dotnet", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--patch", type=Path, default=HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch")
    print(json.dumps(run(parser.parse_args()), indent=2))
