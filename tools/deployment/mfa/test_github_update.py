"""Compile actual GitHub update discovery and rate-limit handling; no network or restore."""
from __future__ import annotations

import os
import json
import shutil
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

from test_download_transport import CHECKER_BLOB, SOURCE_SHA256, extract_method

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
    checker = source / "MFAAvalonia/Helper/VersionChecker.cs"
    with zipfile.ZipFile(archive) as zipped:
        original = zipped.read("MFAAvalonia-2.15.2/MFAAvalonia/Helper/VersionChecker.cs")
    if hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest() != CHECKER_BLOB:
        raise ValueError("Unexpected original VersionChecker blob")
    code = checker.read_text(encoding="utf-8-sig")
    start = code.index("    public static async Task<(string url, string latestVersion, string sha256)> GetLatestVersionAndDownloadUrlFromGithubAsync(")
    end = code.index("    private static void GetDownloadUrlFromMirror", start)
    methods = code[start:end] + "\n" + "\n".join(extract_method(code, name) for name in (
        "IsNewVersionAvailable", "ParseAndNormalizeVersion"))
    production = work / "ProductionMethods.cs"
    production.write_text("using System; using System.Linq; using System.Net; using System.Net.Http; "
                          "using System.Net.Http.Headers; using System.Collections.Generic; using System.Threading.Tasks; "
                          "using System.Numerics; using System.Runtime.InteropServices; using System.Text.RegularExpressions; "
                          "using Newtonsoft.Json.Linq; using Semver; using MFAAvalonia.Helper;\n"
                          + "partial class VersionChecker {\n" + methods + "\n}", encoding="utf-8")
    dependencies = [args.libs.resolve() / name for name in ("Newtonsoft.Json.dll", "Semver.dll")]
    for dependency in dependencies:
        shutil.copy2(dependency, work / dependency.name)
    helper = source / "MFAAvalonia/Helper/GitHubApiRequests.cs"
    sdk = sorted((dotnet.parent / "sdk").glob("10.*"))[-1]
    refs = sorted((dotnet.parent / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))[-1]
    runtime = sorted((dotnet.parent / "shared/Microsoft.NETCore.App").glob("10.*"))[-1]
    output = work / "GitHubUpdateHarness.dll"
    response = work / "compile.rsp"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable",
        f'/out:"{output}"', *(f'/reference:"{item}"' for item in sorted(refs.glob("*.dll"))),
        *(f'/reference:"{item}"' for item in dependencies),
        *([f'"{helper}"'] if helper.exists() else []),
        f'"{production}"', f'"{HERE / "github_update_harness.cs"}"',
    ]), encoding="utf-8")
    (work / "GitHubUpdateHarness.runtimeconfig.json").write_text(json.dumps({
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
                "network": "none; no restore", "scope": "actual release discovery, version comparison, asset selection and rate-limit helper; no GUI or installation"}
    (work / "result.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--libs", required=True, type=Path)
    parser.add_argument("--dotnet", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--patch", type=Path, default=HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch")
    print(json.dumps(run(parser.parse_args()), indent=2))
