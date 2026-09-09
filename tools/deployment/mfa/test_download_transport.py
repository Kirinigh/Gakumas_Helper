"""Run the pinned, patched MFA download methods against loopback HTTP failures.

Requires an existing .NET 10 SDK; never restores or downloads dependencies.
The C# fixture supplies only UI/proxy seams. Production method bodies and the
resource download/verification gate are extracted verbatim after git apply.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

SOURCE_SHA256 = "76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A"
CHECKER_BLOB = "6e6d1118fa414ba21c7efa4f15a58ad95dd08bd7"
HERE = Path(__file__).resolve().parent


def extract_method(source: str, name: str) -> str:
    start = re.search(rf"^    (?:public|private|async)[^\n=]*\b{re.escape(name)}\(", source, re.MULTILINE)
    if start is None:
        raise ValueError(f"Missing production method: {name}")
    end = source.index("\n    }", start.start()) + len("\n    }")
    return source[start.start():end]


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
    checker = source_root / "MFAAvalonia/Helper/VersionChecker.cs"
    original = checker.read_bytes()
    if hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest() != CHECKER_BLOB:
        raise ValueError("The pinned VersionChecker Git object does not match")
    env = os.environ.copy()
    # Prevent git apply from discovering a surrounding GKH repository and
    # silently skipping the upstream paths because of a cwd prefix.
    env["GIT_CEILING_DIRECTORIES"] = str(work)
    env["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    env["DOTNET_SKIP_FIRST_TIME_EXPERIENCE"] = "1"
    env["DOTNET_GENERATE_ASPNET_CERTIFICATE"] = "false"
    env["DOTNET_ADD_GLOBAL_TOOLS_TO_PATH"] = "false"
    env["DOTNET_CLI_HOME"] = str(work / "dotnet-home")
    patch = HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
    for options in (["--check"], []):
        subprocess.run(["git", "apply", *options, "--", str(patch)], cwd=source_root, env=env, check=True)
    if not (source_root / "MFAAvalonia/Directory.Build.targets").is_file():
        raise ValueError("The source patch was skipped")
    source = checker.read_text(encoding="utf-8-sig")
    names = ["DownloadWithRetry", "DownloadFileAsync", "ExecuteTasksAsync",
             "IsGitHubReleaseAssetApiUrl", "ParseFileNameFromContentDisposition",
             "VerifyFileSha256Async"]
    methods = "\n\n".join(extract_method(source, name) for name in names)
    resource = extract_method(source, "UpdateResource")
    start = resource.index("        if (!isLocalPackage)\n        {", resource.index("var tempZipFilePath"))
    end = resource.index("        var changesPath =", start)
    gate = resource[start:end]
    processor = (source_root / "MFAAvalonia/Extensions/MaaFW/MaaProcessor.cs").read_text(encoding="utf-8-sig")
    cleanup_start = processor.index("                var tempMFADir =")
    cleanup_end = processor.index("\n            }", cleanup_start)
    cleanup = processor[cleanup_start:cleanup_end]
    fixture = (HERE / "download_transport_harness.cs").read_text(encoding="utf-8")
    fixture = fixture.replace("// PRODUCTION_METHODS", methods).replace("// PRODUCTION_RESOURCE_GATE", gate)
    fixture = fixture.replace("// PRODUCTION_TASK_CLEANUP", cleanup)
    fixture_path = work / "Program.cs"
    fixture_path.write_text(fixture, encoding="utf-8")
    sdk_root = dotnet.parent
    sdk_dirs = sorted((sdk_root / "sdk").glob("10.*"))
    ref_dirs = sorted((sdk_root / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))
    runtime_dirs = sorted((sdk_root / "shared/Microsoft.NETCore.App").glob("10.*"))
    if not sdk_dirs or not ref_dirs or not runtime_dirs:
        raise ValueError("A complete, already installed .NET 10 SDK is required")
    output = work / "DownloadTransportHarness.dll"
    response = work / "compile.rsp"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable",
        f'/out:"{output}"',
        *(f'/reference:"{item}"' for item in sorted(ref_dirs[-1].glob("*.dll"))),
        f'"{fixture_path}"',
    ]), encoding="utf-8")
    (work / "DownloadTransportHarness.runtimeconfig.json").write_text(json.dumps({
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
    (work / "transport.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    evidence = {
        "source_sha256": SOURCE_SHA256,
        "original_checker_blob": CHECKER_BLOB,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest().upper(),
        "extracted_methods": names,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "result": result.stdout.strip(),
        "network": "loopback only; no dependency restore",
    }
    (work / "result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--dotnet", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
