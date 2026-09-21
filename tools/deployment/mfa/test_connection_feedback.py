"""Replay pinned client's actual connection methods without devices or network."""
from __future__ import annotations

import os
import json
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

from test_dynamic_option_cases import SOURCE_SHA256, extract_method


def run(source_zip: Path, dotnet: Path, work: Path):
    assert hashlib.sha256(source_zip.read_bytes()).hexdigest().upper() == SOURCE_SHA256
    work.mkdir(exist_ok=False, parents=True)
    here = Path(__file__).resolve().parent
    patch = here / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
    source = work / "source"
    source.mkdir()
    # Only existing files changed by this patch, not another full source/build tree.
    paths = [line[6:] for line in patch.read_text(encoding="utf-8").splitlines() if line.startswith("--- a/")]
    with zipfile.ZipFile(source_zip) as archive:
        for rel in paths:
            entry = "MFAAvalonia-2.15.2/" + rel
            target = source / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(work), DOTNET_CLI_HOME=str(work / "home"), DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1")
    for flags in [("--check",), ()]:
        subprocess.run(["git", "apply", *flags, str(patch)], cwd=source, env=env, check=True, capture_output=True)
    text = (source / "MFAAvalonia/Extensions/MaaFW/MaaProcessor.cs").read_text(encoding="utf-8-sig")
    assert "recoIdToken != null && CanReadRecognitionResult(args.Message)" in text
    assert "connected = await RetryConnectionAsync(token, showMessage, StartSoftware" in text
    signatures = ["async private Task HandleDeviceConnectionAsync(", "async private Task<bool> RetryConnectionAsync(", "async private Task<(bool, bool, bool)> TryConnectAsync(", "private static string? ConnectionTargetError("]
    methods = [extract_method(text, sig) for sig in signatures]
    start = text.index("    private static bool CanReadRecognitionResult(")
    methods.append(text[start:text.index(";", start) + 1])
    extracted = work / "Methods.cs"
    extracted.write_text("using System; using System.Threading; using System.Threading.Tasks;\npartial class Harness {\n" + "\n".join(methods) + "\n}", encoding="utf-8")
    sdk = sorted((dotnet.parent / "sdk").iterdir())[-1]
    ref_pack = sorted((dotnet.parent / "packs/Microsoft.NETCore.App.Ref").iterdir())[-1]
    ref = next((ref_pack / "ref").iterdir())
    runtime = sorted((dotnet.parent / "shared/Microsoft.NETCore.App").iterdir())[-1]
    output = work / "ConnectionHarness.dll"
    rsp = work / "compile.rsp"
    rsp.write_text("\n".join(["/nostdlib+", "/target:exe", "/langversion:latest", "/nullable:enable", f'/out:"{output}"', *(f'/reference:"{p}"' for p in ref.glob("*.dll")), f'"{extracted}"', f'"{here / "connection_feedback_harness.cs"}"']), encoding="utf-8")
    (work / "ConnectionHarness.runtimeconfig.json").write_text(json.dumps({"runtimeOptions":{"tfm":ref.name,"framework":{"name":"Microsoft.NETCore.App","version":runtime.name}}}),encoding="utf-8")
    for name, command in [("compile", [str(dotnet), str(sdk / "Roslyn/bincore/csc.dll"), "/noconfig", f"@{rsp}"]), ("run", [str(dotnet), str(output)])]:
        result = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
        (work / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        print(result.stdout + result.stderr)
        result.check_returncode()
    (work / "result.json").write_text(json.dumps({"status":"PASS","implementation":"Actual HandleDeviceConnectionAsync, TryConnectAsync, RetryConnectionAsync and guards extracted from patched pinned source; external UI/native callbacks stubbed","native_input":False,"full_core_build":False},indent=2),encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip",type=Path,required=True)
    parser.add_argument("--dotnet",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,required=True)
    args = parser.parse_args()
    run(args.source_zip.resolve(),args.dotnet.resolve(),args.work_dir.resolve())
