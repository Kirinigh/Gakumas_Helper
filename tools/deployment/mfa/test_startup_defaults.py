"""Compile patched configuration/layout methods against an offline test harness.

Configuration file I/O and dictionary conversion use a small Newtonsoft adapter;
rendering and save callbacks are recorded. The actual Core is built separately.
"""
from __future__ import annotations

import os
import json
import shutil
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

from test_dynamic_option_cases import SOURCE_SHA256, extract_method

HERE = Path(__file__).resolve().parent


def run(args: argparse.Namespace) -> dict:
    archive, dotnet, work = args.source_zip.resolve(), args.dotnet.resolve(), args.work_dir.resolve()
    if work.exists():
        raise ValueError("work-dir must be new")
    if hashlib.sha256(archive.read_bytes()).hexdigest().upper() != SOURCE_SHA256:
        raise ValueError("Unexpected pinned source archive")
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(work)
    source = work / "MFAAvalonia-2.15.2/MFAAvalonia"
    patch = HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(work), DOTNET_CLI_TELEMETRY_OPTOUT="1")
    for options in (["--check"], []):
        subprocess.run(["git", "apply", *options, "--", str(patch)], cwd=source.parent, env=env, check=True)
    config = (source / "Configuration/ConfigurationManager.cs").read_text(encoding="utf-8-sig")
    layout = (source / "Views/UserControls/Dashboard/DashboardCardGrid.cs").read_text(encoding="utf-8-sig")
    instance = (source / "Configuration/InstanceConfiguration.cs").read_text(encoding="utf-8-sig")
    methods = {
        "ConfigurationManager": [extract_method(config, signature) for signature in (
            "    private static AvaloniaList<MFAConfiguration> LoadConfigurations()",
            "    private static void ApplyMissingTemplateDefaults(Dictionary<string, object> config)",
            "    public static MFAConfiguration Add(string name)",
        )],
        "DashboardCardGrid": [extract_method(layout, signature) for signature in (
            "    private void EnsureLayoutsLoaded()",
            "    private List<DashboardCardLayout> LoadConfigLayouts(",
        )],
        "InstanceConfiguration": [extract_method(instance, "    public T GetValue<T>(string key, T defaultValue)")],
    }
    production = work / "ProductionMethods.cs"
    production.write_text(
        "using System; using System.IO; using System.Linq; using System.Text.Json; using System.Collections.Generic;\n"
        + "\n".join(f"public partial class {name} {{\n" + "\n".join(bodies) + "\n}" for name, bodies in methods.items()),
        encoding="utf-8",
    )
    sdk = sorted((dotnet.parent / "sdk").glob("10.*"))[-1]
    refs = sorted((dotnet.parent / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))[-1]
    runtime = sorted((dotnet.parent / "shared/Microsoft.NETCore.App").glob("10.*"))[-1]
    newtonsoft = args.newtonsoft.resolve()
    shutil.copyfile(newtonsoft, work / "Newtonsoft.Json.dll")
    response = work / "compile.rsp"
    output = work / "StartupDefaultsHarness.dll"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable",
        f'/out:"{output}"', *(f'/reference:"{item}"' for item in sorted(refs.glob("*.dll"))),
        f'/reference:"{newtonsoft}"', f'"{production}"', f'"{HERE / "startup_defaults_harness.cs"}"',
    ]), encoding="utf-8")
    (work / "StartupDefaultsHarness.runtimeconfig.json").write_text(json.dumps({
        "runtimeOptions": {"tfm": "net10.0", "framework": {"name": "Microsoft.NETCore.App", "version": runtime.name}},
    }), encoding="utf-8")
    result = subprocess.run([str(dotnet), str(sdk / "Roslyn/bincore/csc.dll"), "/noconfig", f"@{response}"],
                            cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (work / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    result = subprocess.run([str(dotnet), str(output), str(work / "cases")], cwd=work, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    (work / "sequence.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    evidence = {"status": "PASS", "source_sha256": SOURCE_SHA256,
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest().upper(),
                "production_methods_sha256": hashlib.sha256(production.read_bytes()).hexdigest().upper(),
                "newtonsoft_sha256": hashlib.sha256(newtonsoft.read_bytes()).hexdigest().upper(),
                "result": result.stdout.strip(), "network": "none; no restore"}
    (work / "result.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "dotnet", "newtonsoft", "work-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    print(json.dumps(run(parser.parse_args()), indent=2))
