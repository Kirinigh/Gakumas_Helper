"""Exercise actual MFA dynamic-case refresh and ComboBox event code offline.

Only rendering, language lookup, option-model plumbing, and configuration-save
callbacks are stubbed; the refresh helper and UI event method come from the patch.
Full Core builds verify their integration with real Avalonia and generated models.
"""
from __future__ import annotations

import os
import json
import time
import shutil
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

SOURCE_SHA256 = "76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A"
HERE = Path(__file__).resolve().parent


def extract_method(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    cursor = brace + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    archive, dotnet, work = args.source_zip.resolve(), args.dotnet.resolve(), args.work_dir.resolve()
    newtonsoft = args.newtonsoft.resolve()
    if bool(args.before) != bool(args.after):
        raise ValueError("before and after must be supplied together")
    if work.exists():
        raise ValueError("work-dir must be a fresh directory")
    if hashlib.sha256(archive.read_bytes()).hexdigest().upper() != SOURCE_SHA256:
        raise ValueError("Source ZIP is not pinned MFA v2.15.2")
    if not newtonsoft.is_file():
        raise ValueError("An existing Newtonsoft.Json DLL is required; no restore is performed")
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source_zip:
        source_zip.extractall(work)
    source_root = work / "MFAAvalonia-2.15.2"
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(work), DOTNET_CLI_TELEMETRY_OPTOUT="1",
               DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1", DOTNET_CLI_HOME=str(work / "dotnet-home"))
    patch = HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
    for options in (["--check"], []):
        subprocess.run(["git", "apply", *options, "--", str(patch)], cwd=source_root, env=env, check=True)
    model = (source_root / "MFAAvalonia/Extensions/MaaFW/MaaInterface.cs").read_text(encoding="utf-8-sig")
    if '[JsonProperty("dynamic_cases")]\n        public bool DynamicCases { get; set; }' not in model:
        raise ValueError("Production model did not opt in with default false")
    refresh_method = extract_method(model, "        private void UpdateDisplayName()")
    if "DisplayDescription =" not in refresh_method or "HasDescription =" not in refresh_method:
        raise ValueError("Display refresh must retain actual description behavior")
    if "internal void RefreshDisplayMetadata() => UpdateDisplayName();" not in model:
        raise ValueError("Display refresh must not resubscribe language events")
    if '[JsonProperty("label_args")]\n        public Dictionary<string, string>? LabelArgs { get; set; }' not in model:
        raise ValueError("Localized label arguments are absent from the production case model")
    label_source = work / "ProductionCaseLabel.cs"
    label_source.write_text(
        "using System;\nusing System.Text.RegularExpressions;\nusing MFAAvalonia.Helper;\n"
        "namespace MFAAvalonia.Extensions.MaaFW {\npublic partial class MaaInterface {\n"
        "public partial class MaaInterfaceOptionCase {\n" + refresh_method
        + "\ninternal void RefreshDisplayMetadata() => UpdateDisplayName();\n}}}\n", encoding="utf-8")
    generator = (source_root / "MFAAvalonia/Helper/TaskOptionGenerator.cs").read_text(encoding="utf-8-sig")
    method = extract_method(generator, "    private Control CreateComboBoxControl(")
    queue_source = (source_root / "MFAAvalonia/ViewModels/Pages/TaskQueueViewModel.cs").read_text(encoding="utf-8-sig")
    start_method = extract_method(queue_source, "    public void StartTask()")
    if not (start_method.index("if (IsRunning)") < start_method.index("RefreshDynamicSelectionsBeforeStart();")
            < start_method.index("Processor.Start();")):
        raise ValueError("Unopened options must migrate before task execution and after the running guard")
    loader_source = (source_root / "MFAAvalonia/Extensions/MaaFW/TaskLoader.cs").read_text(encoding="utf-8-sig")
    if "DynamicOptionCases.ResolveSelection(option, io);\n            EnsureDefaultSubOptions" not in loader_source:
        raise ValueError("Configuration default loading must use the same resolver before sub-option expansion")
    queue_method = extract_method(queue_source, "    private void RefreshDynamicSelectionsBeforeStart()")
    queue_extracted = work / "ProductionBeforeStart.cs"
    queue_extracted.write_text(
        "using System.Linq;\nusing MFAAvalonia.Helper;\nusing MFAAvalonia.Extensions.MaaFW;\n"
        "namespace MFAAvalonia.ViewModels.Pages {\npublic partial class TaskQueueViewModel {\n"
        + queue_method + "\n}}\n", encoding="utf-8")
    extracted = work / "ProductionComboBox.cs"
    extracted.write_text(
        "using System;\nusing System.Linq;\nusing System.Collections.Generic;\n"
        "using MFAAvalonia.Extensions.MaaFW;\nnamespace MFAAvalonia.Helper {\n"
        "public partial class TaskOptionGenerator {\n" + method + "\n}}\n", encoding="utf-8")
    helper = source_root / "MFAAvalonia/Helper/DynamicOptionCases.cs"
    sdk_root = dotnet.parent
    sdk = sorted((sdk_root / "sdk").glob("10.*"))[-1]
    refs = sorted((sdk_root / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))[-1]
    runtime = sorted((sdk_root / "shared/Microsoft.NETCore.App").glob("10.*"))[-1]
    shutil.copyfile(newtonsoft, work / "Newtonsoft.Json.dll")
    output = work / "DynamicCasesHarness.dll"
    response = work / "compile.rsp"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable",
        f'/out:"{output}"',
        *(f'/reference:"{item}"' for item in sorted(refs.glob("*.dll"))),
        f'/reference:"{newtonsoft}"', f'"{helper}"', f'"{extracted}"', f'"{label_source}"', f'"{queue_extracted}"',
        f'"{HERE / "dynamic_option_cases_harness.cs"}"',
    ]), encoding="utf-8")
    (work / "DynamicCasesHarness.runtimeconfig.json").write_text(json.dumps({
        "runtimeOptions": {"tfm": "net10.0", "framework": {
            "name": "Microsoft.NETCore.App", "version": runtime.name,
        }},
    }), encoding="utf-8")
    compile_result = subprocess.run([
        str(dotnet), str(sdk / "Roslyn/bincore/csc.dll"), "/noconfig", f"@{response}",
    ], cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (work / "compile.log").write_text(compile_result.stdout + compile_result.stderr, encoding="utf-8")
    compile_result.check_returncode()
    language_files = [HERE.parents[2] / "assets/lang" / name for name in ("zh-CN.json", "zh-Hant.json")]
    command = [str(dotnet), str(output), str(work / "cases"), *(str(path) for path in language_files)]
    if args.before and args.after:
        command.extend([str(args.before.resolve()), str(args.after.resolve())])
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
    (work / "sequence.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    evidence = {
        "status": "PASS",
        "source_sha256": SOURCE_SHA256,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest().upper(),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest().upper(),
        "combobox_method_sha256": hashlib.sha256(method.encode()).hexdigest().upper(),
        "newtonsoft_sha256": hashlib.sha256(newtonsoft.read_bytes()).hexdigest().upper(),
        "implementation": "complete production DynamicOptionCases.cs plus extracted CreateComboBoxControl and case UpdateDisplayName",
        "label_method_sha256": hashlib.sha256(refresh_method.encode()).hexdigest().upper(),
        "language_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper()}
                           for path in language_files],
        "seams": "rendering, generated option model, language localization, idle binding, save callback",
        "result": result.stdout.strip(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "network": "none; no restore",
    }
    (work / "result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--dotnet", required=True, type=Path)
    parser.add_argument("--newtonsoft", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
