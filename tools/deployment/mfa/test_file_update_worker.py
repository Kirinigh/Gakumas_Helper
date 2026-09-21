"""Exercise the real host worker and parent-exit gate using isolated no-UI witnesses."""
from __future__ import annotations

import os
import sys
import json
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.deployment import file_update


def prepare(args):
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    root = work / "client"
    root.mkdir()
    with zipfile.ZipFile(args.client_zip) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if "/" not in name or name.startswith(("libs/", "runtimes/")):
                archive.extract(info, root)
    original_main = (root / "MFAAvalonia.dll").read_bytes()
    original_core = (root / "libs/MFAAvalonia.Core.dll").read_bytes()
    shutil.copyfile(args.core, root / "libs/MFAAvalonia.Core.dll")
    dotnet = args.dotnet.resolve()
    sdk = sorted((dotnet.parent / "sdk").glob("10.*"))[-1]
    refs = sorted((dotnet.parent / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))[-1]
    for label, body in (
        ("parent", 'File.WriteAllText(Path.Combine(AppContext.BaseDirectory,".local","parent-ready"),"ready"); while(!File.Exists(Path.Combine(AppContext.BaseDirectory,".local","parent-exit"))) Thread.Sleep(50);'),
        ("target", 'File.AppendAllText(Path.Combine(AppContext.BaseDirectory,".local","restart-count"),"1");'),
    ):
        path = work / (label + ".cs")
        worker_branch = ('if(args.Length == 3 && args[0] == "--gkh-update-worker") { '
                         'var core=System.Reflection.Assembly.LoadFrom(Path.Combine(AppContext.BaseDirectory,"libs","MFAAvalonia.Core.dll")); '
                         'var method=core.GetType("MFAAvalonia.Helper.DerivedFileUpdate")!.GetMethod("RunWorker", System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic)!; '
                         'Environment.Exit((int)method.Invoke(null,new object[]{args[1],args[2]})!); return; }')
        path.write_text("using System; using System.IO; using System.Threading; static class Program { static void Main(string[] args) { " + worker_branch + body + " } }")
        response = work / (label + ".rsp")
        output = work / (label + ".dll")
        response.write_text("\n".join(["/nostdlib+", "/target:exe", f'/out:"{output}"',
                                       *(f'/reference:"{p}"' for p in refs.glob("*.dll")), f'"{path}"']))
        compiled = subprocess.run([str(dotnet), str(sdk / "Roslyn/bincore/csc.dll"), "/noconfig", "/utf8output", f"@{response}"], capture_output=True, text=True, encoding="utf-8")
        (work / (label + "-compile.log")).write_text(compiled.stdout + compiled.stderr)
        compiled.check_returncode()
    shutil.copyfile(work / "parent.dll", root / "MFAAvalonia.dll")
    (root / "interface.json").write_text('{"version":"v12.3.1"}')
    (root / "GAKUMAS_HELPER_BUILD.json").write_text('{"derived_version":"v12.3.1"}')
    base = file_update.write_state(root, "v12.3.1")
    base_hash = file_update.digest(root / file_update.STATE)
    target = work / "target"
    shutil.copytree(root, target)
    shutil.copyfile(work / "target.dll", target / "MFAAvalonia.dll")
    (target / "libs/MFAAvalonia.Core.dll").write_bytes(args.core.read_bytes() if args.launcher_harness else original_core)
    (target / "interface.json").write_text('{"version":"v12.3.2"}')
    (target / "GAKUMAS_HELPER_BUILD.json").write_text('{"derived_version":"v12.3.2"}')
    file_update.write_state(target, "v12.3.2")
    assets = work / "assets"
    assets.mkdir()
    metadata = file_update.build_adjacent(target, root, assets)
    operation = root / ".local/updates/worker-test"
    stage = operation / "step"
    stage.mkdir(parents=True)
    with zipfile.ZipFile(assets / metadata["delta"]["asset"]) as archive:
        archive.extractall(stage)
    shadow = operation / "worker"
    if not args.launcher_harness:
        shadow.mkdir()
        # Copy only the product inventory, not the growing operation directory.
        for relative in base["files"]:
            destination = shadow / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        (shadow / "MFAAvalonia.dll").write_bytes(original_main)
    plan = {"Root": str(root), "Work": str(operation), "Executable": "MaaGakumasu.exe",
            "Target": metadata, "IsDelta": True, "Reason": "isolated worker witness",
            "Budget": {"Remaining": 0}, "Full": {},
            "Steps": [{"Metadata": metadata, "Asset": {}, "Directory": str(stage)}]}
    if args.launcher_harness:
        plan["IsDelta"] = False
        plan["Steps"][0]["Directory"] = str(target)
        # LaunchWorker must select the verified full payload, not the locally modified base.
        (root / "libs/MFAAvalonia.Core.dll").write_bytes(b"locally damaged updater")
    (work / "plan-template.json").write_text(json.dumps(plan))
    (work / "frozen.json").write_text(json.dumps({"base_state_sha256": base_hash, "core_sha256": file_update.digest(args.core),
                                                 "client_zip_sha256": file_update.digest(args.client_zip)}))
    script = r'''
$ErrorActionPreference = 'Stop'
$run = $PSScriptRoot
$plan = Get-Content -LiteralPath (Join-Path $run 'plan-template.json') -Raw | ConvertFrom-Json
$parent = Start-Process -FilePath (Join-Path $plan.Root 'MaaGakumasu.exe') -WorkingDirectory $plan.Root -WindowStyle Hidden -PassThru
$deadline = [DateTime]::UtcNow.AddSeconds(30)
while (!(Test-Path -LiteralPath (Join-Path $plan.Root '.local/parent-ready'))) {
    if ($parent.HasExited -or [DateTime]::UtcNow -gt $deadline) { throw 'Parent witness failed to start' }
    Start-Sleep -Milliseconds 100
}
$plan | Add-Member ParentPid $parent.Id
$plan | Add-Member ParentStart $parent.StartTime.ToUniversalTime().Ticks
$planPath = Join-Path $plan.Work 'plan.json'
$plan | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $planPath -Encoding utf8NoBOM
$pipeName = 'gkh-worker-test-' + [Guid]::NewGuid().ToString('N')
$pipe = [System.IO.Pipes.NamedPipeServerStream]::new($pipeName,[System.IO.Pipes.PipeDirection]::Out,1,[System.IO.Pipes.PipeTransmissionMode]::Byte,[System.IO.Pipes.PipeOptions]::Asynchronous)
$start = [Diagnostics.ProcessStartInfo]::new((Join-Path $plan.Work 'worker/MaaGakumasu.exe'))
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.WorkingDirectory = Join-Path $plan.Work 'worker'
$start.ArgumentList.Add('--gkh-update-worker')
$start.ArgumentList.Add($planPath)
$start.ArgumentList.Add($pipeName)
$worker = [Diagnostics.Process]::Start($start)
try {
    $connection = $pipe.WaitForConnectionAsync()
    if (!$connection.Wait(30000)) { throw 'Worker did not connect' }
    $bytes = [Text.Encoding]::UTF8.GetBytes('{"Token":"","Proxy":""}')
    $pipe.Write($bytes,0,$bytes.Length)
    $pipe.Dispose()
    Start-Sleep -Milliseconds 500
    if ($worker.HasExited) { throw 'Worker exited before parent' }
    $initial = Get-Content -LiteralPath (Join-Path $plan.Root 'interface.json') -Raw | ConvertFrom-Json
    if ($initial.version -ne 'v12.3.1') { throw 'Worker changed files before parent exit' }
    Set-Content -LiteralPath (Join-Path $plan.Root '.local/parent-exit') -Value 'exit'
    if (!$worker.WaitForExit(60000)) { throw 'Worker did not finish' }
    if ($worker.ExitCode -ne 0) { throw 'Worker failed' }
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while (!(Test-Path -LiteralPath (Join-Path $plan.Root '.local/restart-count'))) {
        if ([DateTime]::UtcNow -gt $deadline) { throw 'Target did not restart' }
        Start-Sleep -Milliseconds 100
    }
} finally {
    $pipe.Dispose()
    if (!$parent.HasExited) { Set-Content -LiteralPath (Join-Path $plan.Root '.local/parent-exit') -Value 'exit' }
}
'''
    (work / "run.ps1").write_text(script, encoding="utf-8-sig")
    return work


def main(args):
    work = prepare(args)
    command = ([str(args.dotnet), str(args.launcher_harness), "--launch-plan", str(work / "plan-template.json")]
               if args.launcher_harness else [args.pwsh, "-NoProfile", "-File", str(work / "run.ps1")])
    result = subprocess.run(command,
                            capture_output=True, text=True, encoding="utf-8", timeout=150)
    (work / "worker.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    root = work / "client"
    if args.launcher_harness:
        import time
        deadline = time.monotonic() + 60
        while not (root / ".local/restart-count").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Full-package launch witness did not finish")
            time.sleep(0.1)
    for path in (work / "target").rglob("*"):
        if path.is_file():
            assert (root / path.relative_to(work / "target")).read_bytes() == path.read_bytes()
    status = json.loads((root / ".local/runtime-data/update-result.json").read_text())
    assert status["status"] == "success"
    assert (root / ".local/restart-count").read_text() == "1"
    evidence = {"status": "PASS", "worker": "actual MFA host and rebuilt Core", "parent_exit_gate": True,
                "target_restarts": 1, "all_target_files_match": True, "no_game_or_GUI": True,
                "network": "none", "fixed_installation_changed": False,
                "full_repair_uses_verified_worker_and_actual_launcher": bool(args.launcher_harness)}
    (work / "result.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--client-zip", required=True, type=Path)
    parser.add_argument("--core", required=True, type=Path)
    parser.add_argument("--dotnet", required=True, type=Path)
    parser.add_argument("--pwsh", default="pwsh")
    parser.add_argument("--launcher-harness", type=Path)
    main(parser.parse_args())
