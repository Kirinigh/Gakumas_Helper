"""Compile the production cooldown helper and test real cross-process state/locks."""
from __future__ import annotations

import os
import json
import time
import shutil
import hashlib
import zipfile
import argparse
import subprocess
import importlib.util
from pathlib import Path

from test_github_update import SOURCE_SHA256

HERE = Path(__file__).resolve().parent


def run(args):
    archive, dotnet, work = args.source_zip.resolve(), args.dotnet.resolve(), args.work_dir.resolve()
    if os.name != "nt":
        raise RuntimeError("The shared byte-range lock contract requires Windows")
    import msvcrt

    if work.exists() or hashlib.sha256(archive.read_bytes()).hexdigest().upper() != SOURCE_SHA256:
        raise ValueError("Use a new work directory and the pinned source ZIP")
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(work)
    source = work / "MFAAvalonia-2.15.2"
    patch = args.patch.resolve()
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(work), DOTNET_CLI_TELEMETRY_OPTOUT="1")
    for options in (["--check"], []):
        subprocess.run(["git", "apply", *options, "--", str(patch)], cwd=source, env=env, check=True)
    helper = source / "MFAAvalonia/Helper/GitHubApiRequests.cs"
    sdk = sorted((dotnet.parent / "sdk").glob("10.*"))[-1]
    refs = sorted((dotnet.parent / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))[-1]
    runtime = sorted((dotnet.parent / "shared/Microsoft.NETCore.App").glob("10.*"))[-1]
    assembly = work / "RateLimitStateHarness.dll"
    response = work / "compile.rsp"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable",
        f'/out:"{assembly}"', *(f'/reference:"{p}"' for p in sorted(refs.glob("*.dll"))),
        f'"{helper}"', f'"{HERE / "rate_limit_state_harness.cs"}"',
    ]), encoding="utf-8")
    compiled = subprocess.run([str(dotnet), str(sdk / "Roslyn/bincore/csc.dll"), "/noconfig", f"@{response}"],
                              cwd=work, env=env, capture_output=True, text=True, encoding="utf-8")
    (work / "compile.log").write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
    compiled.check_returncode()
    config = {"runtimeOptions": {"tfm": "net10.0", "framework": {
        "name": "Microsoft.NETCore.App", "version": runtime.name}}}
    results = []

    def case(name, state=None, *, bom=False):
        directory = work / "cases" / name
        directory.mkdir(parents=True)
        shutil.copy2(assembly, directory / assembly.name)
        (directory / "RateLimitStateHarness.runtimeconfig.json").write_text(json.dumps(config), encoding="utf-8")
        file = directory / ".local/runtime-data/github-api-rate-limit.json"
        file.parent.mkdir(parents=True)
        if state is not None:
            file.write_text(json.dumps(state), encoding="utf-8-sig" if bom else "utf-8")
        return directory, file

    def command(directory, mode="success", delay=120, auth="anon", host="github", *extra):
        return [str(dotnet), str(directory / assembly.name), mode, str(delay), auth, host, *extra]

    def invoke(directory, *options):
        result = subprocess.run(command(directory, *options), env=env, cwd=directory,
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        result.check_returncode()
        return json.loads(result.stdout.strip().splitlines()[-1])

    def check(name, condition, observation):
        if not condition:
            raise AssertionError((name, observation))
        results.append({"case": name, "result": observation})

    future = time.time() + 600
    for name, state, bom in [
        ("python_bom_future", {"schema_version": 1, "resume_at": future}, True),
        ("python_fractional_future", {"schema_version": 1, "resume_at": future + .123}, False),
    ]:
        directory, file = case(name, state, bom=bom)
        original = file.read_bytes()
        result = invoke(directory)
        check(name, result["calls"] == 0 and result["first_limited"] and result["second_limited"]
              and file.read_bytes() == original, result)
    for name, state in [
        ("missing", None), ("expired", {"schema_version": 1, "resume_at": 1}),
        ("invalid_schema", {"schema_version": 2, "resume_at": future}),
        ("boolean_schema", {"schema_version": True, "resume_at": future}),
        ("negative", {"schema_version": 1, "resume_at": -1}),
        ("out_of_range", {"schema_version": 1, "resume_at": 253402300800}),
        ("boolean_epoch", {"schema_version": 1, "resume_at": True}),
        ("not_finite", {"schema_version": 1, "resume_at": float("inf")}),
    ]:
        directory, file = case(name, state)
        original = file.read_bytes() if file.exists() else None
        result = invoke(directory)
        check(name, result["calls"] == 2 and not result["first_limited"]
              and (file.read_bytes() if file.exists() else None) == original, result)
    directory, file = case("corruption_keeps_memory", {"schema_version": 1, "resume_at": future})
    result = invoke(directory, "corrupt-after")
    check("corruption_keeps_memory", result["calls"] == 0 and result["second_limited"]
          and result["diagnostics"] > 0, result)
    directory, file = case("anonymous_persistence")
    result = invoke(directory, "limit")
    saved = json.loads(file.read_text(encoding="utf-8"))
    check("anonymous_persistence", result["calls"] == 1 and result["second_limited"]
          and set(saved) == {"schema_version", "resume_at"} and saved["resume_at"] > time.time(), result)
    result = invoke(directory)
    check("new_process_reuses_persistence", result["calls"] == 0 and result["first_limited"], result)
    for mode in ("success", "limit"):
        directory, file = case("auth_" + mode, {"schema_version": 1, "resume_at": future})
        original = file.read_bytes()
        result = invoke(directory, mode, 120, "auth")
        check("auth_" + mode, result["calls"] == (2 if mode == "success" else 1)
              and file.read_bytes() == original, result)
    directory, file = case("non_api_scope")
    result = invoke(directory, "limit", 120, "anon", "other")
    check("non_api_scope", result["calls"] == 1 and not file.exists(), result)
    directory, file = case("larger_racing_deadline")
    result = invoke(directory, "race")
    saved = json.loads(file.read_text(encoding="utf-8"))
    check("larger_racing_deadline", result["calls"] == 1 and saved["resume_at"] > time.time() + 3500, result)
    directory, file = case("write_failure_keeps_memory")
    file.mkdir()
    result = invoke(directory, "limit")
    check("write_failure_keeps_memory", result["calls"] == 1 and result["second_limited"]
          and result["diagnostics"] > 0 and file.is_dir(), result)
    directory, file = case("python_lock_contention")
    lock = file.with_suffix(".lock")
    with lock.open("w+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            result = invoke(directory, "limit")
            check("python_lock_contention", result["calls"] == 1 and result["second_limited"]
                  and result["diagnostics"] > 0 and result["elapsed_ms"] < 2000 and not file.exists(), result)
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    directory, file = case("concurrent_process_max")
    processes = [subprocess.Popen(command(directory, "barrier", delay, "anon", "github", name),
                                 env=env, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, encoding="utf-8")
                 for delay, name in [(120, "short.ready"), (3600, "long.ready")]]
    try:
        deadline = time.monotonic() + 10
        while not all((directory / name).exists() for name in ("short.ready", "long.ready")):
            if time.monotonic() > deadline:
                raise TimeoutError("Cross-process barrier did not become ready")
            time.sleep(.02)
        (directory / "go").write_text("go", encoding="ascii")
        observations = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            if process.returncode:
                raise RuntimeError(stderr)
            observations.append(json.loads(stdout.strip().splitlines()[-1]))
        saved = json.loads(file.read_text(encoding="utf-8"))
        check("concurrent_process_max", saved["resume_at"] > time.time() + 3500
              and all(row["calls"] == 1 for row in observations), observations)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    peer_sha = None
    if args.python_helper is not None:
        peer_path = args.python_helper.resolve()
        peer_sha = hashlib.sha256(peer_path.read_bytes()).hexdigest().upper()
        spec = importlib.util.spec_from_file_location("rate_limit_peer", peer_path)
        peer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(peer_module)
        directory, file = case("production_python_to_core")
        peer = peer_module.GitHubRateLimit(file)
        peer.record(future)
        with peer._file_lock():
            result = invoke(directory)
        check("production_python_to_core_unlocked_read", result["calls"] == 0
              and result["first_limited"], result)
        directory, file = case("production_core_to_python")
        result = invoke(directory, "limit")
        peer = peer_module.GitHubRateLimit(file)
        try:
            peer.check()
            raise AssertionError("Python ignored Core cooldown")
        except peer_module.GitHubRateLimitError as error:
            check("production_core_to_python", error.resume_at > time.time(), result)
        directory, file = case("production_python_lock_blocks_core")
        peer = peer_module.GitHubRateLimit(file)
        with peer._file_lock():
            result = invoke(directory, "limit")
        check("production_python_lock_blocks_core", result["calls"] == 1 and result["second_limited"]
              and result["diagnostics"] > 0 and not file.exists(), result)
        directory, file = case("core_lock_blocks_production_python")
        process = subprocess.Popen(command(directory, "hold-lock"), cwd=directory, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        try:
            deadline = time.monotonic() + 10
            while not (directory / "locked.ready").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Core lock did not become ready")
                time.sleep(.02)
            errors = []
            peer = peer_module.GitHubRateLimit(file, on_error=errors.append)
            peer.record(future)
            try:
                peer.check()
                raise AssertionError("Python lost in-memory cooldown on contention")
            except peer_module.GitHubRateLimitError:
                check("core_lock_blocks_production_python", bool(errors) and not file.exists(),
                      {"diagnostics": len(errors)})
        finally:
            (directory / "go").write_text("go", encoding="ascii")
            stdout, stderr = process.communicate(timeout=15)
            if process.returncode:
                raise RuntimeError(stderr)
        directory, file = case("production_python_max_merge")
        process = subprocess.Popen(command(directory, "barrier", 120, "anon", "github", "short.ready"),
                                   cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8")
        try:
            deadline = time.monotonic() + 10
            while not (directory / "short.ready").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Core request did not reach barrier")
                time.sleep(.02)
            larger = time.time() + 3600
            peer_module.GitHubRateLimit(file).record(larger)
        finally:
            (directory / "go").write_text("go", encoding="ascii")
            stdout, stderr = process.communicate(timeout=15)
            if process.returncode:
                raise RuntimeError(stderr)
        saved = json.loads(file.read_text(encoding="utf-8"))
        check("production_python_max_merge", saved["resume_at"] >= larger - .001,
              {"expected_minimum": larger, "actual": saved["resume_at"]})
    evidence = {"status": "PASS", "cases": results, "source_sha256": SOURCE_SHA256,
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest().upper(),
                "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest().upper(),
                "python_peer_sha256": peer_sha, "network": "none", "installation_changed": False}
    (work / "result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "dotnet", "work-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--patch", type=Path, default=HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch")
    parser.add_argument("--python-helper", type=Path)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
