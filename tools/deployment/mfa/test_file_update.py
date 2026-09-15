"""Compile actual patched planner/transaction and run isolated file-update scenarios."""
from __future__ import annotations

import os
import sys
import json
import shutil
import hashlib
import zipfile
import argparse
import subprocess
from pathlib import Path

from test_download_transport import SOURCE_SHA256

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.deployment import file_update

HERE = Path(__file__).resolve().parent


def run(args):
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if file_update.digest(args.source_zip).upper() != SOURCE_SHA256:
        raise ValueError("Unexpected pinned MFA source")
    with zipfile.ZipFile(args.source_zip) as archive:
        archive.extractall(work)
    source = work / "MFAAvalonia-2.15.2"
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(work), DOTNET_CLI_TELEMETRY_OPTOUT="1")
    patch = HERE / "MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, env=env, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=source, env=env, check=True)
    checker = (source / "MFAAvalonia/Helper/VersionChecker.cs").read_text(encoding="utf-8-sig")
    start = checker.index("    internal sealed class UpdateFileTransaction")
    end = checker.index("    static void DeleteFileWithBackup", start)
    fixture = (HERE / "file_update_harness.cs").read_text().replace("// PRODUCTION_TRANSACTION", checker[start:end])
    (work / "Program.cs").write_text(fixture, encoding="utf-8")
    dotnet = args.dotnet.resolve()
    sdk = sorted((dotnet.parent / "sdk").glob("10.*"))[-1]
    refs = sorted((dotnet.parent / "packs/Microsoft.NETCore.App.Ref").glob("10.*/ref/net10.0"))[-1]
    runtime = sorted((dotnet.parent / "shared/Microsoft.NETCore.App").glob("10.*"))[-1]
    output = work / "FileUpdateHarness.dll"
    response = work / "compile.rsp"
    response.write_text("\n".join([
        "/nostdlib+", "/target:exe", "/langversion:14", "/nullable:enable", f'/out:"{output}"',
        *(f'/reference:"{p}"' for p in sorted(refs.glob("*.dll"))), f'"{work / "Program.cs"}"',
        *(f'"{source / "MFAAvalonia/Helper" / name}"' for name in ("GitHubApiRequests.cs", "DerivedFileUpdate.cs")),
    ]), encoding="utf-8")
    (work / "FileUpdateHarness.runtimeconfig.json").write_text(json.dumps({"runtimeOptions": {
        "tfm": "net10.0", "framework": {"name": "Microsoft.NETCore.App", "version": runtime.name}}}))
    compiled = subprocess.run([str(dotnet), str(sdk / "Roslyn/bincore/csc.dll"), "/noconfig", f"@{response}"],
                              capture_output=True, text=True, encoding="utf-8", env=env)
    (work / "compile.log").write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
    compiled.check_returncode()
    releases, assets, packages = [], {}, {}
    payload = hashlib.shake_256(b"unchanged-runtime").digest(131072)
    for index in range(1, 5):
        version = f"v0.5.{index}"
        package = work / "packages" / version
        package.mkdir(parents=True)
        (package / "interface.json").write_text(json.dumps({"version": version}))
        (package / "GAKUMAS_HELPER_BUILD.json").write_text(json.dumps({"derived_version": version}))
        (package / "runtime.dll").write_bytes(payload)
        (package / "MaaGakumasu.exe").write_bytes(f"exe-{index}".encode())
        (package / "MFAAvalonia.Core.dll").write_bytes(f"core-{index}".encode())
        (package / "code.py").write_text(f"version = {index}")
        if index == 1:
            (package / "transition").write_text("file before directory transition")
        else:
            (package / "transition").mkdir()
            (package / "transition/module.py").write_text(f"value = {index}")
        if index < 3:
            (package / "obsolete.py").write_text("old")
        file_update.write_state(package, version)
        package_assets = work / "assets" / version
        package_assets.mkdir(parents=True)
        update = file_update.build_adjacent(package, packages.get(f"v0.5.{index - 1}"), package_assets)
        full = package_assets / f"MaaGakumasu-win-x86_64-{version}.zip"
        with zipfile.ZipFile(full, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in package.rglob("*"):
                if path.is_file():
                    zipped.write(path, path.relative_to(package).as_posix())
        manifest = package_assets / f"GakumasHelper-release-{version}.json"
        manifest.write_text(json.dumps({"file_update": update}))
        release = {"tag_name": version, "assets": []}
        for path in package_assets.iterdir():
            identity = str(len(assets) + 1)
            assets[identity] = str(path)
            release["assets"].append({"name": path.name, "size": path.stat().st_size,
                                       "digest": "sha256:" + file_update.digest(path),
                                       "url": "https://api.github.com/repos/example/project/releases/assets/" + identity})
        releases.append(release)
        packages[version] = package
    results = []
    scenarios = [("adjacent", "v0.5.3", "execute", True, 1),
                 ("chain", "v0.5.1", "execute", True, 3),
                 ("modified", "v0.5.1", "execute", False, 1),
                 ("missing", "v0.5.1", "execute", False, 1),
                 ("cross_minor", "v0.4.9", "execute", False, 1),
                 ("rollback_fallback", "v0.5.1", "inject", True, 3),
                 ("locked_replace", "v0.5.1", "lock-recover", True, 3),
                 ("locked_delete", "v0.5.1", "lock-recover", True, 3),
                 ("explicit_full", "v0.5.1", "execute", False, 1),
                 ("size_full", "v0.5.3", "execute", False, 1),
                 ("corrupt_delta", "v0.5.1", "execute", False, 1),
                 ("extra_local_code", "v0.5.1", "execute", False, 1),
                 ("corrupt_full", "v0.4.9", "execute", False, 1),
                 ("budget_exhausted", "v0.5.1", "execute", False, 1),
                 ("rate_limited", "v0.5.1", "execute", False, 1)]
    for name, base_version, mode, delta, count in scenarios:
        case = work / name
        root = case / "client"
        shutil.copytree(packages.get(base_version, packages["v0.5.1"]), root)
        for relative in ("config/settings.json", "logs/old.log", ".local/runtime-data/cache.json", "agent/__pycache__/cache.pyc"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("user data")
        if name == "modified":
            (root / "code.py").write_text("local patch")
        if name == "extra_local_code":
            (root / "local_patch.py").write_text("local patch")
        catalog = json.loads(json.dumps([r for r in releases if name != "missing" or r["tag_name"] != "v0.5.2"]))
        case_assets = dict(assets)
        if name in {"explicit_full", "size_full", "corrupt_delta"}:
            release = catalog[-1]
            metadata_asset = next(a for a in release["assets"] if a["name"].endswith(".json"))
            identity = metadata_asset["url"].rsplit("/", 1)[-1]
            document = json.loads(Path(case_assets[identity]).read_text())
            if name == "explicit_full":
                document["file_update"]["delta"] = None
            elif name == "size_full":
                full = next(a for a in release["assets"] if a["name"].endswith(".zip"))
                delta_asset = next(a for a in release["assets"] if a["name"].endswith(".gkhdelta"))
                delta_asset["size"] = full["size"]
                document["file_update"]["delta"]["bytes"] = full["size"]
            else:
                delta_asset = next(a for a in release["assets"] if a["name"].endswith(".gkhdelta"))
                bad = case / "bad.gkhdelta"
                bad.write_bytes(b"damaged package")
                case_assets[delta_asset["url"].rsplit("/", 1)[-1]] = str(bad)
            metadata_path = case / "target-metadata.json"
            metadata_path.write_text(json.dumps(document))
            case_assets[identity] = str(metadata_path)
            metadata_asset["digest"] = "sha256:" + file_update.digest(metadata_path)
            metadata_asset["size"] = metadata_path.stat().st_size
        config = {"root": str(root), "work": str(case / "operation"), "from": base_version, "to": "v0.5.4",
                  "catalog": catalog, "assets": case_assets, "mode": mode,
                  "locked": "obsolete.py" if name == "locked_delete" else "code.py"}
        if name == "corrupt_full":
            full = next(a for a in catalog[-1]["assets"] if a["name"].endswith(".zip"))
            bad = case / "bad-full.zip"
            bad.write_bytes(b"damaged full package")
            case_assets[full["url"].rsplit("/", 1)[-1]] = str(bad)
        if name == "budget_exhausted":
            deltas = [a for r in catalog for a in r["assets"] if a["name"].endswith(".gkhdelta")]
            config["failures"] = {a["url"].rsplit("/", 1)[-1]: (3 if i == 2 else 1) for i, a in enumerate(deltas)}
        if name == "rate_limited":
            config["rate_id"] = next(a["url"].rsplit("/", 1)[-1] for a in catalog[-1]["assets"] if a["name"].endswith(".json"))
        config_path = case / "input.json"
        config_path.write_text(json.dumps(config))
        result = subprocess.run([str(dotnet), str(output), str(config_path)], capture_output=True, text=True,
                                encoding="utf-8", env=env, timeout=60)
        (case / "run.log").write_text(result.stdout + result.stderr)
        if name in {"corrupt_full", "budget_exhausted", "rate_limited"}:
            assert result.returncode != 0, "Invalid update must fail"
            expected_root = packages.get(base_version, packages["v0.5.1"])
            for target in expected_root.rglob("*"):
                if target.is_file():
                    assert (root / target.relative_to(expected_root)).read_bytes() == target.read_bytes(), "Failure before shutdown must not change installed files"
            requests = (case / "operation/requests.log").read_text().splitlines()
            if name == "budget_exhausted":
                assert len(requests) == 8, requests  # Three metadata + three initial delta attempts + two recoveries.
            if name == "rate_limited":
                assert len(requests) == 1, requests
            results.append({"case": name, "failed_without_payload_changes": True, "requests": requests})
            continue
        result.check_returncode()
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        assert summary["chosen_delta"] == delta and summary["count"] == count, (name, summary)
        # The final shipped payload, including updater files and version metadata, must be byte-identical.
        for target in packages["v0.5.4"].rglob("*"):
            if target.is_file():
                assert (root / target.relative_to(packages["v0.5.4"])).read_bytes() == target.read_bytes(), (name, target.name)
        assert not (root / "obsolete.py").exists()
        assert (root / "config/settings.json").read_text() == "user data"
        assert (root / ".local/runtime-data/cache.json").read_text() == "user data"
        assert (root / "logs/old.log").read_text() == "user data"
        assert (root / "agent/__pycache__/cache.pyc").read_text() == "user data"
        assert len(summary["requests"]) == len(set(summary["requests"])), "No repeated metadata/package requests"
        results.append({"case": name, **summary})
    evidence = {"status": "PASS", "cases": results, "network": "none; actual production planner and transaction with fake HTTP",
                "patch_sha256": file_update.digest(patch), "installation_changed": False}
    (work / "result.json").write_text(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--dotnet", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
