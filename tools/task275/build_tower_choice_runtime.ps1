param(
    [string] $HiroSource,
    [string] $ToolchainRoot,
    [string] $Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$base = Join-Path $repo '.local/task275-rust'
$revision = 'a7df5a9be782765d3c6282566157fbd3f80b99ee'
if (-not $HiroSource) { $HiroSource = Join-Path $base "hiro-gakumas-hero-$revision" }
if (-not $ToolchainRoot) { $ToolchainRoot = $base }
if (-not $Python) { $Python = Join-Path $repo '.venv/Scripts/python.exe' }
$sourceRevision = git -C $HiroSource rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $sourceRevision -ne $revision) {
    throw "Use the pinned Hiro source checkout $revision."
}
$source = Join-Path $HiroSource 'vendor/gakumas-player/gakumas_exam_rs'
$relative = ".local/task275-rust/choice-build-$($revision.Substring(0, 8))"
$build = Join-Path $repo $relative
$patches = 'hiro-tower-choice.patch', 'hiro-tower-live-state.patch', 'hiro-tower-stamina-search.patch',
    'hiro-tower-previous-card.patch'

if (-not (Test-Path -LiteralPath (Join-Path $build 'Cargo.toml'))) {
    Copy-Item -LiteralPath $source -Destination $build -Recurse
}

Push-Location -LiteralPath $repo
try {
    foreach ($name in $patches) {
        $patch = Join-Path $PSScriptRoot $name
        git apply --reverse --check --directory=$relative -- $patch 2>$null
        if ($LASTEXITCODE -ne 0) {
            git apply --check --directory=$relative -- $patch
            if ($LASTEXITCODE -ne 0) { throw "The pinned source does not match $name." }
            git apply --directory=$relative -- $patch
            if ($LASTEXITCODE -ne 0) { throw "$name did not apply." }
        }
    }
} finally {
    Pop-Location
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'The MSVC x64 build tools are required.' }
$setup = Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$toolchain = Join-Path $ToolchainRoot 'toolchain/bin'
$cargo = Join-Path $toolchain 'cargo.exe'
$env:CARGO_HOME = Join-Path $ToolchainRoot 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $build 'target'
$env:RUSTC = Join-Path $toolchain 'rustc.exe'
$env:PYO3_PYTHON = (Resolve-Path -LiteralPath $Python).Path
$env:PATH = "$toolchain;$env:PATH"
$manifest = Join-Path $build 'Cargo.toml'
$command = 'call "{0}" && "{1}" build --manifest-path "{2}" --release --offline --locked' -f $setup,$cargo,$manifest
& $env:ComSpec /d /c $command
if ($LASTEXITCODE -ne 0) { throw 'The offline Rust build failed.' }

$runtimeSource = Join-Path $base "runtime/$revision/src"
if (-not (Test-Path -LiteralPath $runtimeSource -PathType Container)) {
    throw "The pinned Tower runtime directory is missing: $runtimeSource"
}
$artifact = Join-Path $runtimeSource 'gakumas_exam_rs.pyd'
Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR 'release/gakumas_exam_rs.dll') -Destination $artifact
$hiroPackage = Join-Path $HiroSource 'src/hiro'
Copy-Item -LiteralPath $hiroPackage -Destination $runtimeSource -Recurse -Force
$font = Join-Path $HiroSource 'assets/pack/font/Campus-Alphanumeric-SemiBold.ttf'
if (Test-Path -LiteralPath $font -PathType Leaf) {
    $fonts = Join-Path (Split-Path -Parent $runtimeSource) 'assets/pack/font'
    New-Item -ItemType Directory -Path $fonts -Force | Out-Null
    Copy-Item -LiteralPath $font -Destination $fonts -Force
}
Get-FileHash -LiteralPath $artifact -Algorithm SHA256
