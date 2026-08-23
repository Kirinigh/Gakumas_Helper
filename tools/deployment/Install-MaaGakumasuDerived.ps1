[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CandidatePath,

    [string]$InstallRoot = 'C:\Tools\MaaGakumasu',

    [switch]$SkipDefenderScan
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Assert-Manifest {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)]$Manifest
    )

    foreach ($property in $Manifest.critical_files.PSObject.Properties) {
        $relative = [string]$property.Name
        $expected = ([string]$property.Value).ToUpperInvariant()
        $candidate = [IO.Path]::GetFullPath((Join-Path $Root $relative))
        $resolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
        if (-not $candidate.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Build manifest path escapes candidate root: $relative"
        }
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Build manifest file is missing: $relative"
        }
        $actual = Get-Sha256 -Path $candidate
        if ($actual -ne $expected) {
            throw "Build manifest hash mismatch: $relative; actual $actual; expected $expected"
        }
    }
}

function Set-PipBootstrapVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Version
    )

    $configDir = Join-Path $Root 'config'
    $configPath = Join-Path $configDir 'pip_config.json'
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null

    if (Test-Path -LiteralPath $configPath -PathType Leaf) {
        try {
            $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            throw "Existing pip configuration is invalid and was not replaced: $configPath"
        }
    }
    else {
        $config = [pscustomobject][ordered]@{
            enable_pip_update = $true
            enable_pip_install = $true
            last_version = $Version
            mirror = 'https://mirrors.ustc.edu.cn/pypi/simple'
            backup_mirrors = @(
                'https://pypi.tuna.tsinghua.edu.cn/simple',
                'https://mirrors.cloud.tencent.com/pypi/simple/',
                'https://pypi.org/simple'
            )
        }
    }

    if ($null -eq $config -or $config -is [array] -or $config -isnot [pscustomobject]) {
        throw "Existing pip configuration must be a JSON object and was not replaced: $configPath"
    }

    if ($config.PSObject.Properties.Name -contains 'last_version') {
        $config.last_version = $Version
    }
    else {
        $config | Add-Member -NotePropertyName 'last_version' -NotePropertyValue $Version
    }

    $temporaryPath = "$configPath.$([guid]::NewGuid().ToString('N')).tmp"
    try {
        $json = $config | ConvertTo-Json -Depth 16
        [IO.File]::WriteAllText($temporaryPath, $json + [Environment]::NewLine, (New-Object System.Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temporaryPath -Destination $configPath -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

function Invoke-ArenaPeriodConfigMigration {
    param([Parameter(Mandatory = $true)][string]$Root)

    $instancesPath = Join-Path $Root 'config\instances'
    if (-not (Test-Path -LiteralPath $instancesPath -PathType Container)) {
        return '{"schema_version":1,"instances":[]}'
    }
    $pythonPath = Join-Path $Root 'python\python.exe'
    $migrationPath = Join-Path $Root 'agent\arena_winrate\config_migration.py'
    $interfacePath = Join-Path $Root 'interface.json'
    foreach ($requiredPath in @($pythonPath, $migrationPath, $interfacePath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Arena period configuration migration dependency is missing: $requiredPath"
        }
    }

    $output = & $pythonPath -B $migrationPath --instances-dir $instancesPath --interface $interfacePath 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Arena period configuration migration failed: $($output -join [Environment]::NewLine)"
    }
    return ($output -join [Environment]::NewLine)
}

function Initialize-ArenaOwnScoreSummary {
    param([Parameter(Mandatory = $true)][string]$Root)

    $pythonPath = Join-Path $Root 'python\python.exe'
    $code = "import sys; from pathlib import Path; root = Path(sys.argv[1]); sys.path.insert(0, str(root / 'agent')); from arena_winrate import DEFAULT_OWN_SCORE_CACHE, OwnScoreCacheStore; OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE).ensure_summary()"
    $output = & $pythonPath -B -c $code $Root 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Arena own-score task summary initialization failed: $($output -join [Environment]::NewLine)"
    }
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'The candidate installer must run in an elevated PowerShell process.'
}

$candidateRoot = (Resolve-Path -LiteralPath $CandidatePath).Path
$installRootFull = [IO.Path]::GetFullPath($InstallRoot)
$versionsRoot = Join-Path $installRootFull 'versions'
$currentPath = Join-Path $installRootFull 'current'
$manifestPath = Join-Path $candidateRoot 'GAKUMAS_HELPER_BUILD.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Candidate is missing GAKUMAS_HELPER_BUILD.json: $candidateRoot"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.schema_version -ne 1 -or $manifest.product -ne 'MaaGakumasu') {
    throw 'Candidate build manifest identity is invalid.'
}
$version = [string]$manifest.derived_version
if ($version -notmatch '^v\d+\.\d+\.\d+(?:-(?:alpha|beta)\.\d+|\+gkh\.(?:[0-9a-f]{7,40}|\d{6}))?$') {
    throw "Candidate version is invalid: $version"
}

if (Get-Process -Name 'MaaGakumasu', 'MFAAvalonia' -ErrorAction SilentlyContinue) {
    throw 'MaaGakumasu/MFAAvalonia is running; the install root was not changed.'
}

Assert-Manifest -Root $candidateRoot -Manifest $manifest

if (-not $SkipDefenderScan) {
    if (-not (Get-Command 'Start-MpScan' -ErrorAction SilentlyContinue)) {
        throw 'Windows Defender Start-MpScan is unavailable; the candidate was not installed.'
    }
    Start-MpScan -ScanType CustomScan -ScanPath $candidateRoot
}

New-Item -ItemType Directory -Path $versionsRoot -Force | Out-Null
$targetPath = Join-Path $versionsRoot $version
if (Test-Path -LiteralPath $targetPath) {
    throw "Target version directory already exists and was not overwritten: $targetPath"
}

$stagingPath = Join-Path $versionsRoot ('.staging-' + $version + '-' + [guid]::NewGuid().ToString('N'))
$previousTarget = $null
$previousLink = $null
$newLink = Join-Path $installRootFull ('.current-new-' + [guid]::NewGuid().ToString('N'))
$currentMoved = $false
$arenaPeriodMigration = $null

try {
    New-Item -ItemType Directory -Path $stagingPath | Out-Null
    Copy-Item -Path (Join-Path $candidateRoot '*') -Destination $stagingPath -Recurse -Force

    if (Test-Path -LiteralPath $currentPath) {
        $currentItem = Get-Item -LiteralPath $currentPath -Force
        if ($currentItem.LinkType -ne 'Junction' -or -not $currentItem.Target) {
            throw "current is not a verifiable junction: $currentPath"
        }
        $previousTarget = [string]$currentItem.Target

        foreach ($mutablePath in @('config', '.local')) {
            $source = Join-Path $currentPath $mutablePath
            $destination = Join-Path $stagingPath $mutablePath
            if (Test-Path -LiteralPath $source) {
                if (Test-Path -LiteralPath $destination) {
                    Remove-Item -LiteralPath $destination -Recurse -Force
                }
                Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
            }
        }
        $appSettings = Join-Path $currentPath 'appsettings.json'
        if (Test-Path -LiteralPath $appSettings -PathType Leaf) {
            Copy-Item -LiteralPath $appSettings -Destination (Join-Path $stagingPath 'appsettings.json') -Force
        }
    }

    # The derived package already contains the validated Python dependency set.
    # Preserve the user's pip update/install switches, but mark this exact resource
    # version as bootstrapped so Agent startup does not replace the embedded set.
    Set-PipBootstrapVersion -Root $stagingPath -Version $version
    $arenaPeriodMigration = Invoke-ArenaPeriodConfigMigration -Root $stagingPath
    Initialize-ArenaOwnScoreSummary -Root $stagingPath

    Assert-Manifest -Root $stagingPath -Manifest $manifest
    Move-Item -LiteralPath $stagingPath -Destination $targetPath

    New-Item -ItemType Junction -Path $newLink -Target $targetPath | Out-Null
    if (Test-Path -LiteralPath $currentPath) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $previousLink = Join-Path $installRootFull ("current.previous-$stamp")
        if (Test-Path -LiteralPath $previousLink) {
            throw "Rollback junction name already exists: $previousLink"
        }
        Move-Item -LiteralPath $currentPath -Destination $previousLink
        $currentMoved = $true
    }
    Move-Item -LiteralPath $newLink -Destination $currentPath

    Set-Content -LiteralPath (Join-Path $installRootFull 'CURRENT_VERSION.txt') -Value $version -Encoding ASCII
    $record = @"
# MaaGakumasu local derived deployment record

- Derived version: $version
- Upstream version: $($manifest.upstream.tag)
- Upstream release SHA-256: $($manifest.upstream.sha256)
- Local source revision: $($manifest.source.revision)
- Arena engine revision: $($manifest.arena_engine.commit)
- Update mode: $($manifest.update_contract.mode)
- Update repository: $($manifest.update_contract.repository)
- Install path: $targetPath
- Current entry: $currentPath
- Previous target: $previousTarget
- Rollback junction: $previousLink
- User configuration: preserved from the previous current; EnableAutoUpdateResource was not changed
- Arena period configuration: legacy free-text values migrated to the fixed selector before activation
- Arena own-score summary: initialized from the preserved authoritative JSON cache before activation
- Python bootstrap: last_version synchronized to the embedded candidate; existing pip update/install switches preserved
- Windows Defender: $(if ($SkipDefenderScan) { 'explicitly skipped for this run' } else { 'candidate directory scanned' })
"@
    Set-Content -LiteralPath (Join-Path $installRootFull 'DEPLOYMENT.md') -Value $record -Encoding UTF8

    [pscustomobject]@{
        Version = $version
        TargetPath = $targetPath
        CurrentPath = $currentPath
        PreviousTarget = $previousTarget
        RollbackJunction = $previousLink
        AutoUpdateResourcePreserved = $true
        PipBootstrapVersion = $version
        DefenderScanned = -not $SkipDefenderScan
        ArenaPeriodMigration = (ConvertFrom-Json -InputObject $arenaPeriodMigration)
    } | ConvertTo-Json -Depth 4
}
catch {
    if (-not (Test-Path -LiteralPath $currentPath) -and $currentMoved -and $previousLink -and (Test-Path -LiteralPath $previousLink)) {
        Move-Item -LiteralPath $previousLink -Destination $currentPath
    }
    if (Test-Path -LiteralPath $newLink) {
        Remove-Item -LiteralPath $newLink -Force
    }
    throw
}
