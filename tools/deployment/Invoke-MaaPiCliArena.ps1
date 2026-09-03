[CmdletBinding(DefaultParameterSetName = 'Launch')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Launch')]
    [Parameter(Mandatory = $true, ParameterSetName = 'Prepare')]
    [string]$SourceVersionRoot,

    [Parameter(Mandatory = $true, ParameterSetName = 'Launch')]
    [Parameter(Mandatory = $true, ParameterSetName = 'Prepare')]
    [Parameter(Mandatory = $true, ParameterSetName = 'Worker')]
    [string]$RuntimeRoot,

    [Parameter(ParameterSetName = 'Launch')]
    [Parameter(ParameterSetName = 'Worker')]
    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds = 3600,

    [Parameter(ParameterSetName = 'Launch')]
    [Parameter(ParameterSetName = 'Prepare')]
    [ValidateRange(0, 100)]
    [int]$ThresholdPercent = 70,

    [Parameter(ParameterSetName = 'Launch')]
    [Parameter(ParameterSetName = 'Prepare')]
    [ValidateRange(1000, 2147483647)]
    [int]$Simulations = 2000,

    [Parameter(ParameterSetName = 'Launch')]
    [Parameter(ParameterSetName = 'Prepare')]
    [ValidateRange(1, 2147483647)]
    [int]$SimulationTimeoutSeconds = 180,

    [Parameter(ParameterSetName = 'Launch')]
    [Parameter(ParameterSetName = 'Prepare')]
    [string]$ReuseOwnScoreCachePath,

    [Parameter(Mandatory = $true, ParameterSetName = 'Prepare')]
    [switch]$PrepareOnly,

    [Parameter(Mandatory = $true, ParameterSetName = 'Worker')]
    [switch]$ElevatedWorker
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$script:ArenaTaskName = 'arena_win_rate_manual_challenge'
$script:NativeRelativePath = 'runtimes\win-x64\native'
$script:MutableRootNames = @('.local', 'backup', 'config', 'debug', 'logs', 'temp')
$script:OwnScoreCacheRelativePath = '.local\arena-win-rate\own-score-cache-v1.json'
$script:OwnScoreSeed = 400
$script:ArenaOptionNames = @'
{
  "season": "\u7ade\u6280\u573a\u671f\u6570",
  "win_rate": "\u7ade\u6280\u573a\u80dc\u7387\u53c2\u6570",
  "own_recalculation": "\u6bcf\u65e5\u6311\u6218\u81ea\u52a8\u91cd\u7b97\u5df1\u65b9\u6570\u636e",
  "capture": "\u6bcf\u65e5\u6311\u6218\u672c\u5730\u753b\u9762\u91c7\u96c6",
  "capture_off": "\u5173\u95ed"
}
'@ | ConvertFrom-Json

function Get-GkhFullPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path)
}

function Test-GkhPathIsEqualOrChild([string]$Candidate, [string]$Parent) {
    $candidateFull = (Get-GkhFullPath $Candidate).TrimEnd('\')
    $parentFull = (Get-GkhFullPath $Parent).TrimEnd('\')
    return (
        $candidateFull.Equals($parentFull, [StringComparison]::OrdinalIgnoreCase) -or
        $candidateFull.StartsWith($parentFull + '\', [StringComparison]::OrdinalIgnoreCase)
    )
}

function Assert-GkhNotReparsePoint([string]$Path, [string]$Label) {
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Label must not be a reparse point: $Path"
    }
}

function Assert-GkhExistingDirectoryAncestorsNotReparse([string]$Path, [string]$Label) {
    $cursor = Get-GkhFullPath $Path
    while ($true) {
        if (Test-Path -LiteralPath $cursor -PathType Container) {
            Assert-GkhNotReparsePoint $cursor $Label
        }
        $parent = [IO.Directory]::GetParent($cursor)
        if ($null -eq $parent) {
            break
        }
        $cursor = $parent.FullName
    }
}

function Write-GkhUtf8Json([string]$Path, [object]$Value, [int]$Depth = 8) {
    $json = $Value | ConvertTo-Json -Depth $Depth
    [IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}

function Get-GkhOptionalProperty([object]$Value, [string]$Name) {
    if ($null -eq $Value) {
        return $null
    }
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    # Preserve an explicitly present empty JSON array as one pipeline object.
    # Without the unary comma Windows PowerShell emits no value, and the
    # summary serializer turns the field into ``{}`` instead of ``[]``.
    return ,$property.Value
}

function Write-GkhArenaRunSummary([string]$PreparedRuntime, [int]$ProcessExitCode) {
    $arenaRoot = Join-Path $PreparedRuntime '.local\arena-win-rate'
    $pendingPath = Join-Path $arenaRoot 'pending-challenge-v1.json'
    $resultsRoot = Join-Path $arenaRoot 'challenge-results-v1'
    $summaryPath = Join-Path $PreparedRuntime 'maapicli-arena-summary.json'
    $errors = [Collections.Generic.List[string]]::new()
    $pendingEvidence = $null
    $pendingExists = Test-Path -LiteralPath $pendingPath -PathType Leaf
    if ($pendingExists) {
        try {
            $pending = Get-Content -LiteralPath $pendingPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $pendingEvidence = [ordered]@{
                capture_id = Get-GkhOptionalProperty $pending 'capture_id'
                lifecycle_state = Get-GkhOptionalProperty $pending 'lifecycle_state'
                return_verification = Get-GkhOptionalProperty $pending 'return_verification'
            }
        } catch {
            [void]$errors.Add('pending challenge record is unreadable')
        }
    }

    $records = [Collections.Generic.List[object]]::new()
    $allTerminal = $true
    $resultFiles = if (Test-Path -LiteralPath $resultsRoot -PathType Container) {
        @(Get-ChildItem -LiteralPath $resultsRoot -Filter '*.json' -File)
    } else {
        @()
    }
    foreach ($resultFile in $resultFiles) {
        try {
            $record = Get-Content -LiteralPath $resultFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            $intent = Get-GkhOptionalProperty $record 'intent'
            $result = Get-GkhOptionalProperty $record 'result'
            $lifecycle = Get-GkhOptionalProperty $record 'lifecycle'
            $lifecycleState = Get-GkhOptionalProperty $lifecycle 'state'
            $terminal = $lifecycleState -cin @('FINISHED', 'RECOVERED_FINISHED')
            if (-not $terminal) {
                $allTerminal = $false
            }
            [void]$records.Add([ordered]@{
                capture_id = Get-GkhOptionalProperty $intent 'capture_id'
                selected_position = Get-GkhOptionalProperty $intent 'selected_position'
                estimates = Get-GkhOptionalProperty $intent 'estimates'
                provider_attempts = Get-GkhOptionalProperty $intent 'provider_attempts'
                opponent_read_summary = Get-GkhOptionalProperty $intent 'opponent_read_summary'
                cost_customization_fallbacks = Get-GkhOptionalProperty $intent 'cost_customization_fallbacks'
                outcome = Get-GkhOptionalProperty $result 'outcome'
                stage_results = Get-GkhOptionalProperty $result 'stage_results'
                score_repairs = Get-GkhOptionalProperty $result 'score_repairs'
                lifecycle_state = $lifecycleState
                return_verification = Get-GkhOptionalProperty $record 'return_verification'
                result_path = $resultFile.FullName
                screenshots_persisted = $false
            })
        } catch {
            $allTerminal = $false
            [void]$errors.Add("challenge result record is unreadable: $($resultFile.Name)")
        }
    }

    if ($records.Count -gt 1) {
        [void]$errors.Add('isolated arena runtime contains more than one challenge result')
    }
    $challengeAttempted = $pendingExists -or $records.Count -gt 0
    $strictProductClosed = (
        -not $pendingExists -and
        $errors.Count -eq 0 -and
        $records.Count -le 1 -and
        $allTerminal
    )
    $status = if (-not $challengeAttempted -and $strictProductClosed) {
        'NO_CHALLENGE_RECORDED'
    } elseif ($strictProductClosed) {
        'FINISHED'
    } else {
        'UNFINISHED'
    }
    $summary = [ordered]@{
        schema_version = 1
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        process_exit_code = $ProcessExitCode
        status = $status
        challenge_attempted = $challengeAttempted
        strict_product_closed = $strictProductClosed
        pending = $pendingEvidence
        challenge_records = @($records)
        errors = @($errors)
        screenshots_persisted = $false
    }
    Write-GkhUtf8Json $summaryPath $summary 16
    return [pscustomobject]$summary
}

function Get-GkhSha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha256 = [Security.Cryptography.SHA256]::Create()
        try {
            return [BitConverter]::ToString($sha256.ComputeHash($stream)).Replace('-', '')
        } finally {
            $sha256.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Test-GkhInteger([object]$Value) {
    return (
        $Value -is [sbyte] -or
        $Value -is [byte] -or
        $Value -is [int16] -or
        $Value -is [uint16] -or
        $Value -is [int32] -or
        $Value -is [uint32] -or
        $Value -is [int64] -or
        $Value -is [uint64]
    )
}

function Test-GkhIntegerSequenceEqual([object[]]$Left, [object[]]$Right) {
    if ($Left.Count -ne $Right.Count) {
        return $false
    }
    for ($index = 0; $index -lt $Left.Count; $index++) {
        if (
            -not (Test-GkhInteger $Left[$index]) -or
            -not (Test-GkhInteger $Right[$index]) -or
            [int64]$Left[$index] -ne [int64]$Right[$index]
        ) {
            return $false
        }
    }
    return $true
}

function Get-GkhOwnScoreCacheEvidence([string]$Path, [int]$RequestedSimulations) {
    $cachePath = Get-GkhFullPath $Path
    if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf)) {
        throw "Explicit own-score cache not found: $cachePath"
    }
    Assert-GkhNotReparsePoint $cachePath 'Explicit own-score cache'
    Assert-GkhExistingDirectoryAncestorsNotReparse (Split-Path -Parent $cachePath) 'Own-score cache ancestor'

    try {
        $cache = Get-Content -LiteralPath $cachePath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw "Explicit own-score cache is not valid UTF-8 JSON: $cachePath"
    }
    if ($null -eq $cache -or $cache -is [array]) {
        throw 'Explicit own-score cache root must be one JSON object.'
    }
    $stageIds = @($cache.stageIds)
    $snapshot = $cache.own_snapshot
    $snapshotStageIds = if ($null -eq $snapshot) { @() } else { @($snapshot.stageIds) }
    $cacheSchemaVersion = if (Test-GkhInteger $cache.schema_version) {
        [int64]$cache.schema_version
    } else {
        0
    }
    if (
        $cacheSchemaVersion -notin @(1, 2) -or
        $cache.upstream_commit -isnot [string] -or
        [string]$cache.upstream_commit -cnotmatch '^[0-9a-f]{40}$' -or
        -not (Test-GkhInteger $cache.season) -or
        [int64]$cache.season -lt 1 -or
        $stageIds.Count -ne 3 -or
        @($stageIds | Where-Object { -not (Test-GkhInteger $_) -or [int64]$_ -lt 1 }).Count -ne 0 -or
        @($stageIds | ForEach-Object { [int64]$_ } | Sort-Object -Unique).Count -ne 3 -or
        -not (Test-GkhInteger $cache.simulations) -or
        [int64]$cache.simulations -lt $RequestedSimulations -or
        -not (Test-GkhInteger $cache.seed) -or
        [int64]$cache.seed -ne $script:OwnScoreSeed -or
        $null -eq $snapshot -or
        $snapshot.capture_id -isnot [string] -or
        [string]::IsNullOrWhiteSpace([string]$snapshot.capture_id) -or
        $snapshot.source -cne 'live_screen' -or
        -not (Test-GkhInteger $snapshot.season) -or
        [int64]$snapshot.season -ne [int64]$cache.season -or
        -not (Test-GkhIntegerSequenceEqual $stageIds $snapshotStageIds)
    ) {
        throw 'Explicit own-score cache metadata is incompatible with the isolated arena run.'
    }

    return [pscustomobject][ordered]@{
        sha256 = Get-GkhSha256 $cachePath
        capture_id = [string]$snapshot.capture_id
        season = [int64]$cache.season
        stage_ids = @($stageIds | ForEach-Object { [int64]$_ })
        simulations = [int64]$cache.simulations
        seed = [int64]$cache.seed
        upstream_commit = [string]$cache.upstream_commit
    }
}

function Copy-GkhOwnScoreCache(
    [string]$SourcePath,
    [string]$Destination,
    [int]$RequestedSimulations
) {
    $source = Get-GkhFullPath $SourcePath
    if (Test-GkhPathIsEqualOrChild $source $Destination) {
        throw "Explicit own-score cache must be outside RuntimeRoot: $source"
    }
    $evidence = Get-GkhOwnScoreCacheEvidence $source $RequestedSimulations
    $target = Join-Path $Destination $script:OwnScoreCacheRelativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
    Assert-GkhNotReparsePoint $target 'Copied own-score cache'
    $targetEvidence = Get-GkhOwnScoreCacheEvidence $target $RequestedSimulations
    if (
        $targetEvidence.sha256 -cne $evidence.sha256 -or
        $targetEvidence.capture_id -cne $evidence.capture_id
    ) {
        throw 'Copied own-score cache differs from the explicitly selected source.'
    }
    return [pscustomobject][ordered]@{
        destination = $script:OwnScoreCacheRelativePath
        sha256 = $targetEvidence.sha256
        capture_id = $targetEvidence.capture_id
        season = $targetEvidence.season
        stage_ids = @($targetEvidence.stage_ids)
        simulations = $targetEvidence.simulations
        seed = $targetEvidence.seed
        upstream_commit = $targetEvidence.upstream_commit
    }
}

function Assert-GkhOwnScoreCacheManifest(
    [string]$Runtime,
    [object]$Metadata,
    [int]$RequestedSimulations
) {
    $propertyNames = @($Metadata.PSObject.Properties | ForEach-Object { $_.Name })
    $requiredPropertyNames = @(
        'destination',
        'sha256',
        'capture_id',
        'season',
        'stage_ids',
        'simulations',
        'seed',
        'upstream_commit'
    )
    if (
        $propertyNames.Count -ne $requiredPropertyNames.Count -or
        @($requiredPropertyNames | Where-Object { $propertyNames -cnotcontains $_ }).Count -ne 0 -or
        $Metadata.destination -cne $script:OwnScoreCacheRelativePath
    ) {
        throw 'Prepared explicit own-score cache manifest changed or is incomplete.'
    }

    $target = Join-Path $Runtime $script:OwnScoreCacheRelativePath
    $evidence = Get-GkhOwnScoreCacheEvidence $target $RequestedSimulations
    if (
        $Metadata.sha256 -cne $evidence.sha256 -or
        $Metadata.capture_id -cne $evidence.capture_id -or
        -not (Test-GkhInteger $Metadata.season) -or
        [int64]$Metadata.season -ne $evidence.season -or
        -not (Test-GkhIntegerSequenceEqual @($Metadata.stage_ids) @($evidence.stage_ids)) -or
        -not (Test-GkhInteger $Metadata.simulations) -or
        [int64]$Metadata.simulations -ne $evidence.simulations -or
        -not (Test-GkhInteger $Metadata.seed) -or
        [int64]$Metadata.seed -ne $evidence.seed -or
        $Metadata.upstream_commit -cne $evidence.upstream_commit
    ) {
        throw 'Prepared explicit own-score cache changed after isolation.'
    }
}

function Assert-GkhSourceVersionRoot([string]$Path) {
    $source = Get-GkhFullPath $Path
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Installed version root not found: $source"
    }
    Assert-GkhExistingDirectoryAncestorsNotReparse $source 'Installed version ancestor'

    $sourceItem = Get-Item -LiteralPath $source -Force
    if ($null -eq $sourceItem.Parent -or $sourceItem.Parent.Name -cne 'versions') {
        throw "SourceVersionRoot must be a direct child of an installed versions directory: $source"
    }

    $nativeRoot = Join-Path $source $script:NativeRelativePath
    foreach ($requiredPath in @(
        (Join-Path $source 'interface.json'),
        (Join-Path $source 'agent\main.py'),
        (Join-Path $source 'python\python.exe'),
        (Join-Path $source 'resource'),
        (Join-Path $source 'tasks'),
        (Join-Path $nativeRoot 'MaaPiCli.exe'),
        (Join-Path $nativeRoot 'MaaFramework.dll'),
        (Join-Path $nativeRoot 'MaaWin32ControlUnit.dll')
    )) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Installed version is incomplete for MaaPiCli: $requiredPath"
        }
    }
    Assert-GkhNotReparsePoint $nativeRoot 'Native runtime root'
    return $source
}

function Assert-GkhRuntimeDestination([string]$Source, [string]$Destination) {
    $runtime = Get-GkhFullPath $Destination
    $sourceItem = Get-Item -LiteralPath $Source -Force
    $versionsRoot = $sourceItem.Parent.FullName
    $installRoot = $sourceItem.Parent.Parent.FullName
    $currentRoot = Join-Path $installRoot 'current'

    if (
        (Test-GkhPathIsEqualOrChild $runtime $Source) -or
        (Test-GkhPathIsEqualOrChild $runtime $versionsRoot) -or
        (Test-GkhPathIsEqualOrChild $runtime $currentRoot)
    ) {
        throw "RuntimeRoot must not modify current or versions: $runtime"
    }
    if (Test-Path -LiteralPath $runtime) {
        throw "RuntimeRoot already exists; refusing to overwrite it: $runtime"
    }

    Assert-GkhExistingDirectoryAncestorsNotReparse (Split-Path -Parent $runtime) 'RuntimeRoot ancestor'
    return $runtime
}

function Copy-GkhStaticVersion([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
    Assert-GkhNotReparsePoint $Destination 'Isolated runtime root'

    foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
        if ($item.Name -in $script:MutableRootNames) {
            continue
        }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Installed static entry must not be a reparse point: $($item.FullName)"
        }
        if ($item.PSIsContainer) {
            $nestedReparsePoint = Get-ChildItem -LiteralPath $item.FullName -Recurse -Force |
                Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 } |
                Select-Object -First 1
            if ($null -ne $nestedReparsePoint) {
                throw "Installed static tree contains a reparse point: $($nestedReparsePoint.FullName)"
            }
        }
        Copy-Item -LiteralPath $item.FullName -Destination $Destination -Recurse
    }
}

function Copy-GkhNativeFiles([string]$Source, [string]$Destination) {
    $nativeRoot = Join-Path $Source $script:NativeRelativePath
    foreach ($nativeFile in Get-ChildItem -LiteralPath $nativeRoot -File -Force) {
        if (($nativeFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Native MaaPiCli dependency must not be a reparse point: $($nativeFile.FullName)"
        }
        $target = Join-Path $Destination $nativeFile.Name
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            if ((Get-GkhSha256 $nativeFile.FullName) -ne (Get-GkhSha256 $target)) {
                throw "Native MaaPiCli dependency collides with a different root file: $target"
            }
            continue
        }
        Copy-Item -LiteralPath $nativeFile.FullName -Destination $target
        if ((Get-GkhSha256 $nativeFile.FullName) -ne (Get-GkhSha256 $target)) {
            throw "Flattened native MaaPiCli dependency copy differs from its source: $target"
        }
    }
}

function Write-GkhArenaConfiguration(
    [string]$Destination,
    [string]$Source,
    [int]$WinRateThresholdPercent,
    [int]$WinRateSimulations,
    [int]$WinRateTimeoutSeconds,
    [bool]$AutoRecalculateOwn,
    [object]$OwnCacheMetadata
) {
    $configRoot = Join-Path $Destination 'config'
    New-Item -ItemType Directory -Path $configRoot | Out-Null
    $ownRecalculationValue = if ($AutoRecalculateOwn) { 'Yes' } else { 'No' }

    Write-GkhUtf8Json (Join-Path $configRoot 'maa_pi_config.json') ([ordered]@{
        controller = [ordered]@{ name = 'PC' }
        win32 = [ordered]@{ _placeholder = 0 }
        resource = 'DMM'
        task = @(
            [ordered]@{
                name = $script:ArenaTaskName
                option = @(
                    [ordered]@{
                        name = $script:ArenaOptionNames.season
                        value = 'latest'
                    },
                    [ordered]@{
                        name = $script:ArenaOptionNames.win_rate
                        inputs = [ordered]@{
                            arena_win_rate_threshold_percent = [string]$WinRateThresholdPercent
                            arena_win_rate_simulations = [string]$WinRateSimulations
                            arena_win_rate_timeout_seconds = [string]$WinRateTimeoutSeconds
                        }
                    },
                    [ordered]@{
                        name = $script:ArenaOptionNames.own_recalculation
                        value = $ownRecalculationValue
                    },
                    [ordered]@{
                        name = $script:ArenaOptionNames.capture
                        value = $script:ArenaOptionNames.capture_off
                    }
                )
            }
        )
    })
    Write-GkhUtf8Json (Join-Path $configRoot 'config.json') ([ordered]@{
        CurrentLanguage = 'zh-CN'
        ResourceUpdateChannelInitialized = $true
        EnableAutoUpdateResource = $false
    })
    Write-GkhUtf8Json (Join-Path $configRoot 'pip_config.json') ([ordered]@{
        enable_pip_update = $false
        enable_pip_install = $false
        last_version = 'isolated-maapicli-arena'
        mirror = ''
        backup_mirrors = @()
    })
    Write-GkhUtf8Json (Join-Path $configRoot 'maa_option.json') ([ordered]@{
        draw_quality = 85
        logging = $true
        save_draw = $false
        save_on_error = $false
        stdout_level = 2
    })
    $runManifest = [ordered]@{
        schema_version = 1
        task = $script:ArenaTaskName
        controller = 'PC'
        resource = 'DMM'
        source_version_root = $Source
        season = 'latest'
        threshold_percent = $WinRateThresholdPercent
        simulations = $WinRateSimulations
        simulation_timeout_seconds = $WinRateTimeoutSeconds
        own_cache_mode = if ($AutoRecalculateOwn) { 'recalculate' } else { 'explicit_reuse' }
        auto_recalculate_own = $AutoRecalculateOwn
        capture_mode = 'off'
        updates_enabled = $false
        prepared_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    if ($null -ne $OwnCacheMetadata) {
        $runManifest['own_cache'] = $OwnCacheMetadata
    }
    Write-GkhUtf8Json (Join-Path $Destination 'GKH_MAAPICLI_ARENA_RUN.json') $runManifest
}

function New-GkhIsolatedArenaRuntime(
    [string]$SourceVersion,
    [string]$RuntimeDestination,
    [int]$WinRateThresholdPercent,
    [int]$WinRateSimulations,
    [int]$WinRateTimeoutSeconds,
    [bool]$ReuseOwnScoreCache,
    [string]$OwnScoreCachePath
) {
    $source = Assert-GkhSourceVersionRoot $SourceVersion
    $runtime = Assert-GkhRuntimeDestination $source $RuntimeDestination
    try {
        Copy-GkhStaticVersion $source $runtime
        Copy-GkhNativeFiles $source $runtime
        $ownCacheMetadata = $null
        $autoRecalculateOwn = $true
        if ($ReuseOwnScoreCache) {
            $ownCacheMetadata = Copy-GkhOwnScoreCache `
                $OwnScoreCachePath `
                $runtime `
                $WinRateSimulations
            $autoRecalculateOwn = $false
        }
        Write-GkhArenaConfiguration `
            $runtime `
            $source `
            $WinRateThresholdPercent `
            $WinRateSimulations `
            $WinRateTimeoutSeconds `
            $autoRecalculateOwn `
            $ownCacheMetadata

        foreach ($requiredRuntimePath in @(
            (Join-Path $runtime 'MaaPiCli.exe'),
            (Join-Path $runtime 'MaaFramework.dll'),
            (Join-Path $runtime 'MaaWin32ControlUnit.dll'),
            (Join-Path $runtime 'config\maa_pi_config.json')
        )) {
            if (-not (Test-Path -LiteralPath $requiredRuntimePath -PathType Leaf)) {
                throw "Prepared MaaPiCli runtime is incomplete: $requiredRuntimePath"
            }
        }
        return [pscustomobject][ordered]@{
            RuntimeRoot = $runtime
            AutoRecalculateOwn = $autoRecalculateOwn
            OwnCacheMetadata = $ownCacheMetadata
        }
    } catch {
        if (Test-Path -LiteralPath $runtime -PathType Container) {
            Remove-Item -LiteralPath $runtime -Recurse -Force
        }
        throw
    }
}

function Add-GkhTokenProbeType {
    if ($null -ne ('GkhMaaPiCliTokenProbe' -as [type])) {
        return
    }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public sealed class GkhMaaPiCliTokenEvidence
{
    public bool Elevated { get; private set; }
    public int IntegrityRid { get; private set; }

    public GkhMaaPiCliTokenEvidence(bool elevated, int integrityRid)
    {
        Elevated = elevated;
        IntegrityRid = integrityRid;
    }
}

public static class GkhMaaPiCliTokenProbe
{
    [StructLayout(LayoutKind.Sequential)]
    private struct TOKEN_ELEVATION { public int TokenIsElevated; }

    [StructLayout(LayoutKind.Sequential)]
    private struct SID_AND_ATTRIBUTES { public IntPtr Sid; public UInt32 Attributes; }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(UInt32 access, bool inheritHandle, int processId);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool OpenProcessToken(IntPtr processHandle, UInt32 access, out IntPtr tokenHandle);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool GetTokenInformation(IntPtr tokenHandle, int informationClass, out TOKEN_ELEVATION information, int length, out int returnLength);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool GetTokenInformation(IntPtr tokenHandle, int informationClass, IntPtr information, int length, out int returnLength);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern IntPtr GetSidSubAuthorityCount(IntPtr sid);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern IntPtr GetSidSubAuthority(IntPtr sid, UInt32 index);

    public static GkhMaaPiCliTokenEvidence Inspect(int processId)
    {
        const UInt32 PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
        const UInt32 TOKEN_QUERY = 0x0008;
        const int TokenElevation = 20;
        const int TokenIntegrityLevel = 25;
        IntPtr process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, processId);
        if (process == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        try
        {
            IntPtr token;
            if (!OpenProcessToken(process, TOKEN_QUERY, out token)) throw new Win32Exception(Marshal.GetLastWin32Error());
            try
            {
                TOKEN_ELEVATION elevation;
                int returned;
                if (!GetTokenInformation(token, TokenElevation, out elevation, Marshal.SizeOf(typeof(TOKEN_ELEVATION)), out returned)) throw new Win32Exception(Marshal.GetLastWin32Error());
                int required = 0;
                GetTokenInformation(token, TokenIntegrityLevel, IntPtr.Zero, 0, out required);
                if (required <= 0) throw new Win32Exception(Marshal.GetLastWin32Error());
                IntPtr buffer = Marshal.AllocHGlobal(required);
                try
                {
                    if (!GetTokenInformation(token, TokenIntegrityLevel, buffer, required, out returned)) throw new Win32Exception(Marshal.GetLastWin32Error());
                    SID_AND_ATTRIBUTES label = (SID_AND_ATTRIBUTES)Marshal.PtrToStructure(buffer, typeof(SID_AND_ATTRIBUTES));
                    byte count = Marshal.ReadByte(GetSidSubAuthorityCount(label.Sid));
                    if (count == 0) throw new InvalidOperationException("Integrity SID has no sub-authority.");
                    int rid = Marshal.ReadInt32(GetSidSubAuthority(label.Sid, (UInt32)(count - 1)));
                    return new GkhMaaPiCliTokenEvidence(elevation.TokenIsElevated != 0, rid);
                }
                finally { Marshal.FreeHGlobal(buffer); }
            }
            finally { CloseHandle(token); }
        }
        finally { CloseHandle(process); }
    }
}
'@
}

function Invoke-GkhElevatedArenaWorker([string]$PreparedRuntime, [int]$WorkerTimeoutSeconds) {
    $runtime = Get-GkhFullPath $PreparedRuntime
    if (-not (Test-Path -LiteralPath $runtime -PathType Container)) {
        throw "Prepared RuntimeRoot not found: $runtime"
    }
    Assert-GkhNotReparsePoint $runtime 'Prepared RuntimeRoot'
    $manifestPath = Join-Path $runtime 'GKH_MAAPICLI_ARENA_RUN.json'
    $maaPiConfigPath = Join-Path $runtime 'config\maa_pi_config.json'
    $guiConfigPath = Join-Path $runtime 'config\config.json'
    $pipConfigPath = Join-Path $runtime 'config\pip_config.json'
    $maaOptionPath = Join-Path $runtime 'config\maa_option.json'
    $executable = Join-Path $runtime 'MaaPiCli.exe'
    foreach ($requiredPath in @(
        $manifestPath,
        $maaPiConfigPath,
        $guiConfigPath,
        $pipConfigPath,
        $maaOptionPath,
        $executable
    )) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Prepared RuntimeRoot is incomplete: $requiredPath"
        }
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskConfig = Get-Content -LiteralPath $maaPiConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $guiConfig = Get-Content -LiteralPath $guiConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $pipConfig = Get-Content -LiteralPath $pipConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $maaOption = Get-Content -LiteralPath $maaOptionPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $manifestPropertyNames = @($manifest.PSObject.Properties | ForEach-Object { $_.Name })
    $hasOwnCacheMetadata = $manifestPropertyNames -ccontains 'own_cache'
    $ownCacheMode = if ($manifestPropertyNames -ccontains 'own_cache_mode') {
        [string]$manifest.own_cache_mode
    } else {
        # Preserve compatibility with default runtimes prepared before the
        # explicit cache-reuse mode existed.
        'recalculate'
    }
    $expectedAutoRecalculateOwn = $true
    $expectedOwnRecalculationValue = 'Yes'
    switch ($ownCacheMode) {
        'recalculate' {
            if ($hasOwnCacheMetadata) {
                throw 'Prepared recalculation mode must not include explicit own-score cache metadata.'
            }
        }
        'explicit_reuse' {
            if (-not $hasOwnCacheMetadata) {
                throw 'Prepared explicit own-score cache metadata is missing.'
            }
            $expectedAutoRecalculateOwn = $false
            $expectedOwnRecalculationValue = 'No'
        }
        default {
            throw "Prepared own-score cache mode is unsupported: $ownCacheMode"
        }
    }
    $configuredTasks = @($taskConfig.task)
    $configuredOptions = if ($configuredTasks.Count -eq 1) {
        @($configuredTasks[0].option)
    } else {
        @()
    }
    $seasonOptions = @($configuredOptions | Where-Object { $_.name -ceq $script:ArenaOptionNames.season })
    $winRateOptions = @($configuredOptions | Where-Object { $_.name -ceq $script:ArenaOptionNames.win_rate })
    $ownOptions = @($configuredOptions | Where-Object { $_.name -ceq $script:ArenaOptionNames.own_recalculation })
    $captureOptions = @($configuredOptions | Where-Object { $_.name -ceq $script:ArenaOptionNames.capture })
    $winRateOptionProperties = if ($winRateOptions.Count -eq 1) {
        @($winRateOptions[0].PSObject.Properties)
    } else { @() }
    $winRateOptionPropertyNames = @($winRateOptionProperties | ForEach-Object { $_.Name })
    $winRateInputProperties = if (
        $winRateOptions.Count -eq 1 -and
        $winRateOptionPropertyNames -ccontains 'inputs'
    ) {
        @($winRateOptions[0].inputs.PSObject.Properties)
    } else { @() }
    $winRateInputPropertyNames = @($winRateInputProperties | ForEach-Object { $_.Name })
    $winRateOptionContractValid = (
        $winRateOptionPropertyNames.Count -eq 2 -and
        $winRateOptionPropertyNames -ccontains 'name' -and
        $winRateOptionPropertyNames -ccontains 'inputs' -and
        $winRateOptionPropertyNames -cnotcontains 'value' -and
        $winRateOptionPropertyNames -cnotcontains 'values' -and
        $winRateInputPropertyNames.Count -eq 3 -and
        $winRateInputPropertyNames -ccontains 'arena_win_rate_threshold_percent' -and
        $winRateInputPropertyNames -ccontains 'arena_win_rate_simulations' -and
        $winRateInputPropertyNames -ccontains 'arena_win_rate_timeout_seconds' -and
        @($winRateInputProperties | Where-Object { $_.Value -isnot [string] }).Count -eq 0
    )
    $thresholdText = if ($winRateOptionContractValid) {
        [string]$winRateOptions[0].inputs.arena_win_rate_threshold_percent
    } else { '' }
    $simulationsText = if ($winRateOptionContractValid) {
        [string]$winRateOptions[0].inputs.arena_win_rate_simulations
    } else { '' }
    $simulationTimeoutText = if ($winRateOptionContractValid) {
        [string]$winRateOptions[0].inputs.arena_win_rate_timeout_seconds
    } else { '' }
    $parsedThreshold = 0
    $parsedSimulations = 0
    $parsedSimulationTimeout = 0
    $parametersValid = (
        [int]::TryParse($thresholdText, [ref]$parsedThreshold) -and
        [int]::TryParse($simulationsText, [ref]$parsedSimulations) -and
        [int]::TryParse($simulationTimeoutText, [ref]$parsedSimulationTimeout) -and
        $parsedThreshold -ge 0 -and
        $parsedThreshold -le 100 -and
        $parsedSimulations -ge 1000 -and
        $parsedSimulationTimeout -gt 0
    )
    if (
        $manifest.task -cne $script:ArenaTaskName -or
        $manifest.updates_enabled -ne $false -or
        $manifest.season -cne 'latest' -or
        $manifest.auto_recalculate_own -ne $expectedAutoRecalculateOwn -or
        $manifest.capture_mode -cne 'off' -or
        $configuredTasks.Count -ne 1 -or
        $configuredTasks[0].name -cne $script:ArenaTaskName -or
        $configuredOptions.Count -ne 4 -or
        $seasonOptions.Count -ne 1 -or
        $seasonOptions[0].value -cne 'latest' -or
        $winRateOptions.Count -ne 1 -or
        -not $winRateOptionContractValid -or
        $ownOptions.Count -ne 1 -or
        $ownOptions[0].value -cne $expectedOwnRecalculationValue -or
        $captureOptions.Count -ne 1 -or
        $captureOptions[0].value -cne $script:ArenaOptionNames.capture_off -or
        -not $parametersValid -or
        [int]$manifest.threshold_percent -ne $parsedThreshold -or
        [int]$manifest.simulations -ne $parsedSimulations -or
        [int]$manifest.simulation_timeout_seconds -ne $parsedSimulationTimeout -or
        $taskConfig.controller.name -cne 'PC' -or
        $taskConfig.resource -cne 'DMM' -or
        $guiConfig.EnableAutoUpdateResource -ne $false -or
        $pipConfig.enable_pip_update -ne $false -or
        $pipConfig.enable_pip_install -ne $false -or
        $maaOption.save_draw -ne $false -or
        $maaOption.save_on_error -ne $false
    ) {
        throw 'Prepared MaaPiCli arena configuration changed or is not single-task PC/DMM.'
    }
    if ($ownCacheMode -ceq 'explicit_reuse') {
        Assert-GkhOwnScoreCacheManifest $runtime $manifest.own_cache $parsedSimulations
    }

    Add-GkhTokenProbeType
    $runnerToken = [GkhMaaPiCliTokenProbe]::Inspect($PID)
    if (-not $runnerToken.Elevated -or $runnerToken.IntegrityRid -lt 0x3000) {
        throw "Elevated MaaPiCli worker is not elevated/high integrity: RID=$($runnerToken.IntegrityRid)."
    }

    $tempRoot = Join-Path $runtime '.local\runtime-temp'
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
    Assert-GkhNotReparsePoint $tempRoot 'MaaPiCli TEMP root'
    $runtimeTemp = Join-Path $tempRoot ([guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $runtimeTemp | Out-Null
    Assert-GkhNotReparsePoint $runtimeTemp 'MaaPiCli TEMP leaf'
    $env:TEMP = $runtimeTemp
    $env:TMP = $runtimeTemp
    $env:PATH = $runtime + ';' + (Join-Path $runtime 'python') + ';' + $env:PATH

    $stdoutPath = Join-Path $runtime 'maapicli-arena.stdout.log'
    $stderrPath = Join-Path $runtime 'maapicli-arena.stderr.log'
    $elevationPath = Join-Path $runtime 'maapicli-arena-elevation.json'
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $executable
    $startInfo.Arguments = '-d'
    $startInfo.WorkingDirectory = $runtime
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [Text.Encoding]::UTF8

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw 'MaaPiCli process did not start.'
    }

    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    try {
        $maaToken = [GkhMaaPiCliTokenProbe]::Inspect($process.Id)
        if (
            -not $maaToken.Elevated -or
            $maaToken.IntegrityRid -ne $runnerToken.IntegrityRid
        ) {
            & (Join-Path $env:SystemRoot 'System32\taskkill.exe') /PID $process.Id /T /F | Out-Null
            throw "MaaPiCli token does not match the elevated worker: RID=$($maaToken.IntegrityRid)."
        }
        Write-GkhUtf8Json $elevationPath ([ordered]@{
            schema_version = 1
            checked_at_utc = [DateTime]::UtcNow.ToString('o')
            working_directory = $runtime
            arguments = '-d'
            worker = [ordered]@{
                process_id = $PID
                elevated = [bool]$runnerToken.Elevated
                integrity_rid = [int]$runnerToken.IntegrityRid
            }
            maapicli = [ordered]@{
                process_id = $process.Id
                executable = $executable
                elevated = [bool]$maaToken.Elevated
                integrity_rid = [int]$maaToken.IntegrityRid
            }
        })

        $timeoutMilliseconds = [int64]$WorkerTimeoutSeconds * 1000
        if (-not $process.WaitForExit([int]$timeoutMilliseconds)) {
            & (Join-Path $env:SystemRoot 'System32\taskkill.exe') /PID $process.Id /T /F | Out-Null
            if (-not $process.WaitForExit(10000)) {
                throw 'Timed-out MaaPiCli process tree could not be terminated.'
            }
            [IO.File]::WriteAllText($stdoutPath, $stdoutTask.Result, [Text.UTF8Encoding]::new($false))
            [IO.File]::WriteAllText($stderrPath, $stderrTask.Result, [Text.UTF8Encoding]::new($false))
            $null = Write-GkhArenaRunSummary $runtime 124
            return 124
        }
        $process.WaitForExit()
        [IO.File]::WriteAllText($stdoutPath, $stdoutTask.Result, [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText($stderrPath, $stderrTask.Result, [Text.UTF8Encoding]::new($false))
        $processExitCode = [int]$process.ExitCode
        $runSummary = Write-GkhArenaRunSummary $runtime $processExitCode
        if ($processExitCode -eq 0 -and -not $runSummary.strict_product_closed) {
            [Console]::Error.WriteLine(
                'MaaPiCli exited successfully but the arena challenge lifecycle is unfinished; ' +
                "see $runtime\maapicli-arena-summary.json"
            )
            return 125
        }
        return $processExitCode
    } catch {
        if (-not $process.HasExited) {
            & (Join-Path $env:SystemRoot 'System32\taskkill.exe') /PID $process.Id /T /F | Out-Null
        }
        throw
    } finally {
        $process.Dispose()
    }
}

if ($ElevatedWorker) {
    try {
        exit (Invoke-GkhElevatedArenaWorker $RuntimeRoot $TimeoutSeconds)
    } catch {
        [Console]::Error.WriteLine($_.Exception.Message)
        exit 1
    }
}

try {
    $reuseOwnScoreCache = $PSBoundParameters.ContainsKey('ReuseOwnScoreCachePath')
    if ($reuseOwnScoreCache -and [string]::IsNullOrWhiteSpace($ReuseOwnScoreCachePath)) {
        throw 'ReuseOwnScoreCachePath must name one explicit compatible own-score cache file.'
    }
    $preparedRuntimeState = New-GkhIsolatedArenaRuntime `
        $SourceVersionRoot `
        $RuntimeRoot `
        $ThresholdPercent `
        $Simulations `
        $SimulationTimeoutSeconds `
        $reuseOwnScoreCache `
        $ReuseOwnScoreCachePath
    $preparedRuntime = $preparedRuntimeState.RuntimeRoot
    if ($PrepareOnly) {
        $prepareResult = [ordered]@{
            status = 'prepared'
            runtime_root = $preparedRuntime
            task = $script:ArenaTaskName
            season = 'latest'
            threshold_percent = $ThresholdPercent
            simulations = $Simulations
            simulation_timeout_seconds = $SimulationTimeoutSeconds
            auto_recalculate_own = $preparedRuntimeState.AutoRecalculateOwn
            capture_mode = 'off'
            updates_enabled = $false
            executed = $false
        }
        if ($reuseOwnScoreCache) {
            $prepareResult['own_cache_mode'] = 'explicit_reuse'
            $prepareResult['own_cache_sha256'] = $preparedRuntimeState.OwnCacheMetadata.sha256
            $prepareResult['own_cache_capture_id'] = $preparedRuntimeState.OwnCacheMetadata.capture_id
        }
        [pscustomobject]$prepareResult | ConvertTo-Json -Compress
        exit 0
    }

    # $PSHOME may point at the bundled PowerShell 7 host used by Codex, where
    # the executable is pwsh.exe.  RunAs must target the system Windows
    # PowerShell explicitly so the launcher works from either host.
    $elevatedPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $elevatedPowerShell -PathType Leaf)) {
        throw "Windows PowerShell not found: $elevatedPowerShell"
    }
    $payload = [ordered]@{
        script_path = [IO.Path]::GetFullPath($PSCommandPath)
        runtime_root = $preparedRuntime
        timeout_seconds = $TimeoutSeconds
    } | ConvertTo-Json -Compress
    $payloadBase64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
    $innerScript = @"
`$payload = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('$payloadBase64')) | ConvertFrom-Json
& `$payload.script_path -RuntimeRoot `$payload.runtime_root -TimeoutSeconds ([int]`$payload.timeout_seconds) -ElevatedWorker
exit [int]`$LASTEXITCODE
"@
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($innerScript))
    $elevated = Start-Process -FilePath $elevatedPowerShell -ArgumentList @(
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-EncodedCommand',
        $encodedCommand
    ) -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ErrorAction Stop
    if ($null -eq $elevated) {
        throw 'Elevated MaaPiCli worker process was not created.'
    }
    exit [int]$elevated.ExitCode
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
