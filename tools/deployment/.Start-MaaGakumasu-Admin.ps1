[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'The MaaGakumasu launcher helper must run in an elevated PowerShell process.'
}

$versionRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'current'))
$versionExecutable = Join-Path $versionRoot 'MaaGakumasu.exe'
if (-not (Test-Path -LiteralPath $versionExecutable -PathType Leaf)) {
    $versionRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
}
$executable = Join-Path $versionRoot 'MaaGakumasu.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "MaaGakumasu.exe not found: $executable"
}

$tempRoot = Join-Path $versionRoot '.local\runtime-temp'
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$tempRootItem = Get-Item -LiteralPath $tempRoot -Force
if (($tempRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Runtime TEMP root must not be a reparse point: $tempRoot"
}

$runtimeTemp = Join-Path $tempRoot ([guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $runtimeTemp | Out-Null
$runtimeTempItem = Get-Item -LiteralPath $runtimeTemp -Force
if (($runtimeTempItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Runtime TEMP leaf must not be a reparse point: $runtimeTemp"
}
if (Get-ChildItem -LiteralPath $runtimeTemp -Force | Select-Object -First 1) {
    throw "Runtime TEMP leaf was not empty after creation: $runtimeTemp"
}

# Scope the clean TEMP/TMP path to this elevated process and the Maa/Agent
# process tree only. Existing sockets and older owned leaves are untouched.
$env:TEMP = $runtimeTemp
$env:TMP = $runtimeTemp

Start-Process -FilePath $executable -WorkingDirectory $versionRoot | Out-Null
