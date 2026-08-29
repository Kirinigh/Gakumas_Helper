@echo off
setlocal

set "MAA_ROOT=%~dp0"
set "MAA_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%MAA_ROOT%current\MaaGakumasu.exe" goto :stable_root
if exist "%MAA_ROOT%..\MaaGakumasu.exe" goto :version_local

:stable_root
set "MAA_EXE=%MAA_ROOT%current\MaaGakumasu.exe"
set "MAA_HELPER=%MAA_ROOT%.Start-MaaGakumasu-Admin.ps1"
goto :paths_ready

:version_local
for %%I in ("%MAA_ROOT%..") do set "MAA_VERSION_ROOT=%%~fI"
set "MAA_EXE=%MAA_VERSION_ROOT%\MaaGakumasu.exe"
set "MAA_HELPER=%MAA_ROOT%.Start-MaaGakumasu-Admin.ps1"

:paths_ready

if /i "%~1"=="--check" (
  if not exist "%MAA_EXE%" (
    echo MaaGakumasu.exe not found: "%MAA_EXE%"
    endlocal & exit /b 1
  )
  if not exist "%MAA_HELPER%" (
    echo Elevated launcher helper not found: "%MAA_HELPER%"
    endlocal & exit /b 1
  )
  if not exist "%MAA_POWERSHELL%" (
    echo Windows PowerShell not found: "%MAA_POWERSHELL%"
    endlocal & exit /b 1
  )
  echo [MaaGakumasu] Administrator launcher is ready.
  echo [MaaGakumasu] Active executable: "%MAA_EXE%"
  echo [MaaGakumasu] Elevated helper: "%MAA_HELPER%"
  endlocal & exit /b 0
)

if not exist "%MAA_EXE%" (
  echo MaaGakumasu.exe not found: "%MAA_EXE%"
  pause
  endlocal & exit /b 1
)
if not exist "%MAA_HELPER%" (
  echo Elevated launcher helper not found: "%MAA_HELPER%"
  pause
  endlocal & exit /b 1
)
if not exist "%MAA_POWERSHELL%" (
  echo Windows PowerShell not found: "%MAA_POWERSHELL%"
  pause
  endlocal & exit /b 1
)

set "MAA_ELEVATED_HELPER=%MAA_HELPER%"
"%MAA_POWERSHELL%" -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; try { $helper = [IO.Path]::GetFullPath($env:MAA_ELEVATED_HELPER); if (-not (Test-Path -LiteralPath $helper -PathType Leaf)) { throw ('Elevated launcher helper not found: ' + $helper) }; $elevatedPowerShell = Join-Path $PSHOME 'powershell.exe'; if (-not (Test-Path -LiteralPath $elevatedPowerShell -PathType Leaf)) { throw ('Windows PowerShell not found: ' + $elevatedPowerShell) }; $helperPayload = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($helper)); $script = '$helper = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String(''' + $helperPayload + ''')); & $helper'; $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script)); $process = Start-Process -FilePath $elevatedPowerShell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded) -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ErrorAction Stop; if ($null -eq $process) { throw 'Elevated launcher process was not created.' }; exit [int]$process.ExitCode } catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }"
set "MAA_EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %MAA_EXIT_CODE%
