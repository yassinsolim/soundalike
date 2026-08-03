[CmdletBinding()]
param(
    [ValidateSet("Install", "Status", "Uninstall")]
    [string]$Action = "Install",
    [string]$PythonPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$taskName = "Soundalike Local Server"
$stateDirectory = Join-Path $env:LOCALAPPDATA "Soundalike"
$launcherPath = Join-Path $stateDirectory "start-server.py"
$legacyLauncherPath = Join-Path $stateDirectory "start-server.ps1"
$logPath = Join-Path $stateDirectory "server.log"
$healthUri = "http://127.0.0.1:8787/health"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$sourcePath = Join-Path $repositoryRoot "src"

function Get-SoundalikeTask {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}

function Get-SoundalikeHealth {
    try {
        $response = Invoke-RestMethod $healthUri -TimeoutSec 3
        if ($response.ok) {
            return $response
        }
    }
    catch {
        return $null
    }
    return $null
}

function Show-SoundalikeStatus {
    $task = Get-SoundalikeTask
    $health = Get-SoundalikeHealth

    if ($null -eq $task) {
        Write-Host "Auto-start task: not installed"
    }
    else {
        $info = Get-ScheduledTaskInfo -TaskName $taskName
        Write-Host "Auto-start task: $($task.State)"
        Write-Host "Last task result: $($info.LastTaskResult)"
    }

    if ($null -eq $health) {
        Write-Host "Local server: offline"
    }
    else {
        Write-Host "Local server: ready ($($health.library) tracks)"
    }
    Write-Host "Log: $logPath"

    return ($null -ne $task -and $null -ne $health)
}

if ($Action -eq "Status") {
    if (-not (Show-SoundalikeStatus)) {
        exit 1
    }
    exit 0
}

if ($Action -eq "Uninstall") {
    $task = Get-SoundalikeTask
    if ($null -ne $task) {
        if ($task.State -eq "Running") {
            Stop-ScheduledTask -TaskName $taskName
        }
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
    if (Test-Path -LiteralPath $launcherPath) {
        Remove-Item -LiteralPath $launcherPath -Force
    }
    if (Test-Path -LiteralPath $legacyLauncherPath) {
        Remove-Item -LiteralPath $legacyLauncherPath -Force
    }
    Write-Host "Removed Soundalike auto-start. Existing logs remain at $logPath"
    exit 0
}

$candidates = [System.Collections.Generic.List[string]]::new()
if ($PythonPath) {
    $candidates.Add($PythonPath)
}
else {
    if ($env:VIRTUAL_ENV) {
        $candidates.Add((Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))
    }
    $candidates.Add((Join-Path $repositoryRoot ".venv\Scripts\python.exe"))
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        $candidates.Add($pythonCommand.Source)
    }
}

$selectedPython = $null
$validationFailures = [System.Collections.Generic.List[string]]::new()
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($previousPythonPath) {
    "$sourcePath;$previousPythonPath"
}
else {
    $sourcePath
}

try {
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            $validationFailures.Add("$candidate does not exist")
            continue
        }
        $validation = & $candidate -c "import librosa, soundalike.cli, torch, torchaudio" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $selectedPython = (Resolve-Path $candidate).Path
            break
        }
        $validationFailures.Add("$candidate cannot load soundalike: $validation")
    }
}
finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }
}

if ($null -eq $selectedPython) {
    $details = $validationFailures -join [Environment]::NewLine
    throw "No usable Python environment found. Activate the environment where .[ml] is installed or pass -PythonPath.`n$details"
}

New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
if (Test-Path -LiteralPath $legacyLauncherPath) {
    Remove-Item -LiteralPath $legacyLauncherPath -Force
}

$launcherTemplate = @'
from pathlib import Path
import sys

repository_root = Path(__REPOSITORY_ROOT__)
sys.path.insert(0, str(repository_root / "src"))
log_path = Path(__LOG_PATH__)

with log_path.open("w", encoding="utf-8", buffering=1) as log:
    sys.stdout = log
    sys.stderr = log
    from soundalike.cli import main
    raise SystemExit(main(["serve", "--no-browser"]))
'@

function ConvertTo-PythonLiteral([string]$Value) {
    return ConvertTo-Json $Value -Compress
}

$launcher = $launcherTemplate.
    Replace("__REPOSITORY_ROOT__", (ConvertTo-PythonLiteral $repositoryRoot)).
    Replace("__LOG_PATH__", (ConvertTo-PythonLiteral $logPath))
[System.IO.File]::WriteAllText(
    $launcherPath,
    $launcher,
    [System.Text.UTF8Encoding]::new($false)
)

$existingTask = Get-SoundalikeTask
if ($null -ne $existingTask -and $existingTask.State -eq "Running") {
    Stop-ScheduledTask -TaskName $taskName
    $stopDeadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $existingHealth = Get-SoundalikeHealth
    } while ($null -ne $existingHealth -and [DateTime]::UtcNow -lt $stopDeadline)
    if ($null -ne $existingHealth) {
        throw "The previous task stopped but its server is still running. End that task's process and rerun the installer."
    }
}

$userId = "$env:USERDOMAIN\$env:USERNAME"
$pythonwPath = Join-Path (Split-Path -Parent $selectedPython) "pythonw.exe"
$taskPython = if (Test-Path -LiteralPath $pythonwPath -PathType Leaf) {
    $pythonwPath
}
else {
    $selectedPython
}
$taskArguments = "`"$launcherPath`""
$taskAction = New-ScheduledTaskAction `
    -Execute $taskPython `
    -Argument $taskArguments `
    -WorkingDirectory $repositoryRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable
$task = New-ScheduledTask `
    -Action $taskAction `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Keeps the local Soundalike recommendation engine available for Spicetify."

Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $taskName

$deadline = [DateTime]::UtcNow.AddMinutes(10)
$nextProgress = [DateTime]::UtcNow.AddSeconds(30)
do {
    Start-Sleep -Seconds 2
    $health = Get-SoundalikeHealth
    if ($null -ne $health) {
        Write-Host "Soundalike auto-start installed and running."
        Write-Host "Local server: ready ($($health.library) tracks)"
        Write-Host "Task: $taskName"
        Write-Host "Log: $logPath"
        exit 0
    }
    $runningTask = Get-SoundalikeTask
    if ($null -eq $runningTask -or $runningTask.State -ne "Running") {
        throw "The task stopped before the server became ready. Check $logPath"
    }
    if ([DateTime]::UtcNow -ge $nextProgress) {
        Write-Host "Soundalike is still warming its local model..."
        $nextProgress = [DateTime]::UtcNow.AddSeconds(30)
    }
} while ([DateTime]::UtcNow -lt $deadline)

throw "The task was installed, but the server did not become ready within 10 minutes. Check $logPath"
