param(
    [string]$RepoDir = "D:\cityu\学校事务\Working\world model\VAGEN-navigation-repro",
    [string]$TaskName = "VAGENNavigationReproAutosync"
)

$ErrorActionPreference = "Stop"
$ScriptPath = Join-Path $RepoDir "scripts\sync\autosync.ps1"

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Autosync script not found: $ScriptPath"
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -RepoDir `"$RepoDir`""

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null

Write-Host "Installed scheduled task $TaskName for $RepoDir"
