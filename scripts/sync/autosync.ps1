param(
    [string]$RepoDir = "D:\cityu\学校事务\Working\world model\VAGEN-navigation-repro"
)

$ErrorActionPreference = "Continue"
$LogDir = Join-Path $RepoDir "logs"
$LogFile = Join-Path $LogDir "autosync.log"
$ConflictLog = Join-Path $LogDir "sync-conflicts.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-SyncLog {
    param([string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $LogFile -Value "[$stamp] $Message"
}

Set-Location -LiteralPath $RepoDir

try {
    git rev-parse --is-inside-work-tree *> $null
} catch {
    Write-SyncLog "not a git repository: $RepoDir"
    exit 1
}

try {
    git remote get-url origin *> $null
} catch {
    Write-SyncLog "origin remote is not configured"
    exit 0
}

git pull --rebase --autostash origin main *>> $LogFile
if ($LASTEXITCODE -ne 0) {
    Add-Content -Path $ConflictLog -Value "[$((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))] pull/rebase failed; manual resolution required"
    exit 0
}

git add -- README.md README_REPRO.md .gitignore 2>$null
git add -- ":(glob)scripts/**/*.sh" ":(glob)scripts/**/*.sbatch" ":(glob)scripts/**/*.ps1" 2>$null
git add -- ":(glob)scripts/**/*.yaml" ":(glob)scripts/**/*.yml" 2>$null
git add -- ":(glob)runs/**/*.md" ":(glob)runs/**/*.json" 2>$null

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-SyncLog "no allowlisted changes"
    exit 0
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
git commit -m "autosync: $stamp" *>> $LogFile
git push origin main *>> $LogFile
if ($LASTEXITCODE -ne 0) {
    Add-Content -Path $ConflictLog -Value "[$stamp] push failed; manual resolution required"
    exit 0
}

Write-SyncLog "sync complete"
