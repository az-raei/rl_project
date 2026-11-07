# PowerShell version of scripts/push_results.sh: push runs/<name>/ to a NEW branch results-<name> (never main).
#   $env:GH_TOKEN = "<fine-grained token: Contents read+write on this repo>"
#   .\scripts\push_results.ps1 try1
param([Parameter(Mandatory = $true, Position = 0)][string]$Name)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not $env:GH_TOKEN) { throw 'set the token first:  $env:GH_TOKEN = "<token>"' }
if (-not (Test-Path "runs/$Name")) { throw "runs/$Name not found. Run scripts\reproduce.ps1 $Name first." }
$remote = (& git config --get remote.origin.url)
$slug = ($remote -replace '^(https://([^@/]*@)?|git@)github\.com[:/]', '') -replace '\.git$', ''
$orig = (& git rev-parse --abbrev-ref HEAD)
& git config user.email "run@users.noreply.github.com"
& git config user.name "run"
& git checkout -q -B "results-$Name"
& git add -f "runs/$Name"
& git commit -q -m "results: $Name"          # exits non-zero if there is nothing new; that is fine
$url = "https://x-access-token:$($env:GH_TOKEN)@github.com/$slug.git"
& git push -q --force $url "results-$Name" 2>&1 | ForEach-Object { $_ -replace [regex]::Escape($env:GH_TOKEN), "***" }
& git checkout -q $orig
Write-Host ""
Write-Host "pushed. Paste this link to Claude:"
Write-Host "https://github.com/$slug/tree/results-$Name/runs/$Name/tables"
