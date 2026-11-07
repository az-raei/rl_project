# PowerShell version of scripts/reproduce.sh  (Windows PowerShell 5.1 or PowerShell 7).
#
#   .\scripts\reproduce.ps1 try1 quick                      # all stages, small budgets
#   .\scripts\reproduce.ps1 try1 quick tests baselines      # only some stages
#   $env:TASKS = "reach-v3"; .\scripts\reproduce.ps1 try1 quick tests
#   .\scripts\reproduce.ps1 final full
# stages (default = all): tests baselines ablations querying iql report
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process Bypass
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Name,
    [Parameter(Position = 1)][ValidateSet("quick", "full")][string]$Profile = "quick",
    [Parameter(Position = 2, ValueFromRemainingArguments = $true)][string[]]$Stage
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not $Stage -or $Stage.Count -eq 0) { $Stage = @("tests", "baselines", "ablations", "querying", "iql", "report") }
$Tasks = if ($env:TASKS) { $env:TASKS } else { "reach-v3,pick-place-v3" }

if ($Profile -eq "quick") {
    $Steps = 2000
    $IqlArgs = @("iql.total_steps=20000", "iql.eval_episodes=5", "iql.users_per_group=1", "iql.budgets=[100]")
    $AblArgs = @("exp.k_values=[4,8,16]", "exp.temporal_values=[gru]", "exp.l2_values=[0.1,10]")
} else {
    $Steps = 6000
    $IqlArgs = @("iql.total_steps=100000", "iql.eval_episodes=10", "iql.users_per_group=2")
    $AblArgs = @()
}
$TaskArg = "tasks=[$Tasks]"      # always a quoted string: an unquoted [a,b] would be split at the comma by PowerShell

function Invoke-Py {
    param([string[]]$PyArgs)
    Write-Host ("$ python " + ($PyArgs -join " "))
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "command failed (exit code $LASTEXITCODE): python $($PyArgs -join ' ')" }
}

foreach ($st in $Stage) {
    Write-Host ""
    Write-Host "=================== stage: $st ($Profile) ==================="
    switch ($st) {
        "tests"     { Invoke-Py @("tests/run_tests.py") }
        "baselines" { Invoke-Py (@("main.py", "baselines", $TaskArg, "exp.steps=$Steps")) }
        "ablations" { Invoke-Py (@("main.py", "ablations", $TaskArg, "exp.steps=$Steps") + $AblArgs) }
        "querying"  { Invoke-Py (@("main.py", "querying", $TaskArg, "exp.steps=$Steps")) }
        "iql" {
            & python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('metaworld') else 1)"
            if ($LASTEXITCODE -ne 0) { & pip install -q metaworld }
            Invoke-Py (@("main.py", "iql", $TaskArg) + $IqlArgs)
        }
        "report"    { Invoke-Py @("main.py", "report", $TaskArg) }
        default     { throw "unknown stage: $st" }
    }
}

$Out = "runs/$Name"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
foreach ($d in @("results/logs", "results/tables", "results/figures")) {
    if (Test-Path $d) { Copy-Item -Recurse -Force $d $Out }
}
$commit = (& git rev-parse --short HEAD 2>$null)
@("experiment: $Name ($Profile)", "stages: $($Stage -join ' ')", "tasks: $Tasks", "commit: $commit") | Set-Content "$Out/INFO.txt"
Write-Host ""
Write-Host "== done. results in $Out  (push with: .\scripts\push_results.ps1 $Name) =="
