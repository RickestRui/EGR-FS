param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('train','val','lock','test','robustness','aggregate')]
    [string]$Phase,
    [ValidateSet('main','ablation','sensitivity','confirmatory','all')]
    [string]$Family='main',
    [string]$Root='experiments/egr_fs_suite'
)
$ErrorActionPreference='Stop'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
function Invoke-Experiment {
    param([string[]]$Arguments)
    & python -m experiments.paper --root $Root @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $Arguments" }
}
if ($Phase -eq 'aggregate') {
    Invoke-Experiment -Arguments @('aggregate')
    return
}
$rows=@(Import-Csv -LiteralPath (Join-Path $Root 'manifest.csv') -Encoding UTF8 | Where-Object {
    $Family -eq 'all' -or $_.family -eq $Family -or
    ($Family -eq 'confirmatory' -and $_.family -ne 'sensitivity')
})
if ($rows.Count -eq 0) { throw 'No matching runs.' }
if ($Phase -in @('lock','test','robustness') -and @($rows | Where-Object family -eq 'sensitivity').Count -gt 0) {
    throw 'Sensitivity runs are validation-only.'
}
if ($Phase -eq 'lock') {
    $names=@($rows.experiment | Select-Object -Unique)
    $seeds=@($rows.seed | Select-Object -Unique)
    Invoke-Experiment -Arguments (@('lock','--experiments')+$names+@('--seeds')+$seeds)
    return
}
foreach ($row in $rows) {
    Write-Host "$Phase : $($row.experiment), seed $($row.seed)"
    $selection=@('--experiment',$row.experiment,'--seed',$row.seed)
    if ($Phase -eq 'train') { Invoke-Experiment -Arguments (@('train')+$selection) }
    elseif ($Phase -eq 'val') { Invoke-Experiment -Arguments (@('evaluate')+$selection+@('--split','val')) }
    elseif ($Phase -eq 'test') {
        Invoke-Experiment -Arguments (@('evaluate')+$selection+@('--split','test','--save-predictions'))
    }
    elseif ($Phase -eq 'robustness') {
        Invoke-Experiment -Arguments (@('evaluate')+$selection+@('--split','test','--conditions',
          'rgb_dark_05','rgb_dark_025','rgb_missing','ir_missing','ir_shift_2','ir_shift_4','ir_shift_8'))
    }
}
