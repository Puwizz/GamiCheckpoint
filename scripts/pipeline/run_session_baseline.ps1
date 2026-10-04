param(
    [int]$Epochs = 12,
    [switch]$PrepareOnly,
    [switch]$OverwriteGeneratedData
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$SourceDir = Join-Path $ProjectRoot 'data\sessions'
$ExperimentRoot = Join-Path $ProjectRoot 'data\step6\session_baselines'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

function Invoke-Experiment {
    param(
        [string]$Name,
        [string]$TrainSession,
        [string]$ValSession
    )

    $DataDir = Join-Path $ExperimentRoot $Name
    $CheckpointDir = Join-Path $ProjectRoot "checkpoints\session_baselines\$Name"
    $PlotsDir = Join-Path $ProjectRoot "outputs\plots\session_baselines\$Name"

    Write-Host "`n=== $($Name): train=$TrainSession, validation=$ValSession ===" -ForegroundColor Cyan
    $PrepareArgs = @(
        'scripts\dataset\prepare_session_baseline.py',
        '--source-dir', $SourceDir,
        '--train-session', $TrainSession,
        '--val-session', $ValSession,
        '--output-dir', $DataDir
    )
    if ($OverwriteGeneratedData) {
        $PrepareArgs += '--overwrite'
    }
    & $Python @PrepareArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed for $Name" }

    if ($PrepareOnly) { return }

    & $Python 'scripts\training\train.py' `
        '--data-dir' $DataDir `
        '--checkpoints-dir' $CheckpointDir `
        '--plots-dir' $PlotsDir `
        '--epochs' $Epochs
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $Name" }

    & $Python 'scripts\evaluation\evaluate.py' `
        '--data-dir' $DataDir `
        '--split' 'val' `
        '--model' (Join-Path $CheckpointDir 'origami_step6_best.pth') `
        '--plots-dir' $PlotsDir
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $Name" }
}

Invoke-Experiment -Name 'noon_train_night_val' -TrainSession 'noon_session' -ValSession 'night_session'
Invoke-Experiment -Name 'night_train_noon_val' -TrainSession 'night_session' -ValSession 'noon_session'

Write-Host "`nFinished. Compare reports under outputs\plots\session_baselines\." -ForegroundColor Green
