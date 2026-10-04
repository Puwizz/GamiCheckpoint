param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 12,
    [string]$RunName = "",
    [switch]$PreflightOnly,
    [switch]$ReusePreparedData,
    [switch]$OverwritePreparedData,
    [switch]$FineTune,
    [switch]$AutoExtract,
    [switch]$FilterQuality,
    [switch]$GenerateEDA
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Config = Join-Path $Root "configs\five_session_split.json"
$PreparedData = Join-Path $Root "data\step6\five_session\processed"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Five-session config not found: $Config"
}
if ($ReusePreparedData -and $OverwritePreparedData) {
    throw "Choose only one: -ReusePreparedData or -OverwritePreparedData."
}

Push-Location $Root
$PipelineError = $null
try {
    Write-Host "`n[1/5] Auditing all raw sessions..." -ForegroundColor Cyan
    
    $ConfigJson = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    $ConfigSessions = @($ConfigJson.splits.train) + @($ConfigJson.splits.val) + @($ConfigJson.splits.test) | Select-Object -Unique

    if ($AutoExtract) {
        Write-Host "Running Auto Frame Extraction for raw video sessions (0.3s interval)..." -ForegroundColor Yellow
        foreach ($s in $ConfigSessions) {
            $sdir = Join-Path $ConfigJson.source_dir $s
            $videoDir = Join-Path $sdir "raw_data_video"
            if (Test-Path $videoDir) {
                Invoke-Python "scripts\utils\auto_extract_frames.py" "--session-dir" $sdir
            }
        }
    }
    
    if ($FilterQuality) {
        Write-Host "Running Quality Filter (Blur & Dupes)..." -ForegroundColor Yellow
        foreach ($s in $ConfigSessions) {
            $sdir = Join-Path $ConfigJson.source_dir $s
            $videoDir = Join-Path $sdir "raw_data_video"
            if (Test-Path $videoDir) {
                Invoke-Python "scripts\utils\quality_filter.py" "--session-dir" $sdir
            }
        }
    }

    if ($GenerateEDA -or $AutoExtract -or $FilterQuality) {
        Write-Host "Generating EDA Plot..." -ForegroundColor Yellow
        Invoke-Python "scripts\utils\generate_eda.py"
    }
    
    Invoke-Python "scripts\dataset\prepare_dataset.py" "--config" $Config "--validate-only"
    if ($PreflightOnly) {
        Write-Host "`nPreflight passed. No dataset was copied and no model was trained." -ForegroundColor Green
        return
    }

    Write-Host "`n[2/5] Preparing ImageFolder dataset..." -ForegroundColor Cyan
    if ($ReusePreparedData) {
        $Marker = Join-Path $PreparedData ".gamisenpai_generated_dataset.json"
        if (-not (Test-Path -LiteralPath $Marker -PathType Leaf)) {
            throw "Prepared data is missing or unmarked: $PreparedData"
        }
        Write-Host "Reusing prepared data: $PreparedData"
    }
    else {
        $PrepareArgs = @(
            "scripts\dataset\prepare_dataset.py",
            "--config", $Config,
            "--output-dir", $PreparedData
        )
        if ($OverwritePreparedData) {
            $PrepareArgs += "--overwrite"
        }
        Invoke-Python @PrepareArgs
    }

    if ([string]::IsNullOrWhiteSpace($RunName)) {
        $RunName = Get-Date -Format "yyyyMMdd_HHmmss"
    }
    if ($RunName -notmatch "^[A-Za-z0-9._-]+$") {
        throw "RunName may contain only letters, numbers, dot, underscore, and dash."
    }

    $CheckpointDir = Join-Path $Root "checkpoints\five_session_runs\$RunName"
    $PlotsDir = Join-Path $Root "outputs\plots\five_session_runs\$RunName"
    if ((Test-Path -LiteralPath $CheckpointDir) -or (Test-Path -LiteralPath $PlotsDir)) {
        throw "Run already exists: $RunName. Choose a new -RunName."
    }

    Write-Host "`n[3/5] Training run '$RunName'..." -ForegroundColor Cyan
    $TrainArgs = @(
        "scripts\training\train.py",
        "--data-dir", $PreparedData,
        "--checkpoints-dir", $CheckpointDir,
        "--plots-dir", $PlotsDir,
        "--epochs", $Epochs,
        "--workers", "4"
    )
    if ($FineTune) {
        $TrainArgs += "--fine-tune"
    }
    Invoke-Python @TrainArgs

    $BestModel = Join-Path $CheckpointDir "origami_step6_best.pth"
    Write-Host "`n[4/5] Evaluating validation and untouched test sessions..." -ForegroundColor Cyan
    Invoke-Python "scripts\evaluation\evaluate.py" "--data-dir" $PreparedData "--split" "val" "--model" $BestModel "--plots-dir" $PlotsDir
    Invoke-Python "scripts\evaluation\evaluate.py" "--data-dir" $PreparedData "--split" "test" "--model" $BestModel "--plots-dir" $PlotsDir

    Write-Host "`n[5/5] Loading the candidate through the real API..." -ForegroundColor Cyan
    $SmokeImage = Get-ChildItem -LiteralPath (Join-Path $PreparedData "test\Chk6") -File | Select-Object -First 1
    if ($null -eq $SmokeImage) {
        throw "No Chk6 test image is available for the API smoke test."
    }
    $SmokeReport = Join-Path $PlotsDir "api_smoke_test.json"
    Invoke-Python "tests\smoke_test_checkpoint.py" "--model" $BestModel "--image" $SmokeImage.FullName "--output" $SmokeReport

    $RunMetadata = [ordered]@{
        run_name = $RunName
        completed_at = (Get-Date).ToString("o")
        epochs = $Epochs
        fine_tune = [bool]$FineTune
        config = $Config
        prepared_data = $PreparedData
        best_model = $BestModel
        plots_dir = $PlotsDir
    }
    New-Item -ItemType Directory -Force -Path $PlotsDir | Out-Null
    $RunMetadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PlotsDir "run_metadata.json") -Encoding UTF8

    Write-Host "`nFive-session pipeline completed." -ForegroundColor Green
    Write-Host "Best model: $BestModel"
    Write-Host "Reports:    $PlotsDir"
    Write-Host "`nTest this candidate on the web without replacing the current model:"
    Write-Host ('$env:GAMISENPAI_MODEL_PATH = "{0}"' -f $BestModel)
    Write-Host ".\.venv\Scripts\python.exe -m uvicorn app.app:app --reload"
}
catch {
    $PipelineError = $_.Exception.Message
}
finally {
    Pop-Location
}

if ($null -ne $PipelineError) {
    Write-Host "`nPipeline stopped: $PipelineError" -ForegroundColor Red
    exit 1
}
