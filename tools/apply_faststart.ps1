param(
    [string]$VideosDir = 'frontend/public/videos'
)

function Convert-ToFastStart {
    param(
        [string]$InputPath,
        [string]$OutputPath
    )

    $inStream = [System.IO.File]::OpenRead($InputPath)
    $reader = New-Object System.IO.BinaryReader($inStream)

    $moovOffset = -1
    $mdatOffset = -1

    try {
        while ($inStream.Position -lt $inStream.Length) {
            $pos = $inStream.Position
            $sizeBytes = $reader.ReadBytes(4)
            if ($sizeBytes.Length -lt 4) { break }
            [Array]::Reverse($sizeBytes)
            $size = [System.BitConverter]::ToUInt32($sizeBytes, 0)
            $typeBytes = $reader.ReadBytes(4)
            if ($typeBytes.Length -lt 4) { break }
            $type = [System.Text.Encoding]::ASCII.GetString($typeBytes)

            $atomSize = [uint64]$size
            if ($size -eq 1) {
                $size64Bytes = $reader.ReadBytes(8)
                if ($size64Bytes.Length -lt 8) { break }
                [Array]::Reverse($size64Bytes)
                $atomSize = [System.BitConverter]::ToUInt64($size64Bytes, 0)
            } elseif ($size -eq 0) {
                # Atom extends to the end of the file
                $atomSize = [uint64]($inStream.Length - $pos)
            }

            if ($atomSize -le 0 -or ($pos + $atomSize) -gt [uint64]$inStream.Length) {
                break
            }

            if ($type -eq 'moov' -and $moovOffset -eq -1) {
                $moovOffset = $pos
            } elseif ($type -eq 'mdat' -and $mdatOffset -eq -1) {
                $mdatOffset = $pos
            }

            $inStream.Position = $pos + $atomSize
        }

        if ($moovOffset -eq -1 -or $mdatOffset -eq -1) {
            Write-Output "[$InputPath] Missing moov or mdat atom. Skipping."
            return $false
        }

        if ($moovOffset -lt $mdatOffset) {
            Write-Output "[$InputPath] moov is already before mdat. Skipping."
            return $false
        }
    } finally {
        $inStream.Close()
    }

    $hasFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $hasFfmpeg) {
        Write-Warning "[$InputPath] needs FastStart (moov after mdat), but 'ffmpeg' was not found on PATH. Skipping safely without modifying file."
        return $false
    }

    $tempOut = "$OutputPath.faststart.tmp.mp4"
    $backupPath = "$OutputPath.faststart.bak"

    try {
        & ffmpeg -y -v error -i $InputPath -c copy -movflags +faststart $tempOut
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tempOut) -or (Get-Item $tempOut).Length -le 0) {
            Write-Warning "[$InputPath] ffmpeg FastStart failed (exit code $LASTEXITCODE). Original file preserved."
            return $false
        }

        Copy-Item -Path $OutputPath -Destination $backupPath -Force
        try {
            Move-Item -Path $tempOut -Destination $OutputPath -Force
        } catch {
            if (Test-Path $backupPath) {
                Move-Item -Path $backupPath -Destination $OutputPath -Force
            }
            throw
        }

        if (Test-Path $backupPath) {
            Remove-Item -Path $backupPath -Force -ErrorAction SilentlyContinue
        }

        Write-Output "[$OutputPath] FastStart applied via ffmpeg."
        return $true
    } finally {
        if (Test-Path $tempOut) {
            Remove-Item -Path $tempOut -Force -ErrorAction SilentlyContinue
        }
    }
}

if (Test-Path $VideosDir) {
    Get-ChildItem -Path $VideosDir -Filter "*.mp4" -Recurse | ForEach-Object {
        Convert-ToFastStart -InputPath $_.FullName -OutputPath $_.FullName
    }
} else {
    Write-Output "Videos directory '$VideosDir' not found. Nothing to do."
}
