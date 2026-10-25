function Convert-ToFastStart {
    param(
        [string]$InputPath,
        [string]$OutputPath
    )

    $inStream = [System.IO.File]::OpenRead($InputPath)
    $reader = New-Object System.IO.BinaryReader($inStream)
    
    $ftypOffset = -1; $ftypSize = 0
    $moovOffset = -1; $moovSize = 0
    $mdatOffset = -1; $mdatSize = 0; $mdatHeaderSize = 0

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
            
            $headerSize = 8
            $atomSize = [uint64]$size

            if ($size -eq 1) {
                $size64Bytes = $reader.ReadBytes(8)
                if ($size64Bytes.Length -lt 8) { break }
                [Array]::Reverse($size64Bytes)
                $atomSize = [System.BitConverter]::ToUInt64($size64Bytes, 0)
                $headerSize = 16
            } elseif ($size -eq 0) {
                # Atom extends to the end of the file
                $atomSize = [uint64]($inStream.Length - $pos)
            }

            if ($atomSize -le 0 -or ($pos + $atomSize) -gt [uint64]$inStream.Length) {
                break
            }

            if ($type -eq 'ftyp') {
                $ftypOffset = $pos
                $ftypSize = $atomSize
            } elseif ($type -eq 'moov') {
                $moovOffset = $pos
                $moovSize = $atomSize
            } elseif ($type -eq 'mdat') {
                $mdatOffset = $pos
                $mdatSize = $atomSize
                $mdatHeaderSize = $headerSize
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

    # 1. Try ffmpeg if available (standard, handles all atom variations safely)
    $hasFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($hasFfmpeg) {
        $tempOut = "$OutputPath.faststart.tmp.mp4"
        try {
            & ffmpeg -y -v error -i $InputPath -c copy -movflags +faststart $tempOut
            if ($LASTEXITCODE -eq 0 -and (Test-Path $tempOut)) {
                Move-Item -Path $tempOut -Destination $OutputPath -Force
                Write-Output "[$OutputPath] FastStart applied via ffmpeg."
                return $true
            }
        } catch {
            # Fall back to script-based converter
        } finally {
            if (Test-Path $tempOut) { Remove-Item -Path $tempOut -Force -ErrorAction SilentlyContinue }
        }
    }

    # 2. Pure PowerShell fallback
    $inStream = [System.IO.File]::OpenRead($InputPath)
    $reader = New-Object System.IO.BinaryReader($inStream)

    try {
        $inStream.Position = $ftypOffset
        $ftypBytes = $reader.ReadBytes($ftypSize)

        $inStream.Position = $moovOffset
        $moovBytes = $reader.ReadBytes($moovSize)

        $newMdatOffset = [uint64]$ftypSize + [uint64]$moovSize
        $newMdatDataOffset = $newMdatOffset + [uint64]$mdatHeaderSize
        $oldMdatDataOffset = [uint64]$mdatOffset + [uint64]$mdatHeaderSize
        $delta = [int64]$newMdatDataOffset - [int64]$oldMdatDataOffset

        for ($i = 0; $i -lt $moovBytes.Length - 4; $i++) {
            if ($moovBytes[$i] -eq 115 -and $moovBytes[$i+1] -eq 116 -and $moovBytes[$i+2] -eq 99 -and $moovBytes[$i+3] -eq 111) {
                if ($i + 12 -le $moovBytes.Length) {
                    $cntBytes = $moovBytes[($i+8)..($i+11)]
                    [Array]::Reverse($cntBytes)
                    $count = [System.BitConverter]::ToUInt32($cntBytes, 0)
                    for ($c = 0; $c -lt $count; $c++) {
                        $idx = $i + 12 + $c * 4
                        if ($idx + 4 -gt $moovBytes.Length) { break }
                        $valBytes = $moovBytes[$idx..($idx+3)]
                        [Array]::Reverse($valBytes)
                        $val = [System.BitConverter]::ToUInt32($valBytes, 0)
                        $newVal = [uint32]([int64]$val + $delta)
                        $newBytes = [System.BitConverter]::GetBytes($newVal)
                        [Array]::Reverse($newBytes)
                        [Array]::Copy($newBytes, 0, $moovBytes, $idx, 4)
                    }
                }
            }
            if ($moovBytes[$i] -eq 99 -and $moovBytes[$i+1] -eq 111 -and $moovBytes[$i+2] -eq 54 -and $moovBytes[$i+3] -eq 52) {
                if ($i + 12 -le $moovBytes.Length) {
                    $cntBytes = $moovBytes[($i+8)..($i+11)]
                    [Array]::Reverse($cntBytes)
                    $count = [System.BitConverter]::ToUInt32($cntBytes, 0)
                    for ($c = 0; $c -lt $count; $c++) {
                        $idx = $i + 12 + $c * 8
                        if ($idx + 8 -gt $moovBytes.Length) { break }
                        $valBytes = $moovBytes[$idx..($idx+7)]
                        [Array]::Reverse($valBytes)
                        $val = [System.BitConverter]::ToUInt64($valBytes, 0)
                        $newVal = [uint64]([int64]$val + $delta)
                        $newBytes = [System.BitConverter]::GetBytes($newVal)
                        [Array]::Reverse($newBytes)
                        [Array]::Copy($newBytes, 0, $moovBytes, $idx, 8)
                    }
                }
            }
        }

        $tempOut = "$OutputPath.tmp"
        $outStream = [System.IO.File]::Create($tempOut)
        $writer = New-Object System.IO.BinaryWriter($outStream)

        try {
            $writer.Write($ftypBytes)
            $writer.Write($moovBytes)

            $inStream.Position = $mdatOffset
            $buffer = New-Object byte[] (8 * 1024 * 1024)
            $remaining = [int64]$mdatSize
            while ($remaining -gt 0) {
                $toRead = [int][Math]::Min($remaining, $buffer.Length)
                $readCount = $inStream.Read($buffer, 0, $toRead)
                if ($readCount -le 0) { break }
                $writer.Write($buffer, 0, $readCount)
                $remaining -= $readCount
            }
        } finally {
            $writer.Flush()
            $outStream.Close()
        }

        Move-Item -Path $tempOut -Destination $OutputPath -Force
        Write-Output "[$OutputPath] FastStart applied successfully."
        return $true
    } finally {
        $inStream.Close()
    }
}

$publicDir = 'frontend/public/videos'
$distDir = 'frontend/dist/videos'

if (Test-Path $publicDir) {
    # 1. Scan and apply FastStart to all MP4s in public/videos (including subfolders)
    Get-ChildItem -Path $publicDir -Filter "*.mp4" -Recurse | ForEach-Object {
        Convert-ToFastStart -InputPath $_.FullName -OutputPath $_.FullName
    }

    # 2. Sync to dist/videos automatically
    if (-not (Test-Path $distDir)) {
        New-Item -ItemType Directory -Path $distDir -Force | Out-Null
    }

    $publicRoot = (Resolve-Path $publicDir).Path
    Get-ChildItem -Path $publicDir -Filter "*.mp4" -Recurse | ForEach-Object {
        $relPath = $_.FullName.Substring($publicRoot.Length).TrimStart('\', '/')
        $targetPath = Join-Path $distDir $relPath
        $targetParent = Split-Path $targetPath -Parent
        if (-not (Test-Path $targetParent)) {
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
        }
        if ((-not (Test-Path $targetPath)) -or ($_.LastWriteTime -gt (Get-Item $targetPath).LastWriteTime)) {
            Copy-Item -Path $_.FullName -Destination $targetPath -Force
            Write-Output "Synced -> $targetPath"
        }
    }
}

# 3. Sync references
$publicRefDir = 'frontend/public/references'
$distRefDir = 'frontend/dist/references'
if (Test-Path $publicRefDir) {
    if (-not (Test-Path $distRefDir)) {
        New-Item -ItemType Directory -Path $distRefDir -Force | Out-Null
    }
    Copy-Item -Path "$publicRefDir/*" -Destination $distRefDir -Force
}

# 4. Sync figures
$publicFigDir = 'frontend/public/figures'
$distFigDir = 'frontend/dist/figures'
if (Test-Path $publicFigDir) {
    if (-not (Test-Path $distFigDir)) {
        New-Item -ItemType Directory -Path $distFigDir -Force | Out-Null
    }
    Copy-Item -Path "$publicFigDir/*" -Destination $distFigDir -Force
}
