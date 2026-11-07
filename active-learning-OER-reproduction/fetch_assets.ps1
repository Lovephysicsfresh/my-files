# 下载作者仓库里的大文件（图 PDB 和 checkpoint）
#
# 用法（在项目根目录）：
#     powershell -ExecutionPolicy Bypass -File fetch_assets.ps1
#
# 下载内容：
#     OER2022/initial_train/graph/*.pdb        600 个（约 23 MB）
#     OER2022/final_train/graph/*.pdb          800 个（约 29 MB）
#     OER2022/initial_train/checkpoint/*       50 个（约 82 MB，只有 full 模型必须）
#
# 想只下必需的最小集合，把 $WhatIf 改成 "minimal"

param(
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"
$Repo = "JunseokMoon/OER2022"
$Branch = "main"
$Root = $PSScriptRoot

function Get-RepoListing {
    param([string]$Path)
    $url = "https://api.github.com/repos/$Repo/contents/$Path`?ref=$Branch"
    $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 120
    return $resp.Content | ConvertFrom-Json
}

function Save-File {
    param([string]$Url, [string]$Dest)
    if (Test-Path -LiteralPath $Dest) { return }
    $dir = Split-Path -Parent $Dest
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    for ($i = 1; $i -le 4; $i++) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Dest -TimeoutSec 180
            return
        } catch {
            if ($i -eq 4) { throw }
            Start-Sleep -Seconds 3
        }
    }
}

function Fetch-Dir {
    param([string]$Path)
    $items = Get-RepoListing -Path $Path
    $n = 0
    foreach ($item in $items) {
        if ($item.type -ne "file") { continue }
        $dest = Join-Path $Root (Join-Path "OER2022" ($Path -replace "^OER2022/", "") | Join-Path -ChildPath $item.name)
        $dest = Join-Path $Root ("OER2022/" + ($Path -replace "^OER2022/", "") + "/" + $item.name)
        Save-File -Url $item.download_url -Dest $dest
        $n++
        if ($n % 50 -eq 0) { Write-Host "  $Path : $n files" }
    }
    Write-Host "$Path done ($n files)"
}

Write-Host "Downloading author assets from $Repo ..."

if ($Mode -eq "all" -or $Mode -eq "minimal") {
    Fetch-Dir "initial_train/graph"
    Fetch-Dir "final_train/graph"
}

if ($Mode -eq "all") {
    Fetch-Dir "initial_train/checkpoint"
} elseif ($Mode -eq "minimal") {
    Write-Host "minimal 模式：checkpoint 跳过（只有 step_04/05/17 需要）"
}

Write-Host ""
Write-Host "完成。验证："
Get-ChildItem -Path (Join-Path $Root "OER2022") -Recurse -File -Filter *.pdb -ErrorAction SilentlyContinue |
    Measure-Object |
    ForEach-Object { Write-Host ("  PDB 文件数: " + $_.Count) }
