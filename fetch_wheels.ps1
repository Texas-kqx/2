# fetch_wheels.ps1 —— 分块并行 + 断点续传 下载 PyTorch CUDA 轮子
#
# 背景：官方源 download.pytorch.org 单连接只有 0.2~1.9 MB/s，
#       而 torch 的 CUDA 轮子有 2.3 GB，单线程要跑一两个小时。
#       SJTU 镜像支持 Range 请求，开 12 个连接并行可以稳定到 ~7 MB/s。
#
# 注意（踩过的坑）：
#   * curl 的 -C - (--continue-at) 与 -r (--range) **互斥**，
#     所以断点续传必须自己算剩余区间，不能交给 -C。
#   * 本机是 Windows PowerShell 5.1，脚本文件必须带 UTF-8 BOM，
#     否则中文注释会被按 GBK 解码而导致语法错误。
#
# 用法：powershell -ExecutionPolicy Bypass -File tools\fetch_wheels.ps1
# 产物：.wheels\torch-2.4.1+cu121-cp38-cp38-win_amd64.whl
#       .wheels\torchvision-0.19.1+cu121-cp38-cp38-win_amd64.whl

$ErrorActionPreference = 'Stop'
$dest = Join-Path (Split-Path $PSScriptRoot -Parent) '.wheels'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$BASE = 'https://mirror.sjtu.edu.cn/pytorch-wheels/cu121'
$THREADS = 12
$TARGETS = @(
    @{ name = 'torch-2.4.1+cu121-cp38-cp38-win_amd64.whl'
       file = 'torch-2.4.1%2Bcu121-cp38-cp38-win_amd64.whl'
       sha  = 'a48b991cd861266523cbed4705f89bef09669d5d2bbfa2524486156f74a222a8' },
    @{ name = 'torchvision-0.19.1+cu121-cp38-cp38-win_amd64.whl'
       file = 'torchvision-0.19.1%2Bcu121-cp38-cp38-win_amd64.whl'
       sha  = '7779b5dbe10994380d8baf5d7c754e6f57dac0f513009e003525f13a29f45491' }
)

function Get-FileLen($p) {
    if (Test-Path $p) { return [int64](Get-Item $p).Length }
    return [int64]0
}

function Get-RemoteSize($url) {
    # 用 HEAD 拿 Content-Length
    $h = & curl.exe -sIL $url
    $m = $h | Select-String -Pattern '^content-length:\s*(\d+)' | Select-Object -Last 1
    if ($m) { return [int64]$m.Matches[0].Groups[1].Value }
    return [int64]0
}

# 一个块的续传作业：把 [from,to] 下到 .cont 再追加到 part
$chunkJob = {
    param($u, $from, $to, $part, $cont)
    if ($from -gt $to) { return }
    & curl.exe -sL --retry 8 --retry-delay 2 --retry-all-errors `
        --speed-limit 5000 --speed-time 60 -r "$from-$to" -o $cont $u 2>$null
    if ((Test-Path $cont) -and ((Get-Item $cont).Length -gt 0)) {
        $outFs = [System.IO.File]::Open($part, [System.IO.FileMode]::Append,
                                        [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        $inFs = [System.IO.File]::OpenRead($cont)
        try { $inFs.CopyTo($outFs, 1MB) } finally { $inFs.Dispose(); $outFs.Dispose() }
        Remove-Item $cont -Force -ErrorAction SilentlyContinue
    }
}

function Download-Chunked($url, $outPath, $total) {
    $tmp = "$outPath.parts"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $chunk = [int64][math]::Ceiling($total / $THREADS)

    $jobs = @()
    for ($i = 0; $i -lt $THREADS; $i++) {
        $start = [int64]($i * $chunk)
        if ($start -ge $total) { continue }
        $end = [int64]([math]::Min($start + $chunk - 1, $total - 1))
        $jobs += [pscustomobject]@{
            idx = $i; start = $start; end = $end
            part = Join-Path $tmp ("part{0:D2}.bin" -f $i)
            cont = Join-Path $tmp ("part{0:D2}.cont" -f $i)
            expect = [int64]($end - $start + 1)
        }
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    for ($attempt = 1; $attempt -le 12; $attempt++) {
        $pending = @($jobs | Where-Object { (Get-FileLen $_.part) -lt $_.expect })
        if ($pending.Count -eq 0) { break }
        $doneMB = 0.0
        foreach ($j in $jobs) { $doneMB += (Get-FileLen $j.part) / 1MB }
        $elapsed = [math]::Max($sw.Elapsed.TotalSeconds, 0.001)
        Write-Host ("    第 {0,2} 轮：待补齐 {1}/{2} 块，已下 {3:N0}/{4:N0} MB，平均 {5:N2} MB/s" -f `
            $attempt, $pending.Count, $jobs.Count, $doneMB, ($total / 1MB), ($doneMB / $elapsed))

        $running = @()
        foreach ($j in $pending) {
            $have = Get-FileLen $j.part
            $from = $j.start + $have
            $running += Start-Job -ScriptBlock $chunkJob `
                -ArgumentList $url, $from, $j.end, $j.part, $j.cont
        }
        $running | Wait-Job -Timeout 1500 | Out-Null
        $running | Stop-Job -ErrorAction SilentlyContinue
        $running | Remove-Job -Force -ErrorAction SilentlyContinue
    }
    $sw.Stop()

    foreach ($j in $jobs) {
        $got = Get-FileLen $j.part
        if ($got -ne $j.expect) {
            throw ("块 {0} 不完整：期望 {1} 字节，实际 {2} 字节" -f $j.idx, $j.expect, $got)
        }
    }

    Write-Host "    拼接分块 -> $(Split-Path $outPath -Leaf)"
    $outFs = [System.IO.File]::Create($outPath)
    try {
        foreach ($j in ($jobs | Sort-Object idx)) {
            $inFs = [System.IO.File]::OpenRead($j.part)
            try { $inFs.CopyTo($outFs, 4MB) } finally { $inFs.Dispose() }
        }
    } finally { $outFs.Dispose() }
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

foreach ($t in $TARGETS) {
    $url = "$BASE/$($t.file)"
    $out = Join-Path $dest $t.name

    if (Test-Path $out) {
        $h = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
        if ($h -eq $t.sha) { Write-Host "[跳过] $($t.name) 已存在且校验通过"; continue }
        Remove-Item $out -Force
    }

    Write-Host "[下载] $($t.name)"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $size = Get-RemoteSize $url
    if ($size -le 0) { throw "无法获取 $url 的大小" }
    Write-Host ("    远端大小 {0:N1} MB，{1} 线程并行" -f ($size / 1MB), $THREADS)

    if ($size -lt 64MB) {
        & curl.exe -L --retry 10 --retry-all-errors -o $out $url
        if ($LASTEXITCODE -ne 0) { throw "curl 失败：$url" }
    } else {
        Download-Chunked $url $out $size
    }
    $sw.Stop()

    $len = Get-FileLen $out
    $elapsed = [math]::Max($sw.Elapsed.TotalSeconds, 0.001)
    $h = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
    $ok = ($h -eq $t.sha)
    Write-Host ("    用时 {0:N1}s  平均 {1:N2} MB/s  大小 {2:N1} MB" -f `
        $sw.Elapsed.TotalSeconds, ($len / 1MB / $elapsed), ($len / 1MB))
    if ($ok) {
        Write-Host "    SHA256 校验通过"
    } else {
        throw "$($t.name) 哈希校验失败：实际 $h"
    }
}

Write-Host ""
Write-Host "全部轮子下载完成，位于 $dest"
Get-ChildItem $dest | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } } | Format-Table -AutoSize
