# run_all.ps1 —— 一键跑完手册第 0~5 节的全部实验
#
# 用法（必须绕过执行策略）：
#     powershell -ExecutionPolicy Bypass -File tools\run_all.ps1
# 或者只跑某几步：
#     powershell -ExecutionPolicy Bypass -File tools\run_all.ps1 -Only 2.1,4
#
# 每一步的 stdout 都会同时写到 results\logs\*.log。

param(
    [string[]]$Only = @(),        # 例：-Only 2.1,2.2,4 ；留空表示全跑
    [switch]$SkipData,            # 跳过数据准备
    [switch]$Smoke                # 冒烟模式：每步只跑很少的轮数/样本
)

# 注意：这里必须是 'Continue'，不能是 'Stop'。
# Windows PowerShell 5.1 在 $ErrorActionPreference='Stop' 下，会把**原生命令写到
# stderr 的任何一行**（例如 torch 输出的一条 UserWarning）当成终止性错误 ——
# 结果是 python 明明正常跑完，脚本却在这里中断。所以改成 'Continue'，
# 一律用 $LASTEXITCODE 显式判断成功与否。
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv3\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "找不到虚拟环境: $py" }

function Want($tag) {
    if ($Only.Count -eq 0) { return $true }
    return ($Only -contains $tag)
}

function Step($tag, $title, $file, [string[]]$extra = @()) {
    if (-not (Want $tag)) { return }
    Write-Host ''
    Write-Host ('=' * 78)
    Write-Host "[$tag] $title"
    Write-Host ('=' * 78)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $pyargs = @("$root\src\$file") + $extra
    & $py @pyargs
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "[$tag] $title 失败，退出码 $code" }
    $sw.Stop()
    Write-Host ("[$tag] 完成，用时 {0:N1} 分钟" -f $sw.Elapsed.TotalMinutes)
}

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# ---------------------------------------------------------------- 0 数据准备
if (-not $SkipData) {
    Write-Host ''
    Write-Host ('=' * 78)
    Write-Host "[0] 数据准备：6:2:2 分层划分 + annotations JSON"
    Write-Host ('=' * 78)
    & $py "$root\src\data_prep.py"
    if ($LASTEXITCODE -ne 0) { throw "数据准备失败，退出码 $LASTEXITCODE" }
}

# ---------------------------------------------------------------- 2 单模态
if ($Smoke) {
    Step '2.1' 'Image-Only（冒烟：3 轮）' 'step2_1_image_only.py' @('--epochs', '3', '--tag', 'smoke')
    Step '2.2' 'Text-Only BERT（冒烟：1 轮）' 'step2_2_text_only.py' @('--epochs', '1', '--tag', 'smoke')
} else {
    Step '2.1' 'Image-Only: ResNet18' 'step2_1_image_only.py' @('--backbone', 'resnet18')
    Step '2.1' 'Image-Only: MobileNetV2' 'step2_1_image_only.py' @('--backbone', 'mobilenet_v2')
    Step '2.2' 'Text-Only: BERT-base 微调' 'step2_2_text_only.py'
}

# ---------------------------------------------------------------- 3 编码器拆解
Step '3' '模型拆解与特征分析' 'step3_encoder_analysis.py'

# ---------------------------------------------------------------- 4 多模态融合
# 跑两遍：一遍用冻结的编码器，一遍用 2.1 / 2.2 训练好的微调编码器。
# 这样才回答得了「融合没用是不是因为文本编码器太弱」。
Step '4' '缓存编码器特征（冻结）' 'step4_0_extract_features.py' @('--tag', 'frozen')
Step '4' '四种融合消融（冻结编码器）' 'step4_fusion_ablation.py' `
    @('--tag', 'frozen', '--label', '冻结编码器', '--eval-every', '1')
Step '4' '缓存编码器特征（微调）' 'step4_0_extract_features.py' `
    @('--tag', 'finetuned',
      '--image-ckpt', "$root\results\best_image_resnet18.pth",
      '--text-ckpt', "$root\results\best_text_bert_mean.pth")
Step '4' '四种融合消融（微调编码器）' 'step4_fusion_ablation.py' `
    @('--tag', 'finetuned', '--label', '微调编码器', '--eval-every', '1')
Step '4' '融合可解释性分析（冻结）' 'step4_2_analyze_fusion.py' @('--tag', 'frozen')
Step '4' '融合可解释性分析（微调）' 'step4_2_analyze_fusion.py' @('--tag', 'finetuned')

# ---------------------------------------------------------------- 5 CLIP
Step '5' 'CLIP 零样本分类与图文检索' 'step5_clip_zeroshot.py'

# ---------------------------------------------------------------- 汇总
Write-Host ''
Write-Host ('=' * 78)
Write-Host "汇总所有指标 -> results\summary_tables.md"
Write-Host ('=' * 78)
& $py "$root\tools\summarize.py"

Write-Host ''
Write-Host '全部实验完成。'
Write-Host "  指标 JSON : $root\results"
Write-Host "  图表      : $root\results\figures"
Write-Host "  日志      : $root\results\logs"
Write-Host "  汇总表    : $root\results\summary_tables.md"
