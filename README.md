# 实验二：基于图像-文本联合表示的鸟类多模态分类

本仓库是《实验2：鸟类多模态分类实验手册》的**完整可运行实现 + 实验记录**。

实验路线：**单模态 → 双模态 → 跨模态对齐 → 多模态融合 → 消融与错误分析**

---

## 1. 结论速览

测试集准确率（融合类为 3 个随机种子的均值）：

| 方法 | 测试准确率 | 是否训练过 CUB |
|---|---|---|
| Image-Only ResNet18（微调） | **0.7875** | 是 |
| Image-Only MobileNetV2（微调） | 0.7850 | 是 |
| **CLIP 零样本**（加限定提示词） | **0.6203** | **否** |
| ① 早融合（微调编码器） | 0.7804 | 是 |
| ② 晚融合（微调编码器） | 0.7782 | 是 |
| ④ 门控融合（微调编码器） | 0.7756 | 是 |
| ③ 跨模态注意力（微调编码器） | 0.6775 | 是 |
| Image-Only ResNet18（冻结特征） | 0.7315 | 否（线性头） |
| ② 晚融合（冻结编码器） | 0.7290 | 是 |
| Text-Only BERT（微调） | 0.4198 | 是 |
| Text-Only BERT（冻结特征） | 0.1658 | 否（线性头） |
| 线性探针（冻结 ResNet 特征） | 0.4970 | 否 |
| 线性探针（冻结 BERT 特征） | 0.1790 | 否 |

**最重要的发现：四种融合方式全部没有超过「仅图像」单模态基线**（净增益
−0.26 ~ −22.87 个百分点）。这不是实验失败，而是本次实验最有价值的部分，
完整解释见 `实验报告.md` 第 6.4 节。摘要：

- 两个模态信息量严重不对等（图像 0.785 vs 文本 0.390）；
- 门控融合自己学到 `a ≈ 0.95~0.98`，即「几乎完全信图像」，但 sigmoid 压不到 0，
  残余噪声抵消了本来就不大的增益；
- 晚融合的 `w` 只有 1 个标量，无法逐类调节，学到的值偏大；
- 跨模态注意力是唯一不用全局池化图像特征的方式，且注意力退化到 `[CLS]/[SEP]`；
- **支持性趋势**：文本从 0.166 提升到 0.390 后，融合的差距从 −2.4pp 收窄到 −0.9pp。

---

## 2. 数据

原始数据（不在本仓库内，路径在 `src/config.py` 的 `DATA_ROOT` 配置）：

```
D:\cxdownload\CUB_200_subset\
├── images\<001.Black_footed_Albatross>\<同名>.jpg
└── text  \<001.Black_footed_Albatross>\<同名>.txt   ← 每张图 10 行英文描述
```

| 项目 | 数值 |
|---|---|
| 类别数 | 100 |
| 图片 | 5871 |
| 有同名 txt 的图文对 | **5864**（7 张图无描述，已丢弃） |
| 每类样本 | 41 ~ 60（均值 58.6） |
| 划分 (6:2:2, seed=42, 分层) | train 3515 / val 1177 / test 1172 |

`src/data_prep.py` 同时生成两套标注（同一份划分、两种格式）：

- `datasets/annotations.json` —— 手册 1.3 节的单文件格式
  （`meta` / `classes` / `class_prompts` / `splits`）
- `datasets/annotations_{train,val,test}.json` —— 手册附带的
  `02_singleModal_TextOnly.py` 可直接读取的 `{"annotations": [...]}` 格式

---

## 3. 目录结构

```
.
├── datasets/                         # 生成的标注 JSON
├── results/                          # 指标 JSON / 日志 / 图表 / 权重
│   ├── features/                     # 缓存的编码器特征（mm_frozen_*/mm_finetuned_*）
│   ├── figures/                      # t-SNE、混淆矩阵、融合对比、注意力热力图
│   ├── logs/                         # 每个脚本的完整 stdout
│   └── summary_tables.md             # 汇总总表（自动生成）
├── src/
│   ├── config.py                     # 全局配置（路径/种子/超参）
│   ├── data_prep.py                  # 0. 数据准备：体检 + 6:2:2 划分 + 标注生成
│   ├── dataset.py                    # 三个 Dataset：Image-Only / Text-Only / 图文联合
│   ├── image_encoder.py              # 手册给的图像编码器（原样沿用，未改动）
│   ├── text_encoder.py               # 手册给的文本编码器（原样沿用，未改动）
│   ├── fusion.py                     # ★ 四种融合模块 + 两个单模态基线
│   ├── train_utils.py                # 日志 / 指标 / 绘图 / 训练历史
│   ├── step2_1_image_only.py         # 2.1 Image-Only 训练 + 模型拆解
│   ├── step2_2_text_only.py          # 2.2 Text-Only BERT 微调
│   ├── step3_encoder_analysis.py     # 第 3 节 编码器特征分析（Q1~Q5）
│   ├── step4_0_extract_features.py   # 第 4 节前置：冻结/微调编码器特征缓存
│   ├── step4_fusion_ablation.py      # 第 4 节 融合消融（6 配置 × 3 种子）
│   ├── step4_2_analyze_fusion.py     # 第 4 节 可解释性分析（注意力 / 门控 / 逐类）
│   └── step5_clip_zeroshot.py        # 第 5 节 CLIP 零样本 + 图文检索
├── tools/
│   ├── fetch_wheels.ps1              # 分块并行下载 CUDA 版 torch 轮子
│   ├── run_all.ps1                   # 一键跑完所有实验
│   └── summarize.py                  # 汇总所有 JSON -> summary_tables.md
├── 原始文件/                          # 老师提供的原始文件（一字未改，存档备查）
│   ├── 02_singleModal_TextOnly.py    #   2.2 节的参考脚本
│   ├── 03_image_encoder(1).py        #   第 3 节的图像编码器
│   └── 03_text_encoder(2).py         #   第 3 节的文本编码器
├── requirements.txt
├── 实验2：鸟类多模态分类实验手册.md      # 原始实验手册（只读留档）
├── 实验报告.md                        # ★ 实验报告
└── README.md
```

---

## 4. 环境

Windows + Python **3.8.6**，NVIDIA RTX 4060 Laptop 8 GB。

> Python 3.8 决定了版本上限：torch ≤ 2.4.1、transformers ≤ 4.46.3
> （更高版本已放弃 3.8）。

```powershell
# 1) 建虚拟环境
python -m venv .venv3

# 2) 下载 CUDA 版 torch（2.3 GB）。官方源单线程只有 0.2~1.9 MB/s 且会卡死，
#    脚本用 SJTU 镜像开 12 线程分块并行 + 断点续传，实测 9.56 MB/s，244 秒下完。
powershell -ExecutionPolicy Bypass -File tools\fetch_wheels.ps1

# 3) 安装
.venv3\Scripts\python.exe -m pip install --no-cache-dir `
    .wheels\torch-2.4.1+cu121-cp38-cp38-win_amd64.whl `
    .wheels\torchvision-0.19.1+cu121-cp38-cp38-win_amd64.whl `
    -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv3\Scripts\python.exe -m pip install --no-cache-dir `
    transformers==4.46.3 scikit-learn matplotlib tqdm pyyaml pandas `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4) 验证
.venv3\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

预训练权重与 HuggingFace 模型缓存在仓库内（`.torch_cache` / `.hf_cache`），
不写系统盘；`HF_ENDPOINT` 默认设为 `https://hf-mirror.com`（`src/config.py`）。

---

## 5. 运行

一键跑完（首次约 1 小时，主要是 2.3 GB 的 torch 轮子下载与两个 backbone 的训练）：

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_all.ps1
```

冒烟测试（每步只跑极少轮数，2~3 分钟）：

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_all.ps1 -Smoke
```

单步运行：见 `实验报告.md` 第 9.1 节。

---

## 6. 与手册的差异说明

代码沿用手册的结构、命名与教学要点，只在「能跑起来 / 可复现 / 结论可靠」上做了
必要调整，每一处都在对应脚本的 docstring 里写明：

1. **`image_encoder.py` / `text_encoder.py` 原样沿用**手册给的两个文件，
   未做任何修改。
2. **`02_singleModal_TextOnly.py` 补全**为 `src/step2_2_text_only.py`：
   原脚本只有训练 + 测试，**没有验证集**，无法做模型选择。补齐了验证集评估、
   最佳权重保存、Top-5、混淆矩阵、训练历史落盘，并加了 `--pooling {mean,cls}`
   与判别式微调（主干 2e-5 / 分类头 1e-3）。
3. **数据划分固定**：手册要求 6:2:2 分层划分并生成 `annotations.json`，
   但附带脚本读的是 `annotations_train/val/test.json`（格式还不一样）。
   `data_prep.py` **同时产出两种格式**。
4. **手册 2.1 节原文是「请补全....」**，`step2_1_image_only.py` 是完整新写的。
5. **手册 2.2 节标题写「模型：CLIP」，但参考脚本用的是 BERT**。
   本实现按脚本走 BERT（这也是「单模态分类」的正确做法），
   把 CLIP 放到第 5 节做零样本与检索，两节各司其职、不重复。
6. **手册第 4 节只给了 `fusion.py` 的注释说明，没有实现**，
   `src/fusion.py` 是完整新写的，四种融合 + 两个单模态基线统一接口。
7. **手册第 3 节的 tokenizer 示例数值有误**：手册写 `blue → 2304`、`bird → 4742`，
   实测 `bert-base-uncased` 里是 **blue = 2630、bird = 4743**。
   手册想表达的机制（查表返回行号）完全正确，只是示例数值不对。
8. **融合消融加了「冻结特征缓存」这一步**（`step4_0`）：
   六个配置共用同一份冻结特征，差异只来自融合方式；同时让「多种子重复」
   变得可行。并额外跑了一遍**微调编码器**版本，用来验证
   「融合失败是否因为文本编码器太弱」。
9. **手册第 5 节的零样本 / 检索完全空白**，`step5_clip_zeroshot.py` 是新写的，
   并额外做了提示词敏感性对照。
