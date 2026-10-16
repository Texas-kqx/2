# 实验结果汇总表

> 本文件由 `tools/summarize.py` 自动生成，所有数字来自 `results/*.json`，请勿手改。

## 0. 数据概况

| 项目 | 数值 |
|---|---|
| 类别数 | 100 |
| 可用图文对 | 5864 |
| 有图无文（丢弃） | 7 |
| caption 行数异常 | 0 |
| 训练集 | 3515 |
| 验证集 | 1177 |
| 测试集 | 1172 |
| 每类样本数 | 41 ~ 60（均值 58.6） |

## 2.1 Image-Only 训练（手册 2.1 节）

| backbone | 可训练参数 | 最佳验证准确率 | 测试准确率 | 宏精确率 | 宏召回率 | 宏F1 | 每轮耗时 |
|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 2,351,972 | 0.7944 | 0.7850 | 0.7931 | 0.7862 | 0.7835 | 35.4s |
| resnet18 | 11,227,812 | 0.7927 | 0.7875 | 0.8034 | 0.7886 | 0.7901 | 30.9s |

## 2.2 Text-Only 训练（手册 2.2 节，BERT）

| 池化/配置 | 可训练参数 | 最佳验证准确率 | 测试准确率 | Top-5 | 宏精确率 | 宏召回率 | 宏F1 | 每轮耗时 |
|---|---|---|---|---|---|---|---|---|
| mean | 109,559,140 | 0.4121 | 0.4198 | 0.7244 | 0.4228 | 0.4204 | 0.4062 | 29.9s |

## 3. 编码器特征分析（手册第 3 节）

**图像编码器形状链路**

| 阶段 | 张量形状 |
|---|---|
| 输入图像 | [4, 3, 224, 224] |
| ResNet 主干输出（去掉 avgpool/fc） | [4, 512, 7, 7] |
| flatten(2) | [4, 512, 49] |
| transpose(1,2) → 视觉 token | [4, 49, 512] |
| 全局平均池化 → 全局向量 | [4, 512] |

**文本编码器**：词表 30522，隐藏维度 768，[CLS]=101 [SEP]=102 [PAD]=0

**池化方式对比**

| 指标 | 数值 |
|---|---|
| 平均真实 token 数 | 22.3 |
| [PAD] 占比 | 65.1% |
| 正确 mean（过 mask）向量模长 | 10.0553 |
| 错误 mean（不过 mask）向量模长 | 8.9676 |
| 两者余弦相似度 | 0.8731 |
| [CLS] 与 mean 的余弦相似度 | 0.4369 |

**模态鸿沟验证**（两个独立编码器的输出能否直接比较）

| 对象 | 平均余弦相似度 |
|---|---|
| 图像-文本（全部配对） | -0.0269 |
| 图像-文本（同一张图配对） | -0.0272 |
| 图像-文本（同类不同样本） | -0.0265 |
| 随机向量基线 | -0.0001 |

**冻结特征的线性探针准确率**（train 2000 / test 1000）

| 特征 | 测试准确率 |
|---|---|
| image(512) | 0.4970 |
| text_cls(768) | 0.1240 |
| text_mean(768) | 0.1790 |
| text_mean_unmasked(768) | 0.1770 |

## 4. 多模态融合消融（手册第 4 节）

### 4.1 编码器：两个编码器都**在本任务上微调过**

| 融合方式 | 参数量 | 验证准确率 | 测试准确率 (mean±std) | 宏F1 | Top-5 | 种子数 |
|---|---|---|---|---|---|---|
| 仅图像 | 157,028 | 0.7876 | 0.7850 ± 0.0007 | 0.7859 | 0.9448 | 3 |
| ① 早融合 (Concat) | 969,828 | 0.7896 | 0.7804 ± 0.0065 | 0.7793 | 0.9334 | 3 |
| ② 晚融合 (Late) | 379,593 | 0.7791 | 0.7782 ± 0.0014 | 0.7764 | 0.9306 | 3 |
| ④ 门控融合 (Gated) | 616,804 | 0.7774 | 0.7756 ± 0.0032 | 0.7761 | 0.9320 | 3 |
| ③ 跨模态注意力 (Cross-Attn) | 683,364 | 0.6890 | 0.6775 ± 0.0030 | 0.6774 | 0.9056 | 3 |
| 仅文本 | 222,564 | 0.4191 | 0.3899 ± 0.0049 | 0.3815 | 0.7062 | 3 |

- 最佳**多模态**配置：**① 早融合 (Concat)**，测试准确率 0.7804；「仅图像」0.7850；Δ = **-0.46 pp**

- 各多模态配置相对「仅图像」：① 早融合 (Concat) -0.46 pp；② 晚融合 (Late) -0.68 pp；④ 门控融合 (Gated) -0.94 pp；③ 跨模态注意力 (Cross-Attn) -10.75 pp

- 晚融合学到的文本权重 w = 0.3775（图像权重 1-w = 0.6225）

- 门控融合平均门控值 a = 0.9496（越接近 1 越依赖图像）


### 4.2 编码器：两个编码器都**冻结**（ImageNet ResNet18 + 预训练 BERT）

| 融合方式 | 参数量 | 验证准确率 | 测试准确率 (mean±std) | 宏F1 | Top-5 | 种子数 |
|---|---|---|---|---|---|---|
| 仅图像 | 157,028 | 0.7400 | 0.7315 ± 0.0040 | 0.7341 | 0.9514 | 3 |
| ② 晚融合 (Late) | 379,593 | 0.7307 | 0.7290 ± 0.0028 | 0.7295 | 0.9150 | 3 |
| ① 早融合 (Concat) | 969,828 | 0.7205 | 0.7207 ± 0.0036 | 0.7201 | 0.9465 | 3 |
| ④ 门控融合 (Gated) | 616,804 | 0.7057 | 0.7079 ± 0.0113 | 0.7084 | 0.9357 | 3 |
| ③ 跨模态注意力 (Cross-Attn) | 683,364 | 0.5336 | 0.5028 ± 0.0074 | 0.5005 | 0.8510 | 3 |
| 仅文本 | 222,564 | 0.1776 | 0.1658 ± 0.0026 | 0.1507 | 0.4215 | 3 |

- 最佳**多模态**配置：**② 晚融合 (Late)**，测试准确率 0.7290；「仅图像」0.7315；Δ = **-0.26 pp**

- 各多模态配置相对「仅图像」：② 晚融合 (Late) -0.26 pp；① 早融合 (Concat) -1.08 pp；④ 门控融合 (Gated) -2.36 pp；③ 跨模态注意力 (Cross-Attn) -22.87 pp

- 晚融合学到的文本权重 w = 0.3427（图像权重 1-w = 0.6573）

- 门控融合平均门控值 a = 0.9834（越接近 1 越依赖图像）


### 4.3 冻结 vs 微调编码器：融合增益对照

| 融合方式 | 冻结编码器 | 微调编码器 | Δ(微调 − 冻结) |
|---|---|---|---|
| 仅图像 | 0.7315 | 0.7850 | +5.35 pp |
| 仅文本 | 0.1658 | 0.3899 | +22.41 pp |
| ① 早融合 (Concat) | 0.7207 | 0.7804 | +5.97 pp |
| ② 晚融合 (Late) | 0.7290 | 0.7782 | +4.92 pp |
| ③ 跨模态注意力 (Cross-Attn) | 0.5028 | 0.6775 | +17.46 pp |
| ④ 门控融合 (Gated) | 0.7079 | 0.7756 | +6.77 pp |

**融合相对「仅图像」的净增益（关键：正数才说明融合有用）**

| 编码器 | 融合方式 | 测试准确率 | 相对「仅图像」 |
|---|---|---|---|
| frozen | ① 早融合 (Concat) | 0.7207 | -1.08 pp |
| frozen | ② 晚融合 (Late) | 0.7290 | -0.26 pp |
| frozen | ③ 跨模态注意力 (Cross-Attn) | 0.5028 | -22.87 pp |
| frozen | ④ 门控融合 (Gated) | 0.7079 | -2.36 pp |
| finetuned | ① 早融合 (Concat) | 0.7804 | -0.46 pp |
| finetuned | ② 晚融合 (Late) | 0.7782 | -0.68 pp |
| finetuned | ③ 跨模态注意力 (Cross-Attn) | 0.6775 | -10.75 pp |
| finetuned | ④ 门控融合 (Gated) | 0.7756 | -0.94 pp |

### 4.4 McNemar 配对检验：融合 vs「仅图像」

> 两个模型跑在**同一份**测试集上，所以必须做配对检验。
> b = 仅图像对、融合错；c = 仅图像错、融合对；χ² = (|b−c|−1)²/(b+c) ~ 卡方(1)。

**编码器 = finetuned**（测试集 1172 张）

| 配置 | b | c | χ² | p 值 | 结论 |
|---|---|---|---|---|---|
| 仅文本 | 524 | 55 | 378.280 | 0.0000 | 显著更差 |
| ① 早融合 (Concat) | 50 | 33 | 3.084 | 0.0790 | 无显著差异 |
| ② 晚融合 (Late) | 28 | 17 | 2.222 | 0.1360 | 无显著差异 |
| ③ 跨模态注意力 (Cross-Attn) | 181 | 59 | 61.004 | 0.0000 | 显著更差 |
| ④ 门控融合 (Gated) | 33 | 26 | 0.610 | 0.4347 | 无显著差异 |

**编码器 = frozen**（测试集 1172 张）

| 配置 | b | c | χ² | p 值 | 结论 |
|---|---|---|---|---|---|
| 仅文本 | 696 | 31 | 606.459 | 0.0000 | 显著更差 |
| ① 早融合 (Concat) | 67 | 54 | 1.190 | 0.2753 | 无显著差异 |
| ② 晚融合 (Late) | 42 | 35 | 0.468 | 0.4941 | 无显著差异 |
| ③ 跨模态注意力 (Cross-Attn) | 331 | 64 | 179.129 | 0.0000 | 显著更差 |
| ④ 门控融合 (Gated) | 79 | 55 | 3.948 | 0.0469 | 显著更差 |

**跨模态注意力：不同词性的平均被关注程度（finetuned）**

| 词性 | 平均注意力权重 |
|---|---|
| 颜色词 | 0.0569 |
| 身体部位词 | 0.0428 |
| 其他词 | 0.0344 |

被关注最多的词（前 3 个样本）：

| 样本 | 描述（截断） | Top-5 被关注的词 |
|---|---|---|
| 0 | the bird has head and beak proportional to its body and the  | [CLS], [SEP], ., bird, its |
| 1 | the bird has a long black bill that is somewhat curved. | [SEP], bird, [CLS], ., bill |
| 2 | the wings are brown, long and narrow, and have white marking | [CLS], is, ,, the, . |

**跨模态注意力：不同词性的平均被关注程度（frozen）**

| 词性 | 平均注意力权重 |
|---|---|
| 颜色词 | 0.1227 |
| 身体部位词 | 0.0290 |
| 其他词 | 0.0314 |

被关注最多的词（前 3 个样本）：

| 样本 | 描述（截断） | Top-5 被关注的词 |
|---|---|---|
| 0 | the bird has head and beak proportional to its body and the  | ., brown, color, the, head |
| 1 | the bird has a long black bill that is somewhat curved. | ., [SEP], long, curved, black |
| 2 | the wings are brown, long and narrow, and have white marking | [SEP], eye, bill, long, . |

**融合相对「仅图像」的逐类别变化（finetuned，最佳多模态配置 = early）**

| 类别 | Δ 准确率 |
|---|---|
| 023.Brandt_Cormorant | +0.250 |
| 017.Cardinal | +0.167 |
| 046.Gadwall | +0.167 |
| 019.Gray_Catbird | +0.083 |
| 031.Black_billed_Cuckoo | +0.083 |

…（完整 100 类见 results/step4_per_class_delta_finetuned.json）

平均变化 **-0.0154**

**融合相对「仅图像」的逐类别变化（frozen，最佳多模态配置 = late）**

| 类别 | Δ 准确率 |
|---|---|
| 071.Long_tailed_Jaeger | +0.250 |
| 095.Baltimore_Oriole | +0.250 |
| 022.Chuck_will_Widow | +0.182 |
| 011.Rusty_Blackbird | +0.167 |
| 005.Crested_Auklet | +0.111 |

…（完整 100 类见 results/step4_per_class_delta_frozen.json）

平均变化 **-0.0058**

## 5. CLIP 零样本与图文检索（手册第 5 节）

| CLIP 模型 | 划分 | 样本数 | 零样本 Top-1 | 零样本 Top-5 | i→t R@1 | i→t R@5 | t→i R@1 | t→i R@5 |
|---|---|---|---|---|---|---|---|---|
| openai/clip-vit-base-patch32 | test | 1,172 | 0.6203 | 0.9428 | 0.0341 | 0.1357 | 0.0161 | 0.0586 |

**不同提示词写法对零样本准确率的影响**

| 提示词模板 | Top-1 | Top-5 |
|---|---|---|
| 基础 | 0.5998 | 0.9309 |
| 加限定 | 0.6203 | 0.9428 |
| 加细节 | 0.6143 | 0.9343 |
| 多模板集成 | 0.6092 | 0.9352 |

**「类别不是被输出，而是被比较出来」的现场证据**（第 1 张测试图）

| 排名 | 类别 | 余弦相似度 |
|---|---|---|
| 1 | Black footed Albatross | 0.3385 |
| 2 | Sooty Albatross | 0.3231 |
| 3 | Pigeon Guillemot | 0.3230 |
| 4 | Western Gull | 0.3156 |
| 5 | Laysan Albatross | 0.3133 |

## 6. 结论速查

| 方法 | 测试准确率 |
|---|---|
| Image-Only resnet18 | 0.7875 |
| Image-Only mobilenet_v2 | 0.7850 |
| 融合[finetuned/image_only] | 0.7850 |
| 融合[finetuned/early] | 0.7804 |
| 融合[finetuned/late] | 0.7782 |
| 融合[finetuned/gated] | 0.7756 |
| 融合[frozen/image_only] | 0.7315 |
| 融合[frozen/late] | 0.7290 |
| 融合[frozen/early] | 0.7207 |
| 融合[frozen/gated] | 0.7079 |
| 融合[finetuned/cross_attn] | 0.6775 |
| CLIP 零样本[加限定] | 0.6203 |
| CLIP 零样本[加细节] | 0.6143 |
| CLIP 零样本[多模板集成] | 0.6092 |
| CLIP 零样本[基础] | 0.5998 |
| 融合[frozen/cross_attn] | 0.5028 |
| 线性探针 image(512) | 0.4970 |
| Text-Only BERT(mean) | 0.4198 |
| 融合[finetuned/text_only] | 0.3899 |
| 线性探针 text_mean(768) | 0.1790 |
| 线性探针 text_mean_unmasked(768) | 0.1770 |
| 融合[frozen/text_only] | 0.1658 |
| 线性探针 text_cls(768) | 0.1240 |

## 7. 图表清单

| 图 | 文件 |
|---|---|
| step2_1_image_only_mobilenet_v2_curve.png | results/figures/step2_1_image_only_mobilenet_v2_curve.png |
| step2_1_image_only_resnet18_curve.png | results/figures/step2_1_image_only_resnet18_curve.png |
| step2_1_mobilenet_v2_confusion.png | results/figures/step2_1_mobilenet_v2_confusion.png |
| step2_1_resnet18_confusion.png | results/figures/step2_1_resnet18_confusion.png |
| step2_2_bert_mean_confusion.png | results/figures/step2_2_bert_mean_confusion.png |
| step2_2_text_only_bert_mean_curve.png | results/figures/step2_2_text_only_bert_mean_curve.png |
| step3_modality_gap.png | results/figures/step3_modality_gap.png |
| step3_pooling_mask.png | results/figures/step3_pooling_mask.png |
| step3_tsne_image_text.png | results/figures/step3_tsne_image_text.png |
| step4_finetuned_attention_heatmap.png | results/figures/step4_finetuned_attention_heatmap.png |
| step4_finetuned_attention_words.png | results/figures/step4_finetuned_attention_words.png |
| step4_finetuned_cross_attn_confusion.png | results/figures/step4_finetuned_cross_attn_confusion.png |
| step4_finetuned_early_confusion.png | results/figures/step4_finetuned_early_confusion.png |
| step4_finetuned_fusion_compare.png | results/figures/step4_finetuned_fusion_compare.png |
| step4_finetuned_gate_distribution.png | results/figures/step4_finetuned_gate_distribution.png |
| step4_finetuned_gated_confusion.png | results/figures/step4_finetuned_gated_confusion.png |
| step4_finetuned_image_only_confusion.png | results/figures/step4_finetuned_image_only_confusion.png |
| step4_finetuned_late_confusion.png | results/figures/step4_finetuned_late_confusion.png |
| step4_finetuned_text_only_confusion.png | results/figures/step4_finetuned_text_only_confusion.png |
| step4_frozen_attention_heatmap.png | results/figures/step4_frozen_attention_heatmap.png |
| step4_frozen_attention_words.png | results/figures/step4_frozen_attention_words.png |
| step4_frozen_cross_attn_confusion.png | results/figures/step4_frozen_cross_attn_confusion.png |
| step4_frozen_early_confusion.png | results/figures/step4_frozen_early_confusion.png |
| step4_frozen_fusion_compare.png | results/figures/step4_frozen_fusion_compare.png |
| step4_frozen_gate_distribution.png | results/figures/step4_frozen_gate_distribution.png |
| step4_frozen_gated_confusion.png | results/figures/step4_frozen_gated_confusion.png |
| step4_frozen_image_only_confusion.png | results/figures/step4_frozen_image_only_confusion.png |
| step4_frozen_late_confusion.png | results/figures/step4_frozen_late_confusion.png |
| step4_frozen_text_only_confusion.png | results/figures/step4_frozen_text_only_confusion.png |
