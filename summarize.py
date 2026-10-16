# -*- coding: utf-8 -*-
"""summarize.py —— 把各步骤的 JSON 指标汇总成一张 Markdown 总表。

产出：results/summary_tables.md
报告（实验报告.md）里的所有数字都从这里取，避免手抄出错。

用法：
    python tools/summarize.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import RESULTS_DIR, FIGURE_DIR  # noqa: E402


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, n=4):
    try:
        return "{:.{}f}".format(float(x), n)
    except (TypeError, ValueError):
        return str(x)


def pct(x, n=2):
    try:
        return "{:+.{prec}f} pp".format(float(x) * 100, prec=n)
    except (TypeError, ValueError):
        return str(x)


def main():
    L = []
    A = L.append

    A("# 实验结果汇总表\n")
    A("> 本文件由 `tools/summarize.py` 自动生成，所有数字来自 `results/*.json`，请勿手改。\n")

    # ---------------------------------------------------------- 数据
    dr = load(os.path.join(RESULTS_DIR, "data_report.json"))
    if dr:
        rep, sizes = dr["report"], dr["split_sizes"]
        A("## 0. 数据概况\n")
        A(md_table(["项目", "数值"], [
            ["类别数", rep["num_classes"]],
            ["可用图文对", rep["num_pairs"]],
            ["有图无文（丢弃）", rep["num_orphan_image"]],
            ["caption 行数异常", rep["num_bad_caption"]],
            ["训练集", sizes["train"]],
            ["验证集", sizes["val"]],
            ["测试集", sizes["test"]],
            ["每类样本数", "{} ~ {}（均值 {:.1f}）".format(
                min(rep["per_class"].values()), max(rep["per_class"].values()),
                sum(rep["per_class"].values()) / len(rep["per_class"]))],
        ]))
        A("")

    # ---------------------------------------------------------- 2.1 图像
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "step2_1_image_only_*.json"))):
        if "smoke" in os.path.basename(p):
            continue
        d = load(p)
        if not d or "test" not in d:
            continue
        t = d["test"]
        rows.append([d["backbone"] + ("（冻结主干）" if d["freeze_backbone"] else "")
                     + ("（随机初始化）" if not d["pretrained"] else ""),
                     "{:,}".format(d["params_trainable"]),
                     fmt(d["best_val_accuracy"]),
                     fmt(t["accuracy"]), fmt(t["precision_macro"]),
                     fmt(t["recall_macro"]), fmt(t["f1_macro"]),
                     "{:.1f}s".format(d["avg_epoch_sec"])])
    if rows:
        A("## 2.1 Image-Only 训练（手册 2.1 节）\n")
        A(md_table(["backbone", "可训练参数", "最佳验证准确率", "测试准确率",
                    "宏精确率", "宏召回率", "宏F1", "每轮耗时"], rows))
        A("")

    # ---------------------------------------------------------- 2.2 文本
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "step2_2_text_only_*.json"))):
        if "smoke" in os.path.basename(p):
            continue
        d = load(p)
        if not d or "test" not in d:
            continue
        t = d["test"]
        rows.append(["{}{}{}".format(d["pooling"],
                                     "（冻结BERT）" if d["freeze_bert"] else "",
                                     "（随机初始化）" if not d["pretrained"] else ""),
                     "{:,}".format(d["params_trainable"]),
                     fmt(d["best_val_accuracy"]),
                     fmt(t["accuracy"]), fmt(t.get("top5_accuracy", 0)),
                     fmt(t["precision_macro"]), fmt(t["recall_macro"]),
                     fmt(t["f1_macro"]), "{:.1f}s".format(d["avg_epoch_sec"])])
    if rows:
        A("## 2.2 Text-Only 训练（手册 2.2 节，BERT）\n")
        A(md_table(["池化/配置", "可训练参数", "最佳验证准确率", "测试准确率", "Top-5",
                    "宏精确率", "宏召回率", "宏F1", "每轮耗时"], rows))
        A("")

    # ---------------------------------------------------------- 3 特征分析
    d3 = load(os.path.join(RESULTS_DIR, "step3_encoder_analysis.json"))
    if d3:
        A("## 3. 编码器特征分析（手册第 3 节）\n")
        if "Q1_image_shapes" in d3:
            q1 = d3["Q1_image_shapes"]
            A("**图像编码器形状链路**\n")
            A(md_table(["阶段", "张量形状"], [
                ["输入图像", q1["input"]],
                ["ResNet 主干输出（去掉 avgpool/fc）", q1["feature_map"]],
                ["flatten(2)", q1["flatten"]],
                ["transpose(1,2) → 视觉 token", q1["tokens"]],
                ["全局平均池化 → 全局向量", q1["pooled"]],
            ]))
            A("")
        if "Q2_text_shapes" in d3:
            q2 = d3["Q2_text_shapes"]
            A("**文本编码器**：词表 {}，隐藏维度 {}，[CLS]={} [SEP]={} [PAD]={}\n".format(
                q2["vocab_size"], q2["hidden_size"], q2["special_ids"]["cls"],
                q2["special_ids"]["sep"], q2["special_ids"]["pad"]))
        if "Q3_pooling" in d3:
            q3 = d3["Q3_pooling"]
            A("**池化方式对比**\n")
            A(md_table(["指标", "数值"], [
                ["平均真实 token 数", fmt(q3["avg_real_tokens"], 1)],
                ["[PAD] 占比", "{:.1%}".format(q3["pad_ratio"])],
                ["正确 mean（过 mask）向量模长", fmt(q3["norm_mean_masked"])],
                ["错误 mean（不过 mask）向量模长", fmt(q3["norm_mean_unmasked"])],
                ["两者余弦相似度", fmt(q3["cos_masked_vs_unmasked"])],
                ["[CLS] 与 mean 的余弦相似度", fmt(q3["cos_cls_vs_mean"])],
            ]))
            A("")
        if "Q4_modality_gap" in d3:
            q4 = d3["Q4_modality_gap"]
            A("**模态鸿沟验证**（两个独立编码器的输出能否直接比较）\n")
            A(md_table(["对象", "平均余弦相似度"], [
                ["图像-文本（全部配对）", fmt(q4["cross_mean"])],
                ["图像-文本（同一张图配对）", fmt(q4["paired_mean"])],
                ["图像-文本（同类不同样本）", fmt(q4["same_class_mean"])],
                ["随机向量基线", fmt(q4["random_mean"])],
            ]))
            A("")
        if "Q5_linear_probe" in d3:
            A("**冻结特征的线性探针准确率**（train 2000 / test 1000）\n")
            A(md_table(["特征", "测试准确率"],
                       [[k, fmt(v)] for k, v in d3["Q5_linear_probe"].items()]))
            A("")

    # ---------------------------------------------------------- 4 融合
    try:
        from fusion import FUSION_LABELS_CN
    except ImportError:
        FUSION_LABELS_CN = {}

    ab_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "step4_fusion_ablation_*.json")))
    if ab_files:
        A("## 4. 多模态融合消融（手册第 4 节）\n")
        compare = {}
        for p in ab_files:
            d = load(p)
            if not d:
                continue
            tag = d.get("tag", os.path.basename(p))
            label = d.get("label", tag)
            A("### 4.{} 编码器：{}\n".format(
                len(compare) + 1,
                {"frozen": "两个编码器都**冻结**（ImageNet ResNet18 + 预训练 BERT）",
                 "finetuned": "两个编码器都**在本任务上微调过**"}.get(tag, label)))
            rows = []
            for s in sorted(d["summary"], key=lambda x: -x["test_accuracy_mean"]):
                rows.append([FUSION_LABELS_CN.get(s["fusion"], s["fusion"]),
                             "{:,}".format(s["params"]),
                             fmt(s["val_accuracy_mean"]),
                             "{} ± {}".format(fmt(s["test_accuracy_mean"]),
                                              fmt(s["test_accuracy_std"])),
                             fmt(s["test_f1_mean"]), fmt(s["top5_mean"]),
                             "{}".format(s["n_seeds"])])
            A(md_table(["融合方式", "参数量", "验证准确率", "测试准确率 (mean±std)",
                        "宏F1", "Top-5", "种子数"], rows))
            A("")
            base = next((s for s in d["summary"] if s["fusion"] == "image_only"), None)
            mm = [s for s in d["summary"]
                  if s["fusion"] in ("early", "late", "cross_attn", "gated")]
            best_mm = max(mm, key=lambda s: s["test_accuracy_mean"]) if mm else None
            if base and best_mm:
                A("- 最佳**多模态**配置：**{}**，测试准确率 {}；"
                  "「仅图像」{}；Δ = **{}**\n".format(
                      FUSION_LABELS_CN.get(best_mm["fusion"], best_mm["fusion"]),
                      fmt(best_mm["test_accuracy_mean"]),
                      fmt(base["test_accuracy_mean"]),
                      pct(best_mm["test_accuracy_mean"] - base["test_accuracy_mean"])))
                A("- 各多模态配置相对「仅图像」：{}\n".format("；".join(
                    "{} {}".format(FUSION_LABELS_CN.get(s["fusion"], s["fusion"]),
                                   pct(s["test_accuracy_mean"] - base["test_accuracy_mean"]))
                    for s in sorted(mm, key=lambda x: -x["test_accuracy_mean"]))))
            late = [r for r in d["runs"] if r["fusion"] == "late" and "late_weight_text" in r]
            if late:
                ws = [r["late_weight_text"] for r in late]
                A("- 晚融合学到的文本权重 w = {}（图像权重 1-w = {}）\n".format(
                    fmt(statistics.mean(ws)), fmt(1 - statistics.mean(ws))))
            gated = [r for r in d["runs"] if r["fusion"] == "gated" and "gate_mean" in r]
            if gated:
                A("- 门控融合平均门控值 a = {}（越接近 1 越依赖图像）\n".format(
                    fmt(statistics.mean([r["gate_mean"] for r in gated]))))
            A("")
            compare[tag] = {s["fusion"]: s["test_accuracy_mean"] for s in d["summary"]}

        if len(compare) >= 2 and "frozen" in compare and "finetuned" in compare:
            A("### 4.{} 冻结 vs 微调编码器：融合增益对照\n".format(len(compare) + 1))
            fusions = [f for f in ("image_only", "text_only", "early", "late",
                                   "cross_attn", "gated")
                       if f in compare["frozen"] and f in compare["finetuned"]]
            rows = []
            for f in fusions:
                fr = compare["frozen"][f]
                ft = compare["finetuned"][f]
                rows.append([FUSION_LABELS_CN.get(f, f), fmt(fr), fmt(ft), pct(ft - fr)])
            A(md_table(["融合方式", "冻结编码器", "微调编码器", "Δ(微调 − 冻结)"], rows))
            A("")
            # 融合相对自身单模态基线的增量，最能说明「融合到底有没有用」
            rows = []
            for tag in ("frozen", "finetuned"):
                m = compare[tag]
                if "image_only" not in m:
                    continue
                for f in fusions:
                    if f in ("image_only", "text_only"):
                        continue
                    rows.append([tag, FUSION_LABELS_CN.get(f, f),
                                 fmt(m[f]), pct(m[f] - m["image_only"])])
            if rows:
                A("**融合相对「仅图像」的净增益（关键：正数才说明融合有用）**\n")
                A(md_table(["编码器", "融合方式", "测试准确率", "相对「仅图像」"], rows))
                A("")

    # ---------------------------------------------------------- McNemar 配对检验
    mc_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "step4_mcnemar_*.json")))
    if mc_files:
        try:
            from fusion import FUSION_LABELS_CN as _L
        except ImportError:
            _L = {}
        A("### 4.{} McNemar 配对检验：融合 vs「仅图像」\n".format(len(ab_files) + 2))
        A("> 两个模型跑在**同一份**测试集上，所以必须做配对检验。\n"
          "> b = 仅图像对、融合错；c = 仅图像错、融合对；"
          "χ² = (|b−c|−1)²/(b+c) ~ 卡方(1)。\n")
        for p in mc_files:
            d = load(p)
            if not d:
                continue
            A("**编码器 = {}**（测试集 {} 张）\n".format(d["tag"], d["n_test"]))
            rows = [[_L.get(t["fusion"], t["fusion"]), t["b"], t["c"],
                     fmt(t["chi2"], 3), fmt(t["p"], 4), t["verdict"]]
                    for t in d["tests"]]
            A(md_table(["配置", "b", "c", "χ²", "p 值", "结论"], rows))
            A("")

    # ---------------------------------------------------------- 注意力
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "step4_attention_analysis_*.json"))):
        tag = os.path.basename(p).replace("step4_attention_analysis_", "").replace(".json", "")
        d4a = load(p)
        if not d4a or "group_means" not in d4a:
            continue
        A("**跨模态注意力：不同词性的平均被关注程度（{}）**\n".format(tag))
        g = d4a["group_means"]
        A(md_table(["词性", "平均注意力权重"],
                   [["颜色词", fmt(g["color"])], ["身体部位词", fmt(g["body"])],
                    ["其他词", fmt(g["other"])]]))
        A("")
        if "per_sample" in d4a:
            A("被关注最多的词（前 3 个样本）：\n")
            rows = [[s["index"], s["caption"][:60],
                     ", ".join(w["token"] for w in s["top_words"][:5])]
                    for s in d4a["per_sample"][:3]]
            A(md_table(["样本", "描述（截断）", "Top-5 被关注的词"], rows))
            A("")

    # ---------------------------------------------------------- 逐类别
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "step4_per_class_delta_*.json"))):
        tag = os.path.basename(p).replace("step4_per_class_delta_", "").replace(".json", "")
        d4p = load(p)
        if not d4p:
            continue
        A("**融合相对「仅图像」的逐类别变化（{}，最佳多模态配置 = {}）**\n".format(
            tag, d4p["best_fusion"]))
        items = sorted(d4p["delta_per_class"].items(), key=lambda kv: -kv[1])
        A(md_table(["类别", "Δ 准确率"],
                   [[k, "{:+.3f}".format(v)] for k, v in items[:5]]))
        A("\n…（完整 100 类见 results/step4_per_class_delta_{}.json）\n".format(tag))
        A("平均变化 **{:+.4f}**\n".format(d4p["delta_mean"]))

    # ---------------------------------------------------------- 5 CLIP
    rows = []
    first = None
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "step5_clip_*.json"))):
        d = load(p)
        if not d:
            continue
        first = first or d
        zs = d["zero_shot"]
        best_t = d["zero_shot_best_template"]
        rows.append([d["model"], d["split"], "{:,}".format(d["num_samples"]),
                     fmt(zs[best_t]["accuracy"]), fmt(zs[best_t]["top5_accuracy"]),
                     fmt(d["retrieval"]["image_to_text_R@1"]),
                     fmt(d["retrieval"]["image_to_text_R@5"]),
                     fmt(d["retrieval"]["text_to_image_R@1"]),
                     fmt(d["retrieval"]["text_to_image_R@5"])])
    if rows:
        A("## 5. CLIP 零样本与图文检索（手册第 5 节）\n")
        A(md_table(["CLIP 模型", "划分", "样本数", "零样本 Top-1", "零样本 Top-5",
                    "i→t R@1", "i→t R@5", "t→i R@1", "t→i R@5"], rows))
        A("")
        if first:
            A("**不同提示词写法对零样本准确率的影响**\n")
            A(md_table(["提示词模板", "Top-1", "Top-5"],
                       [[k, fmt(v["accuracy"]), fmt(v["top5_accuracy"])]
                        for k, v in first["zero_shot"].items()]))
            A("")
            A("**「类别不是被输出，而是被比较出来」的现场证据**（第 1 张测试图）\n")
            ex = first["example_similarity"]
            A(md_table(["排名", "类别", "余弦相似度"],
                       [[i + 1, t["class"], fmt(t["similarity"], 4)]
                        for i, t in enumerate(ex["top5"])]))
            A("")

    # ---------------------------------------------------------- 6 一图看懂
    A("## 6. 结论速查\n")
    speed = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "step2_1_image_only_*.json"))):
        if "smoke" in p:
            continue
        d = load(p)
        if d and "test" in d:
            speed.append(("Image-Only " + d["backbone"], d["test"]["accuracy"]))
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "step2_2_text_only_*.json"))):
        if "smoke" in p:
            continue
        d = load(p)
        if d and "test" in d:
            speed.append(("Text-Only BERT(" + d["pooling"] + ")", d["test"]["accuracy"]))
    if d3 and "Q5_linear_probe" in d3:
        for k, v in d3["Q5_linear_probe"].items():
            speed.append(("线性探针 " + k, v))
    for tag, m in compare.items():
        for f, v in m.items():
            speed.append(("融合[{}/{}]".format(tag, f), v))
    if first:
        for k, v in first["zero_shot"].items():
            speed.append(("CLIP 零样本[" + k + "]", v["accuracy"]))
    if speed:
        A(md_table(["方法", "测试准确率"],
                   [[k, fmt(v)] for k, v in sorted(speed, key=lambda kv: -kv[1])]))
        A("")

    # ---------------------------------------------------------- 图表清单
    figs = sorted(glob.glob(os.path.join(FIGURE_DIR, "*.png")))
    if figs:
        A("## 7. 图表清单\n")
        A(md_table(["图", "文件"],
                   [[os.path.basename(f), "results/figures/" + os.path.basename(f)]
                    for f in figs]))
        A("")

    out = os.path.join(RESULTS_DIR, "summary_tables.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("已生成 {}".format(out))
    print("共 {} 行".format(len(L)))


if __name__ == "__main__":
    main()
