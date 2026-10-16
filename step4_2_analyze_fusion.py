# -*- coding: utf-8 -*-
"""step4_2_analyze_fusion.py —— 第 4 节的「可解释性」分析。

四种融合方式里，有两种自带可读的内部状态，本脚本把它们挖出来：

  ③ 跨模态注意力：导出注意力权重，回答「模型在看哪个词」
        - 对每张图，把 49 个视觉位置对所有文本 token 的注意力求平均，
          看哪些词被看得最多（预期是颜色/体型的形容词）；
        - 画出「注意力集中在句子什么位置」的热力图。
  ④ 门控融合：导出逐样本的门控值 a
        - a→1 表示这个样本更信图像，a→0 表示更信文本；
        - 看它的分布，验证手册说的「对不可靠文本有抑制能力」。

另外做两个横向对比：
  * 六种配置的测试准确率柱状图（带标准差误差棒）；
  * 融合相对「仅图像」在**每个类别**上的提升/下降，找出融合到底帮了谁。

用法：
    python src/step4_2_analyze_fusion.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURE_DIR, NUM_CLASSES, class_to_display, TEXT_MODEL
from dataset import split_json, load_annotations
from fusion import FUSION_LABELS_CN
from train_utils import log_to_file, save_json, markdown_table


def load_npz(path):
    if not os.path.exists(path):
        return None
    return np.load(path)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="第 4 节融合结果的可解释性分析")
    ap.add_argument("--tag", default="frozen", help="对应 step4_fusion_ablation_<tag>.json")
    args = ap.parse_args()

    with log_to_file("step4_2_analyze_fusion_{}.log".format(args.tag)):
        print("=" * 78)
        print("第 4 节 分析：融合方式的可解释性与逐类别影响   (tag={})".format(args.tag))
        print("=" * 78)

        ab_path = os.path.join(RESULTS_DIR,
                               "step4_fusion_ablation_{}.json".format(args.tag))
        if not os.path.exists(ab_path):
            raise SystemExit("找不到 {}，请先运行 step4_fusion_ablation.py".format(ab_path))
        with open(ab_path, "r", encoding="utf-8") as f:
            ablation = json.load(f)
        summary = ablation["summary"]
        runs = ablation["runs"]

        details = load_npz(os.path.join(
            RESULTS_DIR, "step4_fusion_details_{}.npz".format(args.tag)))
        samples, classes = load_annotations(split_json("test"))

        # ---------------------------------------------------------- 总览
        print("\n--- 六种配置总览 ---")
        rows = []
        for s in summary:
            rows.append([FUSION_LABELS_CN.get(s["fusion"], s["fusion"]),
                         "{:,}".format(s["params"]),
                         "{:.4f} ± {:.4f}".format(s["test_accuracy_mean"], s["test_accuracy_std"]),
                         "{:.4f}".format(s["test_f1_mean"]),
                         "{:.4f}".format(s["top5_mean"])])
        print(markdown_table(["配置", "参数量", "测试准确率", "宏F1", "Top-5"], rows))

        base = next((s for s in summary if s["fusion"] == "image_only"), None)
        for s in summary:
            if base and s["fusion"] != "image_only":
                s["delta_vs_image"] = s["test_accuracy_mean"] - base["test_accuracy_mean"]
        if base:
            print("\n相对「仅图像」的提升（按测试准确率）：")
            rows = [[FUSION_LABELS_CN.get(s["fusion"], s["fusion"]),
                     "{:+.2f} pp".format(s.get("delta_vs_image", 0) * 100)]
                    for s in sorted(summary, key=lambda x: -x.get("delta_vs_image", 0))
                    if s["fusion"] != "image_only"]
            print(markdown_table(["配置", "Δ 准确率"], rows))

        # ---------------------------------------------------------- 柱状图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            names = [FUSION_LABELS_CN.get(s["fusion"], s["fusion"]).replace(" ", "\n", 1)
                     for s in summary]
            means = [s["test_accuracy_mean"] for s in summary]
            stds = [s["test_accuracy_std"] for s in summary]
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#bbbbbb"] * len(summary)
            best_i = int(np.argmax(means))
            colors[best_i] = "#d62728"
            bars = ax.bar(range(len(summary)), means, yerr=stds, capsize=4,
                          color=colors, edgecolor="black", linewidth=0.6)
            ax.set_xticks(range(len(summary)))
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel("测试准确率")
            ax.set_title("手册第4节：四种融合方式 vs 单模态基线（误差棒 = 多种子标准差）")
            ax.grid(axis="y", alpha=0.3)
            for b, m in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.005, "{:.3f}".format(m),
                        ha="center", fontsize=8)
            fig.tight_layout()
            p = os.path.join(FIGURE_DIR, "step4_{}_fusion_compare.png".format(args.tag))
            fig.savefig(p, dpi=130); plt.close(fig)
            print("\n已保存 {}".format(p))
        except ImportError:
            plt = None
            print("跳过绘图（缺 matplotlib）")

        # ---------------------------------------------------------- 门控分析
        gate_key = "gated_gate_values"
        if details is not None and gate_key in details:
            g = details[gate_key]
            print("\n--- ④ 门控融合：逐样本门控值 a 的分布 ---")
            print("  样本数 {}   均值 {:.4f}   标准差 {:.4f}".format(len(g), g.mean(), g.std()))
            qs = np.percentile(g, [5, 25, 50, 75, 95])
            print("  分位数 5/25/50/75/95: {:.3f} / {:.3f} / {:.3f} / {:.3f} / {:.3f}".format(*qs))
            print("  解读：a 越接近 1 表示这个样本越依赖图像，越接近 0 越依赖文本。")
            print("        分布越分散，说明门控确实在「逐样本」做决策，而不是退化成固定权重。")
            if plt is not None:
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                ax.hist(g, bins=40, color="#2ca02c", alpha=0.75, edgecolor="black", linewidth=0.4)
                ax.axvline(g.mean(), color="red", ls="--", label="均值 {:.3f}".format(g.mean()))
                ax.set_xlabel("门控值 a（越接近1越信图像）")
                ax.set_ylabel("样本数")
                ax.set_title("④ 门控融合：门控值分布（说明它在逐样本决策）")
                ax.legend()
                fig.tight_layout()
                p = os.path.join(FIGURE_DIR, "step4_{}_gate_distribution.png".format(args.tag))
                fig.savefig(p, dpi=130); plt.close(fig)
                print("  已保存 {}".format(p))

        # ---------------------------------------------------------- 注意力分析
        attn_key = "cross_attn_attn_weights"
        if details is not None and attn_key in details:
            attn = details[attn_key]           # [N, heads, 49, L]
            print("\n--- ③ 跨模态注意力：模型在看哪个词 ---")
            print("  注意力张量形状 [N, heads, 49, L] = {}".format(attn.shape))

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

            n_show = min(5, attn.shape[0])
            per_sample = []
            for i in range(n_show):
                cap = samples[i]["captions"][0]
                enc = tokenizer(cap, truncation=True, max_length=attn.shape[-1],
                                padding="max_length", return_tensors="pt")
                toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
                mask = enc["attention_mask"][0].numpy().astype(bool)
                # 对 heads 和 49 个视觉位置求平均 -> 每个词被关注的总量
                w = attn[i].mean(axis=(0, 1))            # [L]
                real = np.where(mask)[0]
                order = real[np.argsort(-w[real])]
                print("\n  样本 {}（真实类别: {}）".format(i, class_to_display(classes[samples[i]["label"]])))
                print("    描述: \"{}\"".format(cap))
                print("    被关注最多的词: " + ", ".join(
                    "{} ({:.3f})".format(toks[j], w[j]) for j in order[:6]))
                per_sample.append({
                    "index": i,
                    "caption": cap,
                    "true_class": classes[samples[i]["label"]],
                    "top_words": [{"token": toks[j], "weight": float(w[j])} for j in order[:8]],
                })

            # 颜色词 vs 其他词的注意力对比（教学结论）
            color_words = {"blue", "black", "white", "yellow", "red", "green", "brown",
                           "grey", "gray", "orange", "olive", "rufous"}
            body_words = {"wing", "wings", "beak", "bill", "tail", "head", "throat",
                          "breast", "belly", "back", "eye", "eyes", "crown", "legs"}
            n_all = min(attn.shape[0], 400)
            color_w, body_w, other_w = [], [], []
            for i in range(n_all):
                enc = tokenizer(samples[i]["captions"][0], truncation=True,
                                max_length=attn.shape[-1], padding="max_length",
                                return_tensors="pt")
                toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
                mask = enc["attention_mask"][0].numpy().astype(bool)
                w = attn[i].mean(axis=(0, 1))
                for j in np.where(mask)[0]:
                    t = toks[j].lower().replace("##", "")
                    if t in color_words:
                        color_w.append(w[j])
                    elif t in body_words:
                        body_w.append(w[j])
                    elif t not in ("[cls]", "[sep]", "[pad]"):
                        other_w.append(w[j])
            print("\n  按词性分组的平均注意力（{} 个样本统计）:".format(n_all))
            print("    颜色词   ({:5d} 个): {:.4f}".format(len(color_w), np.mean(color_w) if color_w else 0))
            print("    身体部位词({:5d} 个): {:.4f}".format(len(body_w), np.mean(body_w) if body_w else 0))
            print("    其他词   ({:5d} 个): {:.4f}".format(len(other_w), np.mean(other_w) if other_w else 0))
            print("  解读：颜色是 CUB 描述里最有区分度的属性之一，模型给的注意力权重")
            print("        是否真的偏向这些词，是「跨模态对齐是否学到东西」的直接证据。")

            if plt is not None:
                fig, ax = plt.subplots(figsize=(9, 4.5))
                labels = ["颜色词", "身体部位词", "其他词"]
                vals = [np.mean(color_w) if color_w else 0,
                        np.mean(body_w) if body_w else 0,
                        np.mean(other_w) if other_w else 0]
                ax.bar(labels, vals, color=["#ff7f0e", "#1f77b4", "#bbbbbb"],
                       edgecolor="black", linewidth=0.5)
                ax.set_ylabel("平均注意力权重")
                ax.set_title("③ 跨模态注意力：不同词性的被关注程度")
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.0005, "{:.4f}".format(v), ha="center", fontsize=9)
                fig.tight_layout()
                p = os.path.join(FIGURE_DIR, "step4_{}_attention_words.png".format(args.tag))
                fig.savefig(p, dpi=130); plt.close(fig)
                print("  已保存 {}".format(p))

                # 单个样本的注意力热力图
                i = 0
                cap = samples[i]["captions"][0]
                enc = tokenizer(cap, truncation=True, max_length=attn.shape[-1],
                                padding="max_length", return_tensors="pt")
                toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
                mask = enc["attention_mask"][0].numpy().astype(bool)
                real = np.where(mask)[0]
                hm = attn[i].mean(axis=0)[:, real]        # [49, L_real]
                fig, ax = plt.subplots(figsize=(min(12, 0.55 * len(real) + 3), 5.5))
                im = ax.imshow(hm, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(real)))
                ax.set_xticklabels([toks[j] for j in real], rotation=60, ha="right", fontsize=8)
                ax.set_ylabel("图像位置（49 个视觉 token）")
                ax.set_title("③ 跨模态注意力矩阵：图像的每个位置在看哪个词\n\"{}\"".format(cap))
                fig.colorbar(im, ax=ax, fraction=0.03)
                fig.tight_layout()
                p = os.path.join(FIGURE_DIR, "step4_{}_attention_heatmap.png".format(args.tag))
                fig.savefig(p, dpi=130); plt.close(fig)
                print("  已保存 {}".format(p))

            save_json({"per_sample": per_sample,
                       "group_means": {"color": float(np.mean(color_w)) if color_w else 0.0,
                                       "body": float(np.mean(body_w)) if body_w else 0.0,
                                       "other": float(np.mean(other_w)) if other_w else 0.0}},
                      os.path.join(RESULTS_DIR,
                                   "step4_attention_analysis_{}.json".format(args.tag)))

        # ---------------------------------------------------------- McNemar 配对检验
        # 「融合比仅图像低 0.26 个百分点」这句话本身说明不了问题 ——
        # 两个模型跑在**同一份**测试集上，必须做配对检验才知道差异是否显著。
        # McNemar 只看两个模型判断不一致的样本：
        #     b = 仅图像对、融合错 ；c = 仅图像错、融合对
        #     chi2 = (|b-c| - 1)^2 / (b + c)   ~ 卡方(1)
        if details is not None and "image_only_preds" in details and "image_only_labels" in details:
            from scipy.stats import chi2 as chi2_dist
            y = details["image_only_labels"]
            base_ok = (details["image_only_preds"] == y)
            print("\n--- McNemar 配对检验：融合 vs 仅图像（同一份测试集）---")
            print("  H0：两者错误率相同。|b-c| 越大越显著（p < 0.05 才敢说不同）")
            print("  {:30s} {:>6s} {:>6s} {:>9s} {:>8s}".format(
                "配置", "b", "c", "chi2", "p"))
            mcnemar_rows = []
            for name in [s["fusion"] for s in summary if s["fusion"] != "image_only"]:
                key = "{}_preds".format(name)
                if key not in details:
                    continue
                ok = (details[key] == y)
                b = int((base_ok & ~ok).sum())      # 仅图像对、融合错
                c = int((~base_ok & ok).sum())      # 仅图像错、融合对
                if b + c == 0:
                    stat, pval = 0.0, 1.0
                else:
                    stat = (abs(b - c) - 1) ** 2 / float(b + c)
                    pval = float(1 - chi2_dist.cdf(stat, 1))
                if pval < 0.05 and b > c:
                    verdict = "显著更差"
                elif pval < 0.05 and c > b:
                    verdict = "显著更好"
                else:
                    verdict = "无显著差异"
                print("  {:30s} {:>6d} {:>6d} {:>9.3f} {:>8.4f}   {}".format(
                    FUSION_LABELS_CN.get(name, name), b, c, stat, pval, verdict))
                mcnemar_rows.append({"fusion": name, "b": b, "c": c,
                                     "chi2": float(stat), "p": pval,
                                     "verdict": verdict})
            save_json({"tag": args.tag, "n_test": int(len(y)), "tests": mcnemar_rows},
                      os.path.join(RESULTS_DIR,
                                   "step4_mcnemar_{}.json".format(args.tag)))
            print("\n  注：b / c 是「两个模型判断不一致」的样本数。")
            print("      准确率只差零点几个百分点时，b 和 c 往往只有几十个，")
            print("      差异很容易落在噪声里 —— 这正是必须做配对检验的原因。")

        # ---------------------------------------------------------- 逐类别影响
        if base and details is not None:
            cm_base = details.get("image_only_confusion_matrix")
            # 只在**真正的多模态融合**里挑最好的，单模态基线不参与竞争
            multimodal = [s for s in summary
                          if s["fusion"] in ("early", "late", "cross_attn", "gated")]
            best_name = max(multimodal, key=lambda s: s["test_accuracy_mean"])["fusion"]
            cm_best = details.get("{}_confusion_matrix".format(best_name))
            if cm_base is not None and cm_best is not None:
                acc_base = np.diag(cm_base) / np.maximum(cm_base.sum(1), 1)
                acc_best = np.diag(cm_best) / np.maximum(cm_best.sum(1), 1)
                delta = acc_best - acc_base
                order = np.argsort(-delta)
                print("\n--- 融合（{}）相对「仅图像」帮助最大的 8 个类别 ---".format(best_name))
                rows = [[class_to_display(classes[i]), "{:+.3f}".format(delta[i])]
                        for i in order[:8]]
                print(markdown_table(["类别", "Δ 准确率"], rows))
                print("\n--- 反而变差的 8 个类别 ---")
                rows = [[class_to_display(classes[i]), "{:+.3f}".format(delta[i])]
                        for i in order[-8:][::-1]]
                print(markdown_table(["类别", "Δ 准确率"], rows))
                print("\n平均变化 {:+.4f}（正=融合更好）".format(delta.mean()))
                save_json({"best_fusion": best_name,
                           "delta_per_class": {classes[i]: float(delta[i])
                                               for i in range(len(classes))},
                           "delta_mean": float(delta.mean())},
                          os.path.join(RESULTS_DIR,
                                       "step4_per_class_delta_{}.json".format(args.tag)))

        print("\n分析完成。")


if __name__ == "__main__":
    main()
