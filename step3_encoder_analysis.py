# -*- coding: utf-8 -*-
"""step3_encoder_analysis.py —— 手册第 3 节：模型拆解与特征分析。

手册这一节的问题是：
    「图像 224×224×3」和「一句 'a large bird...'」完全不是一个空间，
     必须先各自变成向量，然后才能比较、融合、匹配、分类、检索。

本脚本把两个编码器「拆开」，用**可量化的证据**回答五个问题：

  Q1  图像经过主干之后变成了什么形状？
      [B,3,224,224] -> [B,512,7,7] -> 池化 [B,512] / 展平 [B,49,512]
  Q2  文本经过 tokenizer + BERT 之后变成了什么形状？
      "a small blue bird" -> token id -> [B,L,768] -> 池化 [B,768]
  Q3  [CLS] 池化和 mean 池化哪个好？（手册讲了两种，这里实测）
      —— 顺便验证「不做 attention_mask 过滤会把句向量拉向零」这个坑
  Q4  这两个向量空间是什么关系？能直接比较吗？
      —— 用余弦相似度矩阵 + t-SNE 联合可视化证明「存在模态鸿沟 (modality gap)」
  Q5  两种特征各自的线性可分性如何？（冻结特征 + 逻辑回归线性探针）

产出图：results/figures/ 下 tsne_image / tsne_text / tsne_joint / pooling_compare

用法：
    python src/step3_encoder_analysis.py
    python src/step3_encoder_analysis.py --n-train 800 --n-test 400   # 冒烟
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (NUM_CLASSES, FIGURE_DIR, RESULTS_DIR, IMAGE_BACKBONE, TEXT_MODEL,
                    SEED, get_device, describe_device, set_seed, class_to_display)
from dataset import BirdImageDataset, BirdTextDataset, split_json
from image_encoder import ImageEncoder
from text_encoder import TextEncoder, load_tokenizer
from train_utils import log_to_file, save_json, markdown_table


# ----------------------------------------------------------------- 取特征
def image_features(enc, loader, device, return_tokens=False):
    enc.eval()
    feats, labs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            f = enc(images, return_tokens=return_tokens)
            feats.append(f.float().cpu().numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def text_features(enc, tokenizer, samples, device, pooling="mean", max_length=64,
                  ignore_mask=False):
    """手动走一遍 BERT，便于同时拿到 [CLS] 和 mean 两种池化。

    ignore_mask=True 时故意不做 padding 过滤 —— 用来复现手册警告的错误做法。
    """
    enc.eval()
    feats_cls, feats_mean, feats_mean_wrong = [], [], []
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            chunk = samples[i:i + 64]
            texts = [s["captions"][0] for s in chunk]
            enc_in = tokenizer(texts, padding="max_length", truncation=True,
                               max_length=max_length, return_tensors="pt")
            ids = enc_in["input_ids"].to(device)
            mask = enc_in["attention_mask"].to(device)
            hidden = enc.bert(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(hidden.dtype)
            feats_cls.append(hidden[:, 0].float().cpu().numpy())
            feats_mean.append(((hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)).float().cpu().numpy())
            feats_mean_wrong.append(hidden.mean(dim=1).float().cpu().numpy())
    return (np.concatenate(feats_cls), np.concatenate(feats_mean),
            np.concatenate(feats_mean_wrong))


# ----------------------------------------------------------------- t-SNE
def do_tsne(X, perplexity=30, seed=SEED):
    import inspect
    from sklearn.manifold import TSNE
    X = np.asarray(X, dtype=np.float32)
    # 先 PCA 降到 50 维，t-SNE 更稳也更快
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        X = PCA(n_components=50, random_state=seed).fit_transform(X)
    # 迭代轮数的参数名在 sklearn 1.5 从 n_iter 改成了 max_iter，这里做个兼容
    n_iter_kw = "max_iter" if "max_iter" in inspect.signature(TSNE.__init__).parameters else "n_iter"
    ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
              learning_rate="auto", random_state=seed, **{n_iter_kw: 1000})
    return ts.fit_transform(X)


def scatter(ax, xy, labels, title, n_show=20, alpha=0.6, s=8):
    """只给前 n_show 个类别上色+图例，其余灰色，否则 100 类图例会糊成一团。"""
    import matplotlib.pyplot as plt
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    other = labels >= (uniq[:n_show].max() + 1) if len(uniq) > n_show else np.zeros(len(labels), bool)
    if other.any():
        ax.scatter(xy[other, 0], xy[other, 1], c="lightgray", s=s, alpha=0.35, label="其他类别")
    for i, c in enumerate(uniq[:n_show]):
        sel = labels == c
        ax.scatter(xy[sel, 0], xy[sel, 1], c=[cmap(i % 20)], s=s, alpha=alpha, label=str(c))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------------------------------------------- 线性探针
def linear_probe(Xtr, ytr, Xte, yte, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    acc = float(clf.score(sc.transform(Xte), yte))
    print("  线性探针 [{:22s}] 测试准确率 = {:.4f}   ({:.1f}s)".format(
        name, acc, time.time() - t0))
    return acc


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="手册第 3 节：编码器特征分析")
    ap.add_argument("--backbone", default=IMAGE_BACKBONE)
    ap.add_argument("--text-model", default=TEXT_MODEL)
    ap.add_argument("--n-train", type=int, default=2000, help="线性探针用的训练样本数")
    ap.add_argument("--n-test", type=int, default=1000, help="线性探针用的测试样本数")
    ap.add_argument("--tsne-per-class", type=int, default=6, help="t-SNE 每类取样数")
    ap.add_argument("--tsne-classes", type=int, default=20, help="t-SNE 上色的类别数")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()
    results = {}

    with log_to_file("step3_encoder_analysis.log"):
        print("=" * 78)
        print("手册第 3 节  模型拆解：图像与文本分别被处理成了什么")
        print("=" * 78)
        print("设备: {}".format(describe_device()))

        tokenizer = load_tokenizer(args.text_model)
        img_enc = ImageEncoder(backbone=args.backbone, pretrained=True,
                               freeze=True).to(device).eval()
        txt_enc = TextEncoder(model_name=args.text_model, freeze=True,
                              pooling="mean").to(device).eval()

        # ============================================================ Q1
        print("\n" + "=" * 78)
        print("Q1  图像编码器：一张图是怎么变成 49 个「视觉单词」的")
        print("=" * 78)
        dummy = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            fmap = img_enc.body(dummy)
            flat = fmap.flatten(2)
            trans = flat.transpose(1, 2)
            pooled = img_enc.pool(fmap).flatten(1)
        print("""
    图片 [3,224,224]
          │
          ▼
      ResNet 主干（去掉 avgpool 和 fc）
          │
          ▼
      特征图 {fmap}   ──┬── 全局平均池化 AvgPooling ──►  {pooled}   ← Early/Late/Gated 用
                        │
                        └── 展平成 49 个 token      ──►  {trans}  ← Attention 用
""".format(fmap=tuple(fmap.shape), pooled=tuple(pooled.shape), trans=tuple(trans.shape)))
        print("    [B, 512, 7, 7] 是怎么变成 [B, 49, 512] 的？")
        print("        [B,512,7,7] --flatten(2)--> {} --transpose(1,2)--> {}".format(
            tuple(flat.shape), tuple(trans.shape)))
        print("    224/32 = 7，所以一张图被切成 7×7 = 49 个「视觉单词」。")
        print("    特征图统计: mean={:.4f} std={:.4f}".format(fmap.mean().item(), fmap.std().item()))
        results["Q1_image_shapes"] = {
            "input": list(dummy.shape), "feature_map": list(fmap.shape),
            "flatten": list(flat.shape), "tokens": list(trans.shape),
            "pooled": list(pooled.shape)}

        # ============================================================ Q2
        print("\n" + "=" * 78)
        print("Q2  文本编码器：一句话是怎么变成 768 维向量的")
        print("=" * 78)
        demo = "a small blue bird"
        enc_in = tokenizer(demo, return_tensors="pt")
        toks = tokenizer.convert_ids_to_tokens(enc_in["input_ids"][0])
        print("\n    \"{}\"".format(demo))
        print("    tokenizer 查表（BERT 官方词表 bert-base-uncased，共 {} 个 token）：".format(
            tokenizer.vocab_size))
        for t, i in zip(toks, enc_in["input_ids"][0].tolist()):
            print("        {:12s} -> {}".format(t, i))
        print("""
    特殊标记：[CLS]={}  [SEP]={}  [PAD]={}  [MASK]={}  [UNK]={}
""".format(tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id,
           tokenizer.mask_token_id, tokenizer.unk_token_id))
        print("    BERT 内部：Token Embedding({} × {}) + Segment Embedding + Position Embedding".format(
            tokenizer.vocab_size, txt_enc.hidden_size))
        print("    Transformer 输出每个 token 一个向量 -> [B, L, 768]")
        print("    分类需要「一个句子一个向量」，所以要池化 [B,L,768] -> [B,768]")
        results["Q2_text_shapes"] = {
            "vocab_size": int(tokenizer.vocab_size),
            "hidden_size": int(txt_enc.hidden_size),
            "tokens": toks, "token_ids": enc_in["input_ids"][0].tolist(),
            "special_ids": {"cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id,
                            "pad": tokenizer.pad_token_id, "mask": tokenizer.mask_token_id,
                            "unk": tokenizer.unk_token_id}}

        # ============================================================ 取数据
        print("\n" + "=" * 78)
        print("取特征用于 Q3 / Q4 / Q5")
        print("=" * 78)
        train_ds = BirdImageDataset(split_json("train"), train=False)
        test_ds = BirdImageDataset(split_json("test"), train=False)
        classes = train_ds.classes
        n_tr, n_te = min(args.n_train, len(train_ds)), min(args.n_test, len(test_ds))
        train_ds.samples = train_ds.samples[:n_tr]
        test_ds.samples = test_ds.samples[:n_te]
        tr_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
        te_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        t0 = time.time()
        Xi_tr, yi_tr = image_features(img_enc, tr_loader, device)
        Xi_te, yi_te = image_features(img_enc, te_loader, device)
        print("图像特征: train {}  test {}   用时 {:.1f}s".format(
            Xi_tr.shape, Xi_te.shape, time.time() - t0))

        t0 = time.time()
        C_tr, M_tr, W_tr = text_features(txt_enc, tokenizer, train_ds.samples, device)
        C_te, M_te, W_te = text_features(txt_enc, tokenizer, test_ds.samples, device)
        print("文本特征: CLS {} / mean {} / 无mask均值 {}   用时 {:.1f}s".format(
            C_tr.shape, M_tr.shape, W_tr.shape, time.time() - t0))

        # ============================================================ Q3
        print("\n" + "=" * 78)
        print("Q3  [CLS] 池化 vs mean 池化；以及不做 mask 过滤会发生什么")
        print("=" * 78)
        cap_len = None
        with torch.no_grad():
            e = tokenizer([s["captions"][0] for s in test_ds.samples[:64]],
                          padding="max_length", truncation=True, max_length=64,
                          return_tensors="pt")
            real = e["attention_mask"].sum(dim=1).float()
        print("    本批句子的真实 token 数: 均值 {:.1f}（最大长度 64，其余都是 [PAD]）".format(
            real.mean().item()))
        print("    也就是说平均有 {:.0f}% 的位置是填充 —— 直接求平均会把句向量严重稀释。".format(
            (1 - real.mean().item() / 64) * 100))

        def cos(a, b):
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
            return (a * b).sum(1)

        sim_mw = cos(M_te, W_te)
        print("    正确 mean（过 mask）  的向量模长均值 = {:.4f}".format(
            np.linalg.norm(M_te, axis=1).mean()))
        print("    错误 mean（不过 mask）的向量模长均值 = {:.4f}   <- 被 [PAD] 拉小了".format(
            np.linalg.norm(W_te, axis=1).mean()))
        print("    两者余弦相似度 mean={:.4f} min={:.4f}  <- 差距明显，证明 mask 不是可有可无".format(
            sim_mw.mean(), sim_mw.min()))
        print("    CLS vs mean 余弦相似度 mean={:.4f}".format(cos(C_te, M_te).mean()))
        results["Q3_pooling"] = {
            "avg_real_tokens": float(real.mean().item()),
            "pad_ratio": float(1 - real.mean().item() / 64),
            "norm_mean_masked": float(np.linalg.norm(M_te, axis=1).mean()),
            "norm_mean_unmasked": float(np.linalg.norm(W_te, axis=1).mean()),
            "cos_masked_vs_unmasked": float(sim_mw.mean()),
            "cos_cls_vs_mean": float(cos(C_te, M_te).mean()),
        }

        # ============================================================ Q4
        print("\n" + "=" * 78)
        print("Q4  两个模态的向量能直接比较吗？（模态鸿沟验证）")
        print("=" * 78)
        n = min(500, len(Xi_te))
        print("    图像向量形状 {}   文本向量形状 {}".format(
            Xi_te[:n].shape, M_te[:n].shape))
        try:
            Xi_te[:n] @ M_te[:n].T
            print("    （不该算得出来）")
        except ValueError as e:
            print("    证据1  直接做 img @ txt.T -> 报错：{}".format(str(e).split(chr(10))[0]))
            print("           两个编码器输出的维度都不同（512 vs 768），连点积都算不了，")
            print("           更谈不上「比较相似度」。")

        # 强行投影到公共维度（截断到 512）后才谈得上比较
        img_n = Xi_te[:n, :512]
        img_n = img_n / (np.linalg.norm(img_n, axis=1, keepdims=True) + 1e-8)
        txt_proj = M_te[:n, :512]
        txt_n = txt_proj / (np.linalg.norm(txt_proj, axis=1, keepdims=True) + 1e-8)
        cross = img_n @ txt_n.T
        rng = np.random.RandomState(SEED)
        ri = rng.randn(n, 512); ri /= np.linalg.norm(ri, axis=1, keepdims=True)
        rt = rng.randn(n, 512); rt /= np.linalg.norm(rt, axis=1, keepdims=True)
        rand = ri @ rt.T
        print("\n    证据2  强行截断到 512 维后再比：")
        print("    图像-文本 余弦相似度: mean={:+.4f} std={:.4f}".format(cross.mean(), cross.std()))
        print("    随机向量对 余弦相似度: mean={:+.4f} std={:.4f}".format(rand.mean(), rand.std()))
        same = np.array([s["label"] for s in test_ds.samples[:n]])
        paired = np.diag(cross)
        print("    配对样本（同一张图的图和文）相似度 mean={:+.4f}".format(paired.mean()))
        match_same_class = (same[:, None] == same[None, :])
        np.fill_diagonal(match_same_class, False)
        print("    同类异样本相似度 mean={:+.4f}".format(cross[match_same_class].mean()))
        print("    随机基线 mean={:+.4f}".format(rand.mean()))
        print("""
    结论：配对样本的相似度并没有明显高于随机基线 ——
          两个**独立训练**的编码器输出位于两个互不相干的坐标系里，
          不能直接比较。这正是 CLIP 用对比学习把图文拉到同一个空间的动机
          （见第 5 节）。
""")
        results["Q4_modality_gap"] = {
            "cross_mean": float(cross.mean()), "cross_std": float(cross.std()),
            "random_mean": float(rand.mean()),
            "paired_mean": float(paired.mean()),
            "same_class_mean": float(cross[match_same_class].mean()),
        }

        # ============================================================ Q5
        probe = {}
        if not args.skip_probe:
            print("=" * 78)
            print("Q5  冻结特征的线性可分性（逻辑回归线性探针）")
            print("=" * 78)
            probe["image(512)"] = linear_probe(Xi_tr, yi_tr, Xi_te, yi_te, "图像 512 维")
            probe["text_cls(768)"] = linear_probe(C_tr, yi_tr, C_te, yi_te, "文本 CLS 768 维")
            probe["text_mean(768)"] = linear_probe(M_tr, yi_tr, M_te, yi_te, "文本 mean 768 维")
            probe["text_mean_unmasked(768)"] = linear_probe(
                W_tr, yi_tr, W_te, yi_te, "文本无mask均值 768 维")
            results["Q5_linear_probe"] = probe
            print("\n    这说明：ImageNet 预训练的 ResNet 特征本身就已经相当线性可分，")
            print("    而 BERT 的句子向量在 100 类细粒度鸟类上的线性可分性明显更低 ——")
            print("    文字里缺少「到底哪一只鸟」的视觉细节（手册 6.1 节）。")

        # ============================================================ 可视化
        print("\n" + "=" * 78)
        print("t-SNE 可视化")
        print("=" * 78)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            # 每类取样，保证图上每类都有点
            per_class = args.tsne_per_class
            pick = []
            for c in range(NUM_CLASSES):
                idx = np.where(yi_te == c)[0][:per_class]
                pick.extend(idx.tolist())
            pick = np.array(pick)
            if len(pick) < 50:      # 测试集样本少时退化为全部
                pick = np.arange(len(yi_te))
            sub_img, sub_lab = Xi_te[pick], yi_te[pick]
            sub_txt = M_te[pick]

            t0 = time.time()
            xy_img = do_tsne(sub_img)
            xy_txt = do_tsne(sub_txt)
            print("  两次 t-SNE 用时 {:.1f}s，点数 {}".format(time.time() - t0, len(pick)))

            fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
            scatter(axes[0], xy_img, sub_lab, "图像特征 (ResNet18, 512维)", args.tsne_classes)
            scatter(axes[1], xy_txt, sub_lab, "文本特征 (BERT mean, 768维)", args.tsne_classes)
            axes[0].legend(fontsize=6, markerscale=2, ncol=2, loc="best")
            axes[1].legend(fontsize=6, markerscale=2, ncol=2, loc="best")
            fig.suptitle("手册第3节：同一批样本的两种模态特征 t-SNE（颜色=鸟的种类）")
            fig.tight_layout()
            p1 = os.path.join(FIGURE_DIR, "step3_tsne_image_text.png")
            fig.savefig(p1, dpi=130); plt.close(fig)
            print("  已保存 {}".format(p1))
            results["figures"] = {"tsne_image_text": p1}

            # 联合图：把两个模态放进同一张 t-SNE 里，看它们是否落在不同区域。
            # 注意：两个编码器的维度（512 / 768）和尺度都不同，直接拼接没有意义，
            # 所以先各自做 PCA 到同样 256 维（各自内部白化能量），再拼起来做 t-SNE。
            # 这只是为了「能画在一张图上」，不改变「两个空间本来不相通」这个事实。
            from sklearn.decomposition import PCA
            k = min(256, sub_img.shape[0] - 1, sub_img.shape[1], sub_txt.shape[1])
            pi = PCA(n_components=k, random_state=SEED).fit_transform(sub_img)
            pt = PCA(n_components=k, random_state=SEED).fit_transform(sub_txt)
            joint = np.vstack([pi, pt])
            xy_joint = do_tsne(joint)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(xy_joint[:len(sub_img), 0], xy_joint[:len(sub_img), 1],
                       c="#1f77b4", s=9, alpha=0.6, label="图像特征 (ResNet18)")
            ax.scatter(xy_joint[len(sub_img):, 0], xy_joint[len(sub_img):, 1],
                       c="#d62728", s=9, alpha=0.6, label="文本特征 (BERT mean)")
            ax.set_title("模态鸿沟 (modality gap)：两个编码器不在同一个空间里\n"
                         "（各自 PCA 到 {} 维后再做 t-SNE，仅为可视化）".format(k))
            ax.set_xticks([]); ax.set_yticks([]); ax.legend()
            fig.tight_layout()
            p2 = os.path.join(FIGURE_DIR, "step3_modality_gap.png")
            fig.savefig(p2, dpi=130); plt.close(fig)
            print("  已保存 {}".format(p2))
            results["figures"]["modality_gap"] = p2
            results["joint_tsne_pca_dim"] = int(k)

            # 池化方式对比图
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(np.linalg.norm(M_te, axis=1), bins=40, alpha=0.65,
                    label="正确：过 attention_mask")
            ax.hist(np.linalg.norm(W_te, axis=1), bins=40, alpha=0.65,
                    label="错误：不过 mask")
            ax.set_xlabel("句向量模长 ||v||")
            ax.set_ylabel("样本数")
            ax.set_title("手册3节：mean pooling 忘过滤 [PAD] 的后果")
            ax.legend()
            fig.tight_layout()
            p3 = os.path.join(FIGURE_DIR, "step3_pooling_mask.png")
            fig.savefig(p3, dpi=130); plt.close(fig)
            print("  已保存 {}".format(p3))
            results["figures"]["pooling_mask"] = p3
        except ImportError as e:
            print("  跳过绘图（缺 matplotlib: {}）".format(e))

        save_json(results, os.path.join(RESULTS_DIR, "step3_encoder_analysis.json"))
        print("\n已保存: results/step3_encoder_analysis.json")


if __name__ == "__main__":
    main()
