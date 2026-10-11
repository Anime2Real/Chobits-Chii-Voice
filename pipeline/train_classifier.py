#!/usr/bin/env python3
"""用人工标注训练小叽分类器 (embedding 上的逻辑回归), 并更新标注资产.

训练集:
  正例 = human_labels.npz 中 chi + 旧 labeled_embeddings.npz 的 chi
  负例 = human not_chi + mixed + 旧 negative + 旧 mixed
  (bad=音质差的小叽、unsure 不参与训练)

产出:
  annotations/labeled_embeddings.npz  正负样本池并入人工标注 (供旧工具)
  annotations/chi_reference_v5.npy    全部确认小叽的均值参考向量
  annotations/chi_lr.pkl              {"model": LogisticRegression, "threshold": float}
  阈值按 5 折交叉验证: 精确率 >= PREC_TARGET 下最大化召回

用法: .venv/bin/python pipeline/train_classifier.py
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PREC_TARGET = 0.95


def main():
    h = np.load("annotations/human_labels.npz")
    lab = np.load("annotations/labeled_embeddings.npz")
    h_lab, h_emb = h["labels"], h["embs"]

    chi_new = h_emb[h_lab == "chi"]
    neg_new = h_emb[(h_lab == "not_chi") | (h_lab == "mixed")]
    chi_all = np.concatenate([lab["chi"], chi_new])
    neg_all = np.concatenate([lab["negative"], lab["mixed"], neg_new])

    # 幂等去重: 样本池可能已并过人工标注, 逐位相同的重复行只留一条
    X_all = np.concatenate([chi_all, neg_all])
    y_all = np.array([1] * len(chi_all) + [0] * len(neg_all))
    _, keep = np.unique(X_all, axis=0, return_index=True)
    keep = np.sort(keep)
    X_all, y_all = X_all[keep], y_all[keep]
    chi_all, neg_all = X_all[y_all == 1], X_all[y_all == 0]
    print(f"正例 {len(chi_all)} (人工 {len(chi_new)}), 负例 {len(neg_all)} (人工 {len(neg_new)})")

    # 更新标注资产
    np.savez("annotations/labeled_embeddings.npz", chi=chi_all, negative=neg_all,
             mixed=np.zeros((0, 192)))
    ref = chi_all.mean(axis=0)
    np.save("annotations/chi_reference_v5.npy", ref / np.linalg.norm(ref))

    X = np.concatenate([chi_all, neg_all])
    y = np.array([1] * len(chi_all) + [0] * len(neg_all))

    clf = LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), cv=5, scoring="f1",
                               class_weight="balanced", max_iter=5000)
    clf.fit(X, y)
    print(f"最优 C={clf.C_[0]:.4g}")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    # 精确率-召回曲线, 选阈值
    print("\n阈值  精确率  召回率   (交叉验证)")
    best = None
    for thr in np.arange(0.50, 0.98, 0.02):
        pred = prob >= thr
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        mark = ""
        if prec >= PREC_TARGET and (best is None or rec > best[1]):
            best = (thr, rec)
            mark = "  <-- 候选"
        print(f"{thr:.2f}  {prec:.3f}   {rec:.3f}{mark}")
    thr = round(float(best[0]), 2) if best else 0.90
    print(f"\n选定阈值 {thr} (目标精确率>={PREC_TARGET})")

    with open("annotations/chi_lr.pkl", "wb") as f:
        pickle.dump({"model": clf, "threshold": thr}, f)
    print("保存 annotations/chi_lr.pkl, annotations/labeled_embeddings.npz, annotations/chi_reference_v5.npy")


if __name__ == "__main__":
    main()
