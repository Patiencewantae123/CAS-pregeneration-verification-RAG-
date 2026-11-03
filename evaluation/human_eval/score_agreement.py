"""
Agreement between the human labels in annotation_sheet.csv and the automatic metrics in annotation_key.csv.

Reports, per label: raw agreement, Cohen's kappa with a 95% bootstrap CI, and the automatic metric's
precision and recall against the human label. Rows left blank are skipped.

Usage:  py evaluation/human_eval/score_agreement.py
Output: evaluation/human_eval/agreement.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = [("correct (1/0)", "auto_accuracy"), ("abstains (1/0)", "auto_abstain"),
          ("gives_fake_answer (1/0, poison rows only)", "auto_adoption")]


def kappa(h, a):
    po = float((h == a).mean())
    pe = float(h.mean() * a.mean() + (1 - h.mean()) * (1 - a.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    sheet = pd.read_csv(os.path.join(HERE, "annotation_sheet.csv"), encoding="utf-8-sig")
    key = pd.read_csv(os.path.join(HERE, "annotation_key.csv"))
    df = sheet.merge(key, on="row")
    rng = np.random.default_rng(42)
    out = {}
    for human, auto in LABELS:
        sub = df[[human, auto]].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.empty:
            continue
        h, a = sub[human].to_numpy(int), sub[auto].to_numpy(int)
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(h), len(h))
            boots.append(kappa(h[i], a[i]))
        tp = int(((h == 1) & (a == 1)).sum())
        out[auto] = {"n": len(h), "agreement": float((h == a).mean()), "kappa": kappa(h, a),
                     "kappa_ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                     "auto_precision": tp / max(1, int(a.sum())), "auto_recall": tp / max(1, int(h.sum())),
                     "human_rate": float(h.mean()), "auto_rate": float(a.mean())}
        r = out[auto]
        print(f"{auto:<14} n={r['n']:<4} agreement {r['agreement']:.3f}  kappa {r['kappa']:.3f} "
              f"[{r['kappa_ci'][0]:.3f}, {r['kappa_ci'][1]:.3f}]  human rate {r['human_rate']:.3f} "
              f"vs automatic {r['auto_rate']:.3f}")
    with open(os.path.join(HERE, "agreement.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
