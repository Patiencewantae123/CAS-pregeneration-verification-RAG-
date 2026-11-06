"""
Build a blind annotation sheet for validating the automatic answer metrics (accuracy, abstention, wrong answer,
poison adoption) on the responses behind the paper's confirmatory results.

The annotator sees the question, the reference answer(s), the fake answer (poison rows only) and the model's answer,
never the method, the study or the automatic score. The automatic scores go to a separate key file that
score_agreement.py joins afterwards.

Sample: 150 responses, stratified by source and by automatic outcome, so that every label has enough cases.

  study 6, HotpotQA   40   clean; equal shares of automatic correct / wrong / abstain where available
  study 6, MuSiQue    20   clean; as above
  study 5, clean      20   clean; as above
  study 5, poison     40   copy and written attacks; half with automatic adoption, half without
  study 2, poison     30   the stronger attack of the CAS v1 studies; as above

Duplicates (the same question and answer text) are labelled once. "CAS v2 + fallback" rows that did not use the
fallback repeat CAS v2's answer and are left out.

Usage:  py evaluation/human_eval/make_sheet.py
Output: evaluation/human_eval/annotation_sheet.csv (to fill in), annotation_key.csv (do not open before labelling)
"""
import json
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from run_evaluation import DATA, OUT_RESULTS, SEED, load  # noqa: E402

N = {"study 6, HotpotQA": 40, "study 6, MuSiQue": 20, "study 5, clean": 20, "study 5, poison": 40,
     "study 2, poison": 30}
SHOWN = ["row", "question", "reference_answer", "fake_answer", "model_answer"]
LABELS = ["correct (1/0)", "abstains (1/0)", "gives_fake_answer (1/0, poison rows only)"]


def outcome(r):
    return "correct" if r["accuracy"] == 1 else "abstain" if r["abstain"] == 1 else "wrong"


def stratified(rng, df, n, stratum):
    """n rows, as equal as possible across the strata; a stratum short of rows passes its share to the others."""
    groups = {k: list(g.index) for k, g in df.groupby(df.apply(stratum, axis=1))}
    for g in groups.values():
        rng.shuffle(g)
    take, left = {k: 0 for k in groups}, n
    while left:
        open_ = sorted(k for k in groups if take[k] < len(groups[k]))
        if not open_:
            break
        for k in open_:
            if left and take[k] < len(groups[k]):
                take[k] += 1
                left -= 1
    return df.loc[[i for k in sorted(groups) for i in groups[k][:take[k]]]]


def candidates():
    """Every scored response per source, with question, references and automatic labels."""
    out = {}
    s6 = json.load(open(os.path.join(DATA, "study6.json"), encoding="utf-8"))
    r6 = pd.read_csv(os.path.join(OUT_RESULTS, "study6_rows.csv"))
    r6 = r6[~((r6.method == "CAS v2 + fallback") & (r6.fallback_used == 0))]
    for ds, label in (("hotpotqa", "study 6, HotpotQA"), ("musique", "study 6, MuSiQue")):
        items = {it["id"]: it for it in s6[ds]}
        sub = r6[r6.dataset == ds].copy()
        sub["question"] = sub.id.map(lambda i: items[i]["question"])
        sub["reference_answer"] = sub.id.map(lambda i: " / ".join(items[i]["answers"]))
        sub["fake_answer"], sub["adoption"] = "", ""
        out[label] = sub

    s5 = json.load(open(os.path.join(DATA, "hotpotqa_study5.json"), encoding="utf-8"))
    r5 = pd.read_csv(os.path.join(OUT_RESULTS, "study5_rows.csv"))
    for part, label in (("clean", "study 5, clean"), ("poison", "study 5, poison")):
        items = {it["id"]: it for it in s5[part]}
        sub = r5[r5.set == part].copy()
        sub["question"] = sub.id.map(lambda i: items[i]["question"])
        sub["reference_answer"] = sub.id.map(lambda i: items[i]["answer"])
        sub["fake_answer"] = sub.id.map(lambda i: items[i].get("substitute", "")) if part == "poison" else ""
        if part == "clean":
            sub["adoption"] = ""
        sub["method"] = sub.method + (" (" + sub.attack + ", " + sub.condition + ")" if part == "poison" else "")
        out[label] = sub

    items = {it["id"]: it for it in load("hotpotqa_poison2.json")}
    sub = pd.read_csv(os.path.join(OUT_RESULTS, "poison2_rows.csv")).copy()
    sub["question"] = sub.id.map(lambda i: items[i]["question"])
    sub["reference_answer"] = sub.id.map(lambda i: items[i]["answer"])
    sub["fake_answer"] = sub.id.map(lambda i: items[i]["substitute"])
    sub["method"] = sub.method + " (" + sub.condition + ")"
    out["study 2, poison"] = sub
    return out


def main():
    rng = random.Random(SEED)
    seen, rows = set(), []
    for source, df in candidates().items():
        df = df.dropna(subset=["pred"]).drop_duplicates(["question", "pred"])
        df = df[[(q, p) not in seen for q, p in zip(df.question, df.pred)]]
        poison = source.endswith("poison")
        pick = stratified(rng, df, N[source],
                          (lambda r: "adopted" if r["adoption"] == 1 else "not adopted") if poison else outcome)
        for _, r in pick.iterrows():
            seen.add((r["question"], r["pred"]))
            rows.append({"source": source, "question": r["question"], "reference_answer": r["reference_answer"],
                         "fake_answer": r["fake_answer"], "model_answer": r["pred"],
                         "auto_accuracy": r["accuracy"], "auto_abstain": r["abstain"],
                         "auto_wrong": (1 - r["accuracy"]) * (1 - r["abstain"]),
                         "auto_adoption": r["adoption"] if poison else "", "model": r["model"], "method": r["method"],
                         "stratum": ("adopted" if r["adoption"] == 1 else "not adopted") if poison else outcome(r)})
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "row", range(1, len(df) + 1))

    sheet_path = os.path.join(HERE, "annotation_sheet.csv")
    write_sheet = True
    if os.path.exists(sheet_path):
        old = pd.read_csv(sheet_path, encoding="utf-8-sig")
        if old[LABELS].notna().any().any():
            # Never overwrite human labels: refresh only the key, and only if it still describes the same rows
            same = old[SHOWN].fillna("").astype(str).equals(df[SHOWN].fillna("").astype(str))
            if not same:
                raise SystemExit("annotation_sheet.csv already holds labels for a different sample; nothing written")
            write_sheet = False
    if write_sheet:
        sheet = df[SHOWN].copy()
        for col in LABELS + ["notes"]:
            sheet[col] = ""
        sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    df[["row", "source", "stratum", "model", "method", "auto_accuracy", "auto_abstain", "auto_wrong",
        "auto_adoption"]].to_csv(os.path.join(HERE, "annotation_key.csv"), index=False, encoding="utf-8")
    counts = df.groupby(["source", "stratum"]).size()
    print(f"{'wrote' if write_sheet else 'kept the labelled sheet and refreshed the key for'} {len(df)} rows")
    print(counts.to_string())


if __name__ == "__main__":
    main()
