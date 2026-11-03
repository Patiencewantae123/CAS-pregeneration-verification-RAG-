"""
Build a blind annotation sheet for validating the automatic answer metrics (accuracy, abstention, poison adoption).

The annotator sees the question, the reference answer(s) and the model's answer, never the automatic score.
The automatic scores go to a separate key file that score_agreement.py joins afterwards.

Usage:  py evaluation/human_eval/make_sheet.py
Output: evaluation/human_eval/annotation_sheet.csv (to fill in), annotation_key.csv (do not open before labelling)
"""
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from run_evaluation import OUT_RESULTS, SEED, load  # noqa: E402
from run_llm_generation import qa_tasks  # noqa: E402

N_CLEAN, N_POISON = 75, 75


def main():
    rng = random.Random(SEED)
    questions = {(ds, k): q for ds, k, q, *_ in qa_tasks(10 ** 6)}
    clean = pd.read_csv(os.path.join(OUT_RESULTS, "llm_scores.csv"))
    clean = clean.drop_duplicates(["dataset", "item", "pred"])
    rows = []
    for _, r in clean.loc[rng.sample(list(clean.index), N_CLEAN)].iterrows():
        rows.append({"source": "clean", "question": questions[(r["dataset"], int(r["item"]))],
                     "reference_answer": r["gold"], "fake_answer": "", "model_answer": r["pred"],
                     "auto_accuracy": r["accuracy"], "auto_abstain": r["abstain"], "auto_adoption": "",
                     "model": r["model"], "method": r["method"]})
    items = {it["id"]: it for it in load("hotpotqa_poison2.json")}
    poison = pd.read_csv(os.path.join(OUT_RESULTS, "poison2_rows.csv")).drop_duplicates(["id", "pred"])
    for _, r in poison.loc[rng.sample(list(poison.index), N_POISON)].iterrows():
        it = items[r["id"]]
        rows.append({"source": "poison", "question": it["question"], "reference_answer": it["answer"],
                     "fake_answer": it["substitute"], "model_answer": r["pred"],
                     "auto_accuracy": r["accuracy"], "auto_abstain": r["abstain"], "auto_adoption": r["adoption"],
                     "model": r["model"], "method": f"{r['method']} ({r['condition']})"})
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "row", range(1, len(df) + 1))
    shown = ["row", "question", "reference_answer", "fake_answer", "model_answer"]
    labels = ["correct (1/0)", "abstains (1/0)", "gives_fake_answer (1/0, poison rows only)"]
    sheet_path = os.path.join(HERE, "annotation_sheet.csv")
    write_sheet = True
    if os.path.exists(sheet_path):
        old = pd.read_csv(sheet_path, encoding="utf-8-sig")
        if old[labels].notna().any().any():
            # Never overwrite human labels: refresh only the key, and only if it still describes the same rows
            same = old[shown].fillna("").astype(str).equals(df[shown].fillna("").astype(str))
            if not same:
                raise SystemExit("annotation_sheet.csv already holds labels for a different sample; nothing written")
            write_sheet = False
    if write_sheet:
        sheet = df[shown].copy()
        for col in labels + ["notes"]:
            sheet[col] = ""
        sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    df[["row", "source", "model", "method", "auto_accuracy", "auto_abstain", "auto_adoption"]].to_csv(
        os.path.join(HERE, "annotation_key.csv"), index=False, encoding="utf-8")
    print(f"{'wrote' if write_sheet else 'kept the labelled sheet and refreshed the key for'} {len(df)} rows "
          f"({N_CLEAN} clean, {N_POISON} poison)")


if __name__ == "__main__":
    main()
