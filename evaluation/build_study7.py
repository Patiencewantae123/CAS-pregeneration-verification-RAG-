"""
Build the study 7 sample exactly as specified in PREREGISTRATION_study7.md.

  hotpotqa  N fresh HotpotQA dev questions (distractor setting, 10 passages), disjoint from every question used,
            attacked or attempted in studies 1-6 and the earlier experiments

Usage:  py evaluation/build_study7.py      (run_study7.py calls it once the plan is committed)
Output: data/study7.json
"""
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_study6 import OUT as STUDY6, used_hotpot_questions  # noqa: E402
from run_evaluation import DATA, SEED  # noqa: E402

N = 1000
OUT = os.path.join(DATA, "study7.json")


def used_questions():
    """Studies 1-5 and the earlier experiments (as excluded by study 6), plus study 6's HotpotQA sample."""
    with open(STUDY6, encoding="utf-8") as f:
        s6 = json.load(f)
    return used_hotpot_questions() | {it["question"] for it in s6["hotpotqa"]}, {it["id"] for it in s6["hotpotqa"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(SEED)
    used_q, used_id = used_questions()
    df = pd.read_parquet(os.path.join(DATA, "external", "hotpot_dev_distractor.parquet"))
    pool = []
    for r in df.itertuples():
        if r.question in used_q or r.id in used_id:
            continue
        titles = list(r.context["title"])
        support = set(r.supporting_facts["title"])
        pool.append({"id": r.id, "question": r.question, "answers": [r.answer.strip()], "type": r.type,
                     "titles": titles,
                     "passages": [f"{t}. {''.join(s)}" for t, s in zip(titles, r.context["sentences"])],
                     "gold": sorted(i for i, t in enumerate(titles) if t in support)})
    sample = rng.sample(pool, N)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"hotpotqa": sample}, f, ensure_ascii=False, indent=1)
    print(f"hotpotqa {len(sample)} (pool {len(pool)} eligible of {len(df)}) -> {OUT}")


if __name__ == "__main__":
    main()
