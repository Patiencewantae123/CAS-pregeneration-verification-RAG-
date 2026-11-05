"""
Build the study 6 samples exactly as specified in PREREGISTRATION_study6.md.

  hotpotqa  N_HOTPOT fresh HotpotQA dev questions (distractor setting, 10 passages), disjoint from every earlier study
  musique   N_MUSIQUE answerable MuSiQue dev questions (20 paragraphs, 2-4 hops), disjoint from the development check

Usage:  py evaluation/build_study6.py      (run_study6.py calls it once the plan is committed)
Output: data/study6.json
"""
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_study6 import MUSIQUE, musique_item  # noqa: E402
from run_evaluation import DATA, SEED, load  # noqa: E402

N_HOTPOT, N_MUSIQUE = 700, 200
OUT = os.path.join(DATA, "study6.json")


def used_hotpot_questions():
    """Every HotpotQA question used, attacked or attempted in studies 1-5 and the earlier experiments."""
    used = {r["question"] for r in load("hotpotqa_hf_samples.json")}
    for name in ("hotpotqa_poison.json", "hotpotqa_poison2.json", "hotpotqa_clean500.json"):
        used |= {it["question"] for it in load(name)}
    s5 = load("hotpotqa_study5.json")
    used |= {it["question"] for it in s5["clean"] + s5["poison"]}
    with open(os.path.join(DATA, "external", "study5_attacker.jsonl"), encoding="utf-8") as f:
        used |= {json.loads(line)["key"].split("|", 2)[2].rsplit("|", 1)[0] for line in f}   # model|seed|q|answer
    return used


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(SEED)

    used = used_hotpot_questions()
    df = pd.read_parquet(os.path.join(DATA, "external", "hotpot_dev_distractor.parquet"))
    pool = []
    for r in df.itertuples():
        if r.question in used:
            continue
        titles = list(r.context["title"])
        support = set(r.supporting_facts["title"])
        pool.append({"id": r.id, "question": r.question, "answers": [r.answer.strip()], "type": r.type,
                     "titles": titles,
                     "passages": [f"{t}. {''.join(s)}" for t, s in zip(titles, r.context["sentences"])],
                     "gold": sorted(i for i, t in enumerate(titles) if t in support)})
    hotpot = rng.sample(pool, N_HOTPOT)

    with open(os.path.join(DATA, "study6_dev_ids.json"), encoding="utf-8") as f:
        excluded = set(json.load(f))
    with open(MUSIQUE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    # Five question texts occur twice under different ids: exclude by text as well as id, and keep one of each
    excluded_q = {r["question"] for r in load("musique.json")} | {r["question"] for r in rows if r["id"] in excluded}
    pool, seen = [], set(excluded_q)
    for r in rows:
        if r["id"] not in excluded and r["question"] not in seen:
            seen.add(r["question"])
            pool.append(musique_item(r))
    musique = rng.sample(pool, N_MUSIQUE)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"hotpotqa": hotpot, "musique": musique}, f, ensure_ascii=False, indent=1)
    print(f"hotpotqa {len(hotpot)} (pool {len(df)} minus {len(used)} used), musique {len(musique)} -> {OUT}")


if __name__ == "__main__":
    main()
