"""
Build HotpotQA-Poison exactly as specified in PREREGISTRATION_poison.md.

Input:  data/external/hotpot_dev_distractor.parquet (official dev set, HF mirror hotpotqa/hotpot_qa)
Output: data/hotpotqa_poison.json
"""
import json
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import DATA, SEED, load, norm_answer  # noqa: E402

N = 200
SRC = os.path.join(DATA, "external", "hotpot_dev_distractor.parquet")
OUT = os.path.join(DATA, "hotpotqa_poison.json")


def contains(text, answer):
    return re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text, re.I) is not None


def substitute(text, answer, sub):
    return re.sub(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", sub, text, flags=re.I)


def shape(answer):
    return any(ch.isdigit() for ch in answer)


def provenance(rng, poison):
    """Condition B: trusted ~U(0.55, 0.95), untrusted ~U(0.05, 0.45), each flipped with probability 0.2."""
    trusted = (not poison) if rng.random() >= 0.2 else poison
    return rng.uniform(0.55, 0.95) if trusted else rng.uniform(0.05, 0.45)


def main():
    df = pd.read_parquet(SRC)
    used = {r["question"] for r in load("hotpotqa_hf_samples.json")}
    eligible = []
    for r in df.itertuples():
        ans = r.answer.strip()
        if norm_answer(ans) in ("yes", "no") or len(ans.split()) > 5 or r.question in used:
            continue
        titles = list(r.context["title"])
        passages = [f"{t}. {''.join(s)}" for t, s in zip(titles, r.context["sentences"])]
        gold = sorted({i for i, t in enumerate(titles) if t in set(r.supporting_facts["title"])})
        if not any(contains(passages[i], ans) for i in gold):
            continue
        eligible.append({"id": r.id, "question": r.question, "answer": ans, "type": r.type,
                         "passages": passages, "gold": gold})

    rng = random.Random(SEED)
    sample = rng.sample(eligible, N)
    items = []
    for k, q in enumerate(sample):
        pool_text = " ".join(q["passages"])
        candidates = [o["answer"] for o in sample if o is not q and shape(o["answer"]) == shape(q["answer"])
                      and norm_answer(o["answer"]) != norm_answer(q["answer"]) and not contains(pool_text, o["answer"])]
        sub = rng.choice(candidates)
        poisons = [substitute(q["passages"][i], q["answer"], sub) for i in q["gold"] if contains(q["passages"][i], q["answer"])]
        passages, is_poison = list(q["passages"]), [False] * len(q["passages"])
        gold = set(q["gold"])
        for p in poisons:
            pos = rng.randint(0, len(passages))
            passages.insert(pos, p)
            is_poison.insert(pos, True)
            gold = {g + 1 if g >= pos else g for g in gold}
        prov = [provenance(rng, f) for f in is_poison]
        items.append({"id": q["id"], "question": q["question"], "answer": q["answer"], "substitute": sub,
                      "type": q["type"], "passages": passages, "gold": sorted(gold),
                      "poison": [i for i, f in enumerate(is_poison) if f], "provenance_B": prov})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    n_poison = [len(it["poison"]) for it in items]
    print(f"eligible {len(eligible)}, sampled {len(items)}, poison passages per question: "
          f"{dict(pd.Series(n_poison).value_counts().sort_index())}")


if __name__ == "__main__":
    main()
