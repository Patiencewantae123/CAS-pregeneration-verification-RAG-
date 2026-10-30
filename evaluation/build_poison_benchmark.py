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


def eligible_questions(exclude):
    df = pd.read_parquet(SRC)
    eligible = []
    for r in df.itertuples():
        ans = r.answer.strip()
        if norm_answer(ans) in ("yes", "no") or len(ans.split()) > 5 or r.question in exclude:
            continue
        titles = list(r.context["title"])
        sentences = [list(s) for s in r.context["sentences"]]
        passages = [f"{t}. {''.join(s)}" for t, s in zip(titles, sentences)]
        gold = sorted({i for i, t in enumerate(titles) if t in set(r.supporting_facts["title"])})
        if not any(contains(passages[i], ans) for i in gold):
            continue
        eligible.append({"id": r.id, "question": r.question, "answer": ans, "type": r.type, "titles": titles,
                         "sentences": sentences, "passages": passages, "gold": gold})
    return eligible


def type_matched_substitute(rng, q):
    """Study 2: a plausible substitute of the same type (PREREGISTRATION_poison2.md)."""
    ans = q["answer"]
    gold_text = " ".join(q["passages"][i] for i in q["gold"])
    for _ in range(100):
        if re.fullmatch(r"(1\d|20)\d\d", ans):
            sub = str(int(ans) + rng.choice([-1, 1]) * rng.randint(3, 15))
        elif any(ch.isdigit() for ch in ans):
            first = [True]

            def digit(_m):
                d = str(rng.randint(1, 9) if first[0] else rng.randint(0, 9))
                first[0] = False
                return d
            sub = re.sub(r"\d", digit, ans)
        else:
            titles = [re.sub(r"\s*\([^)]*\)$", "", t) for i, t in enumerate(q["titles"]) if i not in q["gold"]]
            close = [t for t in titles if abs(len(t.split()) - len(ans.split())) <= 1]
            sub = rng.choice(close or titles)
        if norm_answer(sub) != norm_answer(ans) and not contains(gold_text, sub):
            return sub
    return None


def build_study2(eligible, rng):
    items = []
    for q in rng.sample(eligible, len(eligible)):
        sub = type_matched_substitute(rng, q)
        if sub is None:
            continue
        g = next(i for i in q["gold"] if contains(q["passages"][i], q["answer"]))
        p1 = substitute(q["passages"][g], q["answer"], sub)
        sents = [substitute(s, q["answer"], sub) for s in q["sentences"][g] if contains(s, q["answer"])]
        poisons = [p1, f"{q['question']} {p1}", f"{q['titles'][g]}. {''.join(sents).strip()}"]
        passages, is_poison, gold = list(q["passages"]), [False] * len(q["passages"]), set(q["gold"])
        for p in poisons:
            pos = rng.randint(0, len(passages))
            passages.insert(pos, p)
            is_poison.insert(pos, True)
            gold = {x + 1 if x >= pos else x for x in gold}
        prov = [provenance(rng, f) for f in is_poison]
        items.append({"id": q["id"], "question": q["question"], "answer": q["answer"], "substitute": sub,
                      "type": q["type"], "passages": passages, "gold": sorted(gold),
                      "poison": [i for i, f in enumerate(is_poison) if f], "provenance_B": prov})
        if len(items) == N:
            break
    return items


def main():
    study = sys.argv[1] if len(sys.argv) > 1 else "1"
    used = {r["question"] for r in load("hotpotqa_hf_samples.json")}
    if study == "2":
        with open(OUT, encoding="utf-8") as f:
            used |= {it["question"] for it in json.load(f)}
        items = build_study2(eligible_questions(used), random.Random(SEED))
        out = os.path.join(DATA, "hotpotqa_poison2.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        print(f"study 2: {len(items)} questions, 3 poison passages each -> {out}")
        return
    eligible = eligible_questions(used)

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
