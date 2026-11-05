"""
Development before study 6, after the pilot (dev_study6.py) showed that CAS v2 loses evidence on MuSiQue.

Three ways to recover the evidence were tried on development data only, never on the study 6 samples:
  1. hop expansion   v2, then up to R rounds that add the passages scoring >= ratio x the best BM25 score against
                     the text added in the previous round                                        (not adopted)
  2. cross-encoder   iterative selection with ms-marco-MiniLM: hop 1 scores against the question, later hops against
     chain           the question plus the previous hop's passage; v2's quarantine first, title links last
                                                                                                  (not adopted)
  3. abstention      answer with v2's selection; if the model abstains, answer again with every passage that passed
     fallback        v2's gate and quarantine                                       (adopted as a study 6 arm)
Data: MuSiQue, the 200 development-check questions (answers: the first 60); HotpotQA, study 3's questions for
selection and study 5's 200 clean questions for answers (both already analysed).

Usage:  py evaluation/dev_study6_variants.py
Output: evaluation/results/study6_variants_dev.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_v2  # noqa: E402
from dev_study6 import MUSIQUE, musique_item  # noqa: E402
from run_evaluation import DATA, OUT_RESULTS, bm25_scores, paired_test, prf  # noqa: E402
from run_llm_generation import QA_PROMPT, format_passages, generate, load_cache, score_qa  # noqa: E402

MODELS = ["llama3.2:latest", "gemma3:latest"]


def load_dev():
    ids = json.load(open(os.path.join(DATA, "study6_dev_ids.json"), encoding="utf-8"))
    rows = {}
    with open(MUSIQUE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    mus = [musique_item(rows[i]) for i in ids]
    pq = pd.read_parquet(os.path.join(DATA, "external", "hotpot_dev_distractor.parquet")).set_index("id")
    hot = []
    for it in json.load(open(os.path.join(DATA, "hotpotqa_clean500.json"), encoding="utf-8"))[:200]:
        known = sorted(pq.loc[it["id"], "context"]["title"], key=len, reverse=True)
        titles = [next((t for t in known if p.startswith(t + ". ")), None) for p in it["passages"]]
        hot.append({"question": it["question"], "passages": it["passages"], "titles": titles, "gold": it["gold"]})
    s5 = json.load(open(os.path.join(DATA, "hotpotqa_study5.json"), encoding="utf-8"))["clean"]
    hot_answers = [{"question": it["question"], "passages": it["passages"], "titles": it["titles"],
                    "answers": [it["answer"]]} for it in s5]
    return mus, hot, hot_answers


def safe_pool(it):
    _, info = cas_v2.select(it["question"], it["passages"], it["titles"])
    return [i for i in range(len(it["passages"])) if i not in set(info["quarantined"]) | set(info["gated"])]


def hop_expand(it, rounds, ratio):
    chosen = cas_v2.select(it["question"], it["passages"], it["titles"])[0]
    pool, frontier = set(safe_pool(it)), chosen
    for _ in range(rounds):
        rest = sorted(pool - chosen)
        if not rest or not frontier:
            break
        raw = bm25_scores(" ".join(it["passages"][i] for i in frontier), [it["passages"][i] for i in rest])
        frontier = {rest[j] for j, s in enumerate(raw) if max(raw) > 0 and s >= ratio * max(raw)}
        chosen = chosen | frontier
    return chosen


def chain(it, ce, hops):
    """The passage chosen at each hop, with its cross-encoder score."""
    pool, out, prev = safe_pool(it), [], ""
    for _ in range(hops):
        rest = [i for i in pool if i not in {c for c, _ in out}]
        if not rest:
            break
        s = ce.predict([((it["question"] + " " + prev).strip(), it["passages"][i]) for i in rest],
                       show_progress_bar=False)
        j = int(np.argmax(s))
        out.append((rest[j], float(s[j])))
        prev = it["passages"][rest[j]][:600]
    return out


def with_links(it, chosen):
    context = it["question"] + " " + " ".join(it["passages"][i] for i in chosen)
    return chosen | {i for i in safe_pool(it) if i not in chosen and cas_v2._named(it["titles"][i], context)}


def evidence(items, sets):
    r = np.array([(prf(s, set(it["gold"]))[1] == 1, prf(s, set(it["gold"]))[2], len(s))
                  for s, it in zip(sets, items)], dtype=float)
    return {"full_evidence": r[:, 0].mean(), "f1": r[:, 1].mean(), "n_selected": r[:, 2].mean()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import CrossEncoder
    mus, hot, hot_answers = load_dev()
    out = {"note": "Development only (held-out MuSiQue questions; already-analysed HotpotQA); unadjusted.",
           "selection": [], "fallback": []}
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    for name, items in (("MuSiQue", mus), ("HotpotQA", hot)):
        def add(variant, sets):
            out["selection"].append({"dataset": name, "variant": variant, **evidence(items, sets)})
            print(out["selection"][-1], flush=True)

        add("v2", [cas_v2.select(it["question"], it["passages"], it["titles"])[0] for it in items])
        for rounds in (1, 2, 3):
            for ratio in (1.0, 0.8, 0.6):
                add(f"hop expansion R={rounds} ratio={ratio}", [hop_expand(it, rounds, ratio) for it in items])
        chains = [chain(it, ce, 4) for it in items]
        for hops in (2, 3, 4):
            for t in (None, 0.0, -3.0):
                sets = [{c for k, (c, s) in enumerate(ch[:hops]) if k == 0 or t is None or s >= t} for ch in chains]
                add(f"chain H={hops} t={t}", sets)
                add(f"chain H={hops} t={t} + links", [with_links(it, s) for it, s in zip(items, sets)])

    cache = load_cache()

    def ask(model, it, chosen):
        context = format_passages(it["passages"], chosen)
        rec = generate(model, QA_PROMPT.format(passages=context, question=it["question"]), cache)
        sc = score_qa(rec["response"], it["answers"], context)
        return sc["accuracy"], sc["abstain"], (1 - sc["accuracy"]) * (1 - sc["abstain"])

    for name, items in (("HotpotQA (study 5 clean)", hot_answers), ("MuSiQue (pilot)", mus[:60])):
        for model in MODELS:
            van, v2, fb, used = [], [], [], 0
            for it in items:
                chosen = cas_v2.select(it["question"], it["passages"], it["titles"])[0]
                van.append(ask(model, it, set(range(len(it["passages"])))))
                v2.append(ask(model, it, chosen))
                fb.append(ask(model, it, set(safe_pool(it))) if v2[-1][1] else v2[-1])
                used += bool(v2[-1][1])
            van, v2, fb = (np.array(x) for x in (van, v2, fb))
            rec = {"dataset": name, "model": model, "n": len(items), "fallback_used": used}
            for label, arr in (("vanilla", van), ("v2", v2), ("fallback", fb)):
                rec |= {f"{label}_accuracy": arr[:, 0].mean(), f"{label}_wrong": arr[:, 2].mean(),
                        f"{label}_abstain": arr[:, 1].mean()}
            for metric, k in (("accuracy", 0), ("wrong", 2)):
                m, lo, hi, p = paired_test(fb[:, k], van[:, k])
                rec |= {f"fallback_minus_vanilla_{metric}": m, f"{metric}_ci": [lo, hi], f"{metric}_p": p}
            out["fallback"].append(rec)
            print(rec, flush=True)

    with open(os.path.join(OUT_RESULTS, "study6_variants_dev.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
