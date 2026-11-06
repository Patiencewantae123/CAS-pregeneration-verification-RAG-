"""
Exploratory component ablation of CAS v2 at the selection level (no LLM): what each step contributes.

Variants: full v2; without title links; without the copy quarantine; without both (the trust threshold alone).
  Clean evidence   study 6 (HotpotQA 700, MuSiQue 200) and study 5 clean (HotpotQA 200): evidence F1, full evidence
  Poisoning        study 5 copy and written attacks, condition A: poison passages reaching the prompt, true passage
                   kept
Selections are recomputed; the full-v2 selections are checked against the cached ones of run_study5.py and
run_study6.py.

Usage:  py evaluation/run_v2_ablation.py
Output: evaluation/results/v2_ablation.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_v2  # noqa: E402
from build_poison_benchmark import contains  # noqa: E402
from run_evaluation import DATA, OUT_RESULTS, prf  # noqa: E402

VARIANTS = {"CAS v2": {}, "without links": {"link": False}, "without quarantine": {"quarantine": False},
            "trust threshold only": {"link": False, "quarantine": False}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s6 = json.load(open(os.path.join(DATA, "study6.json"), encoding="utf-8"))
    s5 = json.load(open(os.path.join(DATA, "hotpotqa_study5.json"), encoding="utf-8"))
    sel6 = json.load(open(os.path.join(OUT_RESULTS, "study6_selections.json"), encoding="utf-8"))
    sel5 = json.load(open(os.path.join(OUT_RESULTS, "study5_selections.json"), encoding="utf-8"))
    out = {"note": "Exploratory, selection level; chosen after studies 5 and 6 were analysed.", "clean": [],
           "poison": [], "mismatches_with_cache": 0}

    clean = {"study 6 HotpotQA": [(it, f"hotpotqa|CAS v2|{it['id']}", sel6) for it in s6["hotpotqa"]],
             "study 6 MuSiQue": [(it, f"musique|CAS v2|{it['id']}", sel6) for it in s6["musique"]],
             "study 5 HotpotQA": [(it, f"clean|-|-|CAS v2|{it['id']}", sel5) for it in s5["clean"]]}
    for name, items in clean.items():
        for variant, switches in VARIANTS.items():
            r = []
            for it, key, cache in items:
                chosen = cas_v2.select(it["question"], it["passages"], it["titles"], **switches)[0]
                if variant == "CAS v2":
                    out["mismatches_with_cache"] += sorted(chosen) != cache[key]["selected"]
                p, rec, f = prf(chosen, set(it["gold"]))
                r.append((f, rec == 1, len(chosen)))
            r = np.array(r, dtype=float)
            out["clean"].append({"data": name, "variant": variant, "f1": r[:, 0].mean(),
                                 "full_evidence": r[:, 1].mean(), "n_selected": r[:, 2].mean()})
            print(out["clean"][-1], flush=True)

    for attack in ("copy", "written"):
        for variant, switches in VARIANTS.items():
            r = []
            for it in s5["poison"]:
                view = it[attack]
                chosen = cas_v2.select(it["question"], view["passages"], view["titles"], **switches)[0]
                if variant == "CAS v2":
                    out["mismatches_with_cache"] += sorted(chosen) != sel5[f"poison|{attack}|A|CAS v2|{it['id']}"][
                        "selected"]
                true = next(i for i in view["gold"] if contains(view["passages"][i], it["answer"]))
                r.append((bool(chosen & set(view["poison"])), len(chosen & set(view["poison"])), true in chosen))
            r = np.array(r, dtype=float)
            out["poison"].append({"attack": attack, "condition": "A", "variant": variant,
                                  "poison_in_prompt": r[:, 0].mean(), "poison_passages": r[:, 1].mean(),
                                  "true_passage_kept": r[:, 2].mean()})
            print(out["poison"][-1], flush=True)

    with open(os.path.join(OUT_RESULTS, "v2_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("mismatches with cache:", out["mismatches_with_cache"])


if __name__ == "__main__":
    main()
