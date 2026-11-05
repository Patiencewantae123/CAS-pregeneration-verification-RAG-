"""
Exploratory, post hoc (not in PREREGISTRATION_study5.md): which step of CAS v2 removed what in study 5.

Recomputes v2's selections on the study 5 data (deterministic, no LLM), checks them against the cached selections
of run_study5.py, and joins them with the scored answers in study5_rows.csv:
  1. the fate of every poison passage (gated, quarantined, below tau, kept) and of the true answer passage;
  2. written attack: adoption by the number of poison passages that reached v2's prompt;
  3. clean set: how often the quarantine and the title links fired, and whether the linked passages were gold;
  4. written attack: why v2 dropped the true answer passage, and whether it keeps it once the poison is removed.

Usage:  py evaluation/run_study5_mechanism.py
Output: evaluation/results/study5_mechanism.json
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_v2  # noqa: E402
from build_poison_benchmark import contains  # noqa: E402
from run_evaluation import DATA, OUT_RESULTS, bm25_scores  # noqa: E402

with open(os.path.join(DATA, "hotpotqa_study5.json"), encoding="utf-8") as f:
    data = json.load(f)
with open(os.path.join(OUT_RESULTS, "study5_selections.json"), encoding="utf-8") as f:
    cached = json.load(f)
rows = pd.read_csv(os.path.join(OUT_RESULTS, "study5_rows.csv"))
out = {"note": "Exploratory, post hoc; not part of the pre-registered analysis.", "fate": [], "written_by_count": [],
       "clean": {}}
mismatch = 0

# 1. Fate of the poison passages and of the true answer passage
counts = []
for attack in ("copy", "written"):
    for cond in ("A", "B"):
        fate = {"gated": 0, "quarantined": 0, "below_tau": 0, "kept": 0}
        true_quarantined = true_kept = 0
        for it in data["poison"]:
            view = it[attack]
            prov = view["provenance_B"] if cond == "B" else None
            chosen, info = cas_v2.select(it["question"], view["passages"], view["titles"], prov)
            mismatch += sorted(chosen) != cached[f"poison|{attack}|{cond}|CAS v2|{it['id']}"]["selected"]
            for i in view["poison"]:
                step = ("gated" if i in info["gated"] else "quarantined" if i in info["quarantined"]
                        else "kept" if i in chosen else "below_tau")
                fate[step] += 1
            true = next(i for i in view["gold"] if contains(view["passages"][i], it["answer"]))
            true_quarantined += true in info["quarantined"]
            true_kept += true in chosen
            if cond == "A":
                counts.append({"attack": attack, "id": it["id"], "poison_kept": len(chosen & set(view["poison"])),
                               "n_selected": len(chosen)})
        n = len(data["poison"])
        out["fate"].append({"attack": attack, "condition": cond, "n_questions": n,
                            **{f"poison_{k}_per_question": v / n for k, v in fate.items()},
                            "true_passage_quarantined": true_quarantined / n, "true_passage_kept": true_kept / n})

# 2. Written attack, no metadata: adoption by the number of poison passages in v2's prompt
counts = pd.DataFrame(counts)
w = counts[counts["attack"] == "written"].set_index("id")
for model in rows["model"].unique():
    sub = rows[(rows["model"] == model) & (rows["attack"] == "written") & (rows["condition"] == "A")]
    v2 = sub[sub["method"] == "CAS v2"].set_index("id")
    van = sub[sub["method"] == "Vanilla RAG"].set_index("id")
    for k in range(4):
        ids = w.index[w["poison_kept"] == k]
        if len(ids):
            out["written_by_count"].append({
                "model": model, "poison_kept": k, "n": len(ids),
                "poison_share_of_prompt_v2": float((k / w.loc[ids, "n_selected"]).mean()),
                "adoption_v2": float(v2.loc[ids, "adoption"].mean()),
                "adoption_vanilla_same_questions": float(van.loc[ids, "adoption"].mean())})
out["written_mean_poison_kept"] = float(w["poison_kept"].mean())
out["written_poison_share_of_prompt"] = {"CAS v2": float((w["poison_kept"] / w["n_selected"]).mean()),
                                         "Vanilla RAG": float(pd.Series([3 / len(it["written"]["passages"])
                                                                         for it in data["poison"]]).mean())}

# 3. Clean set: quarantine and title links
fired = linked_q = linked_n = linked_gold = 0
for it in data["clean"]:
    chosen, info = cas_v2.select(it["question"], it["passages"], it["titles"])
    mismatch += sorted(chosen) != cached[f"clean|-|-|CAS v2|{it['id']}"]["selected"]
    fired += bool(info["quarantined"])
    linked_q += bool(info["linked"])
    linked_n += len(info["linked"])
    linked_gold += len(set(info["linked"]) & set(it["gold"]))
n = len(data["clean"])
out["clean"] = {"n_questions": n, "quarantine_fired": fired / n, "links_added_any": linked_q / n,
                "linked_passages": linked_n, "linked_passages_gold": linked_gold}

# 4. Written attack, no metadata: the true answer passage, with and without the poison passages
dropped = poison_top3 = by_trust = kept_without = 0
for it in data["poison"]:
    view = it["written"]
    true = next(i for i in view["gold"] if contains(view["passages"][i], it["answer"]))
    chosen, info = cas_v2.select(it["question"], view["passages"], view["titles"])
    if true not in chosen:
        raw = bm25_scores(it["question"], view["passages"])
        dropped += 1
        poison_top3 += set(sorted(range(len(raw)), key=lambda i: -raw[i])[:3]) == set(view["poison"])
        by_trust += true not in info["quarantined"]
    keep = [i for i in range(len(view["passages"])) if i not in view["poison"]]
    chosen, _ = cas_v2.select(it["question"], [view["passages"][i] for i in keep], [view["titles"][i] for i in keep])
    kept_without += keep.index(true) in chosen
out["written_true_passage"] = {"dropped": dropped, "of_which_poison_held_top3_bm25": poison_top3,
                               "of_which_below_tau_not_quarantined": by_trust,
                               "kept_when_poison_removed": kept_without, "n_questions": len(data["poison"])}
out["selection_mismatches_with_cache"] = int(mismatch)

with open(os.path.join(OUT_RESULTS, "study5_mechanism.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
