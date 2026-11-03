"""
Impact of the 2026-11-03 scoring corrections (see DEVIATIONS.md): as-run vs corrected results, side by side.

The as-run files were archived in results/as_run_2026-11-03/ before anything was re-scored; the corrected files are
the current ones in results/. No model was re-run: every answer comes from the same cached LLM outputs.

Usage:  py evaluation/compare_as_run.py
Output: evaluation/results/deviation_impact.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import OUT_RESULTS  # noqa: E402

AS_RUN = os.path.join(OUT_RESULTS, "as_run_2026-11-03")


def pair(name):
    out = []
    for folder in (AS_RUN, OUT_RESULTS):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def tests(old, new, key, fields):
    """Each confirmatory test before and after, matched on its identifying fields."""
    ident = lambda t: tuple(t.get(k) for k in key)  # noqa: E731
    before = {ident(t): t for t in old}
    return [{**{k: t.get(k) for k in key},
             **{f"{f}_as_run": before[ident(t)][f] for f in fields}, **{f"{f}_corrected": t[f] for f in fields},
             "significance_changed": (before[ident(t)]["p_holm"] < 0.05) != (t["p_holm"] < 0.05)}
            for t in new]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    impact = {}
    for tag, key in (("poison", ("model", "condition", "comparator", "metric")),
                     ("poison2", ("model", "condition", "comparator", "metric")),
                     ("clean500", ("hypothesis", "model", "comparator", "metric"))):
        old, new = pair(f"{tag}_summary.json")
        impact[tag] = {"decisions_as_run": old["decisions"], "decisions_corrected": new["decisions"],
                       "confirmatory": tests(old["confirmatory"], new["confirmatory"], key,
                                             ("mean_diff", "ci_low", "ci_high", "p_holm"))}
        if tag == "clean500":
            impact[tag]["non_inferiority"] = [
                {"model": n["model"], "ci_low_as_run": o["ci_low"], "ci_low_corrected": n["ci_low"],
                 "holds_as_run": o["holds"], "holds_corrected": n["holds"]}
                for o, n in zip(old["non_inferiority"], new["non_inferiority"])]
    old, new = pair("llm_summary.json")
    sig = lambda s: sorted(f"{t['model']} | {t['dataset']} | {t['reference']} vs {t['comparator']} | {t['metric']}"  # noqa: E731
                           for t in s["paired_tests"] if t["p_holm"] < 0.05)
    impact["clean_llm"] = {"significant_as_run": sig(old), "significant_corrected": sig(new)}
    with open(os.path.join(OUT_RESULTS, "deviation_impact.json"), "w", encoding="utf-8") as f:
        json.dump(impact, f, indent=1, ensure_ascii=False)

    for tag in ("poison", "poison2", "clean500"):
        d = impact[tag]
        print(f"\n== {tag}")
        for k in d["decisions_as_run"]:
            print(f"  decision {k}: as run {d['decisions_as_run'][k]!s:<16} corrected {d['decisions_corrected'][k]}")
        for t in d["confirmatory"]:
            flag = "  <-- significance changed" if t["significance_changed"] else ""
            label = " | ".join(str(t[k]) for k in t if k.endswith(("model", "condition", "comparator", "metric",
                                                                   "hypothesis")) and t[k] is not None)
            print(f"  {label:<62} diff {t['mean_diff_as_run']:+.3f} -> {t['mean_diff_corrected']:+.3f}   "
                  f"holm {t['p_holm_as_run']:.3f} -> {t['p_holm_corrected']:.3f}{flag}")
        for t in d.get("non_inferiority", []):
            print(f"  H3 {t['model']}: CI low {t['ci_low_as_run']:+.3f} -> {t['ci_low_corrected']:+.3f}  "
                  f"holds {t['holds_as_run']} -> {t['holds_corrected']}")
    c = impact["clean_llm"]
    print("\n== clean benchmark LLM tests significant after Holm")
    print("  as run:   ", *c["significant_as_run"], sep="\n    ")
    print("  corrected:", *c["significant_corrected"], sep="\n    ")


if __name__ == "__main__":
    main()
