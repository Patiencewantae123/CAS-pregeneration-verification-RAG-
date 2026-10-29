"""Regenerate every journal figure, then check each plotted CSV value against results.json."""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

scripts = sorted(f for f in os.listdir(HERE) if f.startswith("fig") and f.endswith(".py"))
for s in scripts:
    runpy.run_path(os.path.join(HERE, s), run_name="__main__")

# Traceability check: every evidence-selection value in Fig. 1 and Fig. 5 must equal results.json
import pandas as pd  # noqa: E402
R = json.load(open(os.path.join(os.path.dirname(HERE), "evaluation", "results", "results.json"), encoding="utf-8"))
ev = {(r["dataset"], r["method"], r["metric"]): r["mean"] for r in R["evidence_selection"]}
bad = 0
for name, cols in [("fig01_precision_recall_tradeoff", ("precision", "recall", "f1")),
                   ("fig05_performance_cost", ("f1",))]:
    df = pd.read_csv(os.path.join(HERE, "data", f"{name}.csv"))
    for _, row in df.iterrows():
        for c in cols:
            if abs(row[c] - ev[(row["dataset"], row["method"], c)]) > 1e-12:
                bad += 1
                print("MISMATCH", name, row["dataset"], row["method"], c)
pt = {}
for t in R["paired_tests"]:
    pt[(t["dataset"], t["reference"], t["comparator"], t["metric"])] = t["mean_diff"]
    pt[(t["dataset"], t["comparator"], t["reference"], t["metric"])] = -t["mean_diff"]
df = pd.read_csv(os.path.join(HERE, "data", "fig02_paired_differences.csv"))
for _, row in df.iterrows():
    if abs(row["mean_diff"] - pt[(row["dataset"], row["reference"], row["comparator"], row["metric"])]) > 1e-12:
        bad += 1
        print("MISMATCH fig02", row.to_dict())
for f in [f"{s[:-3]}.{ext}" for s in scripts for ext in ("png", "pdf")]:
    if not os.path.getsize(os.path.join(HERE, f)):
        bad += 1
        print("EMPTY", f)
print(f"\nTraceability check: {'PASS' if bad == 0 else f'{bad} problems'} ({len(scripts)} figures)")
