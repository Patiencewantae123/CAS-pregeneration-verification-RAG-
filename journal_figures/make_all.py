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
runpy.run_path(os.path.join(HERE, "make_tables.py"), run_name="__main__")

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

# The same check for the studies with their own result files: each plotted value must equal its source
RES = os.path.join(os.path.dirname(HERE), "evaluation", "results")


def compare(name, plotted, source, cols):
    global bad
    for c in cols:
        if abs(float(plotted[c]) - float(source[c])) > 1e-9:
            bad += 1
            print("MISMATCH", name, c, dict(plotted), source[c])


for name, tag in [("fig12_poison_study1", "poison"), ("fig13_poison_study2", "poison2")]:
    S = json.load(open(os.path.join(RES, f"{tag}_summary.json"), encoding="utf-8"))
    look = {(r["model"], r["condition"], r["method"], r["metric"]): r for r in S["descriptive"]}
    for _, row in pd.read_csv(os.path.join(HERE, "data", f"{name}.csv")).iterrows():
        compare(name, row, look[(row["model"], row["condition"], row["method"], row["metric"])],
                ("mean", "ci_low", "ci_high"))

probe = json.load(open(os.path.join(RES, "detector_probe.json"), encoding="utf-8"))
df = pd.read_csv(os.path.join(HERE, "data", "fig14_detector_tradeoff.csv"))
for (_, row), p in zip(df.iterrows(), probe["nli"] + probe["answer_level"], strict=True):
    compare("fig14_detector_tradeoff", row, p, ("s2_recall", "s2_false_alarm", "s2_sec_per_query"))

df = pd.read_csv(os.path.join(HERE, "data", "fig15_montecarlo.csv"))
e1, e2, e3 = (pd.read_csv(os.path.join(RES, f"montecarlo_e{k}.csv")) for k in (1, 2, 3))
e1, e2, e3 = (t[t["study"] == "pooled"] for t in (e1, e2, e3))
for _, row in df.iterrows():
    if row["panel"] == "a":
        g = e1[(e1["method"] == row["method"]) & (e1["condition"] == row["condition"])]
        rand = g[g["setting"] > 0]["silent_exposure"]
        ref = {"p05": rand.quantile(0.05), "median": rand.median(), "p95": rand.quantile(0.95),
               "default_setting": g[g["setting"] == 0]["silent_exposure"].iloc[0]}
        compare("fig15_montecarlo (a)", row, ref, ref.keys())
    elif row["panel"] == "b":
        g = e2[(e2["method"] == row["method"]) & (e2["separation"] == row["separation"]) & (e2["flip"] == row["flip"])]
        compare("fig15_montecarlo (b)", row, g.iloc[0], ("silent_exposure",))
    else:
        metric = "silent_exposure" if row["panel"] == "cd" else "answer_available"
        g = e3[(e3["condition"] == row["condition"]) & (e3["method"] == row["method"])
               & (e3["false_alarm"] == row["false_alarm"]) & (e3["recall"] == row["recall"])]
        compare(f"fig15_montecarlo ({row['panel']})", row, g.iloc[0], (metric,))

S = json.load(open(os.path.join(RES, "clean500_summary.json"), encoding="utf-8"))
desc = {(r["model"], r["method"], r["metric"]): r for r in S["descriptive"]}
tests = {(t["hypothesis"], t["model"], t["comparator"]): t for t in S["confirmatory"]}
for _, row in pd.read_csv(os.path.join(HERE, "data", "fig16_clean500.csv")).iterrows():
    if row["panel"] in ("H1", "H2"):
        compare("fig16_clean500", row, tests[(row["panel"], row["model"], row["comparator"])],
                ("mean_diff", "ci_low", "ci_high", "p_holm"))
    else:
        compare("fig16_clean500", row, desc[(row["model"], row["method"], row["panel"])], ("mean", "ci_low", "ci_high"))

print(f"\nTraceability check: {'PASS' if bad == 0 else f'{bad} problems'} ({len(scripts)} figures)")
sys.exit(1 if bad else 0)
