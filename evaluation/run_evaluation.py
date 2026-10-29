"""
Reproducible evaluation of the CAS pre-generation verification layer.

Every number written by this script is measured from the datasets in ../data
using the CAS implementation in ../python. Nothing is hardcoded.

Tasks
  1. Evidence selection   HotpotQA (n=50), MuSiQue (n=15 with >=1 gold paragraph)
  2. Contradiction detection  WikiContradict pairs (+), HotpotQA gold pairs (-), MultiNLI
  3. 3-way NLI            MultiNLI (n=20)
  4. Hallucination detection  RAGTruth summaries (n=50, 12 hallucinated)
  5. Threshold sensitivity of the CAS trust threshold (tau)

Usage:  py evaluation/run_evaluation.py
Output: evaluation/results/*.csv, results.json   evaluation/figures/*.png|pdf
"""
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_RESULTS = os.path.join(ROOT, "evaluation", "results")
OUT_FIGS = os.path.join(ROOT, "evaluation", "figures")
sys.path.insert(0, os.path.join(ROOT, "python"))

from models.document import Document  # noqa: E402
from models.nli_edge import NLIRelation  # noqa: E402
from services.nli_verifier import NLIVerifier  # noqa: E402
from services.conflict_graph import ConflictGraph  # noqa: E402
from services.trust_synthesizer import TrustAwareSynthesizer  # noqa: E402

SEED = 42
N_BOOT = 2000
DEFAULT_TAU = 0.45      # CASEngine default
TOP_K = 3               # relevance-only baseline
PROVENANCE = 0.5        # the public benchmarks carry no source-reliability metadata

# ============================================================
# 1. LOADING
# ============================================================

def load(name):
    """Load a JSON array or a JSON Lines file (the *_hf_samples files are JSONL)."""
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


STOP = set("""a an the of in on at to for from by with and or but is are was were be been being
this that these those it its as which who whom whose what when where why how do does did
has have had not no yes than then there their they he she his her him we you i""".split())


def tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


def norm_answer(s):
    return " ".join(re.findall(r"\w+", str(s).lower()))

# ============================================================
# 2. RETRIEVAL SCORING (BM25 over each question's candidate pool)
# ============================================================

def bm25_scores(query, passages, k1=1.5, b=0.75):
    docs = [tokens(p) for p in passages]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1)
    df = Counter(t for d in docs for t in set(d))
    q = tokens(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores

# ============================================================
# 3. SELECTION METHODS
# ============================================================

def to_documents(query, passages):
    """Map raw passages onto the CAS Document schema."""
    raw = bm25_scores(query, passages)
    lo, hi = min(raw), max(raw)
    rel = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in raw]
    order = sorted(range(len(passages)), key=lambda i: -raw[i])
    rank = {i: r for r, i in enumerate(order)}
    n = len(passages)
    return [
        Document(id=str(i), content=p, source="benchmark",
                 retrieval_confidence=1 - rank[i] / n,
                 relevance_score=rel[i], provenance_score=PROVENANCE)
        for i, p in enumerate(passages)
    ], raw


def select_vanilla(query, passages, **_):
    return set(range(len(passages)))


def select_topk(query, passages, k=TOP_K, **_):
    raw = bm25_scores(query, passages)
    return set(sorted(range(len(passages)), key=lambda i: -raw[i])[:k])


def select_cas(query, passages, tau=DEFAULT_TAU, use_graph=True, **_):
    docs, _ = to_documents(query, passages)
    edges = NLIVerifier().build_pairwise_matrix(docs) if use_graph else []
    graph = ConflictGraph(docs, edges)
    verified = TrustAwareSynthesizer(tau).filter_and_weight(graph)
    return {int(v.doc.id) for v in verified if not v.is_filtered}


METHODS = {
    "Vanilla RAG (no filter)": select_vanilla,
    f"Relevance top-{TOP_K} (BM25)": select_topk,
    "CAS w/o conflict graph": lambda q, p, **kw: select_cas(q, p, use_graph=False, **kw),
    "CAS (full pipeline)": select_cas,
}

# ============================================================
# 4. STATISTICS
# ============================================================

def boot_ci(values, stat=np.mean, n=N_BOOT, seed=SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    return float(stat(values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def binary_prf(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, float((y_true == y_pred).mean())


def boot_binary(y_true, y_pred, n=N_BOOT, seed=SEED):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    point = binary_prf(y_true, y_pred)
    samples = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        samples.append(binary_prf(y_true[i], y_pred[i]))
    samples = np.array(samples)
    lo, hi = np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)
    names = ["precision", "recall", "f1", "accuracy"]
    return {m: (point[j], float(lo[j]), float(hi[j])) for j, m in enumerate(names)}


def auroc(scores, labels):
    """Mann-Whitney U formulation, ties counted as 0.5."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def boot_auroc(scores, labels, n=N_BOOT, seed=SEED):
    scores, labels = np.asarray(scores), np.asarray(labels)
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        i = rng.integers(0, len(scores), len(scores))
        if 0 < labels[i].sum() < len(i):
            vals.append(auroc(scores[i], labels[i]))
    return auroc(scores, labels), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

# ============================================================
# 5. TASK 1 - EVIDENCE SELECTION
# ============================================================

def evidence_items():
    items = {"HotpotQA": [], "MuSiQue": []}
    for r in load("hotpotqa_hf_samples.json"):
        titles = r["context"]["title"]
        passages = [f"{t}. {' '.join(s)}" for t, s in zip(titles, r["context"]["sentences"])]
        gold = {i for i, t in enumerate(titles) if t in set(r["supporting_facts"]["title"])}
        answers = [] if norm_answer(r["answer"]) in ("yes", "no") else [r["answer"]]
        items["HotpotQA"].append((r["question"], passages, gold, answers))
    for r in load("musique.json"):
        gold = {i for i, p in enumerate(r["paragraphs"]) if p["is_supporting"]}
        if not gold:
            continue  # unanswerable instances have no gold evidence to score against
        passages = [f"{p['title']}. {p['paragraph_text']}" for p in r["paragraphs"]]
        items["MuSiQue"].append((r["question"], passages, gold, [r["answer"]] + r["answer_aliases"]))
    return items


def answer_covered(selected, passages, answers):
    text = norm_answer(" ".join(passages[i] for i in selected))
    return float(any(norm_answer(a) and norm_answer(a) in text for a in answers))


def run_evidence_selection(items):
    rows = []
    for ds, data in items.items():
        for method, fn in METHODS.items():
            per = {"precision": [], "recall": [], "f1": [], "retention": [], "answer_coverage": []}
            for q, passages, gold, answers in data:
                sel = fn(q, passages)
                p, r, f = prf(sel, gold)
                per["precision"].append(p)
                per["recall"].append(r)
                per["f1"].append(f)
                per["retention"].append(len(sel) / len(passages))
                if answers:
                    per["answer_coverage"].append(answer_covered(sel, passages, answers))
            for metric, vals in per.items():
                m, lo, hi = boot_ci(vals)
                rows.append({"dataset": ds, "method": method, "metric": metric,
                             "mean": m, "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return rows


def run_threshold_sweep(items):
    taus = np.round(np.arange(0.10, 0.81, 0.05), 2)
    pooled = items["HotpotQA"] + items["MuSiQue"]
    rows = []
    for tau in taus:
        per = {"precision": [], "recall": [], "f1": [], "retention": []}
        for q, passages, gold, _ in pooled:
            sel = select_cas(q, passages, tau=float(tau))
            p, r, f = prf(sel, gold)
            per["precision"].append(p)
            per["recall"].append(r)
            per["f1"].append(f)
            per["retention"].append(len(sel) / len(passages))
        rows.append({"tau": float(tau), **{k: float(np.mean(v)) for k, v in per.items()}})
    return rows

# ============================================================
# 6. TASKS 2 & 3 - CONTRADICTION DETECTION AND NLI
# ============================================================

NEGATIONS = {"not", "no", "never", "none", "neither", "nor", "cannot", "without", "false", "n't"}


def negation_baseline(a, b):
    """Lexical baseline: flag a contradiction when one side is negated and the two overlap."""
    ta, tb = set(re.findall(r"\w+|n't", a.lower())), set(re.findall(r"\w+|n't", b.lower()))
    overlap = len(set(tokens(a)) & set(tokens(b))) / max(1, min(len(set(tokens(a))), len(set(tokens(b)))))
    return int(bool(ta & NEGATIONS) != bool(tb & NEGATIONS) and overlap >= 0.2)


def cas_relation(a, b):
    return NLIVerifier().compare_documents(
        Document("a", a, "", 0, 0, 0), Document("b", b, "", 0, 0, 0)).relation


def contradiction_pairs():
    pairs = []  # (text_a, text_b, label, group)
    for r in load("wikicontradict_hf_samples.json"):
        pairs.append((r["context1"], r["context2"], 1, f"WikiContradict: {r['contradictType'].split(' ')[0]}"))
    for r in load("hotpotqa_hf_samples.json"):
        gold = list(dict.fromkeys(r["supporting_facts"]["title"]))[:2]
        lookup = dict(zip(r["context"]["title"], r["context"]["sentences"]))
        if len(gold) == 2:
            pairs.append((" ".join(lookup[gold[0]]), " ".join(lookup[gold[1]]), 0, "HotpotQA gold pair"))
    for r in load("multinli.json"):
        pairs.append((r["premise"], r["hypothesis"], int(r["label"] == 2), "MultiNLI"))
    return pairs


def run_contradiction(pairs):
    y = [p[2] for p in pairs]
    detectors = {
        "Negation-cue baseline": [negation_baseline(a, b) for a, b, _, _ in pairs],
        "CAS NLIVerifier (as implemented)": [int(cas_relation(a, b) == NLIRelation.CONTRADICTION)
                                             for a, b, _, _ in pairs],
    }
    rows, by_group = [], []
    for name, pred in detectors.items():
        for metric, (m, lo, hi) in boot_binary(y, pred).items():
            rows.append({"detector": name, "metric": metric, "mean": m, "ci_low": lo, "ci_high": hi, "n": len(y)})
        for g in sorted({p[3] for p in pairs}):
            idx = [i for i, p in enumerate(pairs) if p[3] == g]
            lab = pairs[idx[0]][2] if len({pairs[i][2] for i in idx}) == 1 else None
            flagged = float(np.mean([pred[i] for i in idx]))
            by_group.append({"detector": name, "group": g, "n": len(idx),
                             "flag_rate": flagged, "all_label": lab})
    return rows, by_group


def run_nli():
    data = load("multinli.json")
    names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    to_label = {NLIRelation.ENTAILMENT: 0, NLIRelation.NEUTRAL: 1, NLIRelation.CONTRADICTION: 2}
    y = [r["label"] for r in data]
    pred = [to_label[cas_relation(r["premise"], r["hypothesis"])] for r in data]
    majority = Counter(y).most_common(1)[0][0]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    f1s = []
    for c in range(3):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    acc = boot_ci([int(a == b) for a, b in zip(y, pred)])
    maj = boot_ci([int(a == majority) for a in y])
    return {"labels": [names[i] for i in range(3)], "confusion": cm.tolist(),
            "accuracy": acc, "macro_f1": float(np.mean(f1s)), "per_class_f1": f1s,
            "majority_baseline_accuracy": maj, "majority_class": names[majority], "n": len(y)}

# ============================================================
# 7. TASK 4 - HALLUCINATION DETECTION (RAGTruth)
# ============================================================

def run_hallucination():
    data = load("ragtruth_hf_samples.json")
    labels, unsupported, cas_flag = [], [], []
    for r in data:
        hl = r["hallucination_labels_processed"]
        labels.append(int(sum(hl.values()) > 0))
        ctx, out = set(tokens(r["context"])), tokens(r["output"])
        unsupported.append(sum(t not in ctx for t in out) / max(1, len(out)))
        rel = cas_relation(r["context"], r["output"])
        cas_flag.append({NLIRelation.CONTRADICTION: 1.0, NLIRelation.NEUTRAL: 0.5,
                         NLIRelation.ENTAILMENT: 0.0}[rel])
    return {
        "n": len(labels), "n_hallucinated": sum(labels),
        "unsupported_token_rate_auroc": boot_auroc(unsupported, labels),
        "cas_verifier_auroc": boot_auroc(cas_flag, labels),
        "cas_relation_counts": dict(Counter(cas_flag)),
        "scores": [{"label": l, "unsupported_token_rate": u, "cas_score": c}
                   for l, u, c in zip(labels, unsupported, cas_flag)],
    }

# ============================================================
# 8. DATASET INVENTORY
# ============================================================

def inventory():
    used = {
        "hotpotqa_hf_samples.json": "Evidence selection; consistent-pair negatives",
        "musique.json": "Evidence selection",
        "wikicontradict_hf_samples.json": "Contradiction detection (positives)",
        "multinli.json": "3-way NLI; contradiction detection",
        "ragtruth_hf_samples.json": "Hallucination detection",
    }
    rows = []
    for f in sorted(os.listdir(DATA)):
        if not f.endswith(".json") or f == "pyrightconfig.json":
            continue
        with open(os.path.join(DATA, f), encoding="utf-8-sig") as fh:
            text = fh.read()
        try:
            obj, fmt = json.loads(text), "JSON"
        except json.JSONDecodeError:
            obj, fmt = [l for l in text.splitlines() if l.strip()], "JSONL"
        n = len(obj) if isinstance(obj, list) else 1
        rows.append({"file": f, "format": fmt, "records": n, "used_for": used.get(f, "Not used (see notes)")})
    return rows

# ============================================================
# 9. MAIN
# ============================================================

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    random.seed(SEED)
    os.makedirs(OUT_RESULTS, exist_ok=True)

    items = evidence_items()
    evidence = run_evidence_selection(items)
    sweep = run_threshold_sweep(items)
    contra, contra_groups = run_contradiction(contradiction_pairs())
    nli = run_nli()
    halluc = run_hallucination()
    inv = inventory()

    write_csv(os.path.join(OUT_RESULTS, "evidence_selection.csv"), evidence)
    write_csv(os.path.join(OUT_RESULTS, "threshold_sweep.csv"), sweep)
    write_csv(os.path.join(OUT_RESULTS, "contradiction_detection.csv"), contra)
    write_csv(os.path.join(OUT_RESULTS, "contradiction_by_group.csv"), contra_groups)
    write_csv(os.path.join(OUT_RESULTS, "dataset_inventory.csv"), inv)
    with open(os.path.join(OUT_RESULTS, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": {"seed": SEED, "bootstrap": N_BOOT, "tau": DEFAULT_TAU, "top_k": TOP_K,
                              "provenance_prior": PROVENANCE},
                   "evidence_selection": evidence, "threshold_sweep": sweep,
                   "contradiction_detection": contra, "contradiction_by_group": contra_groups,
                   "nli": nli, "hallucination": halluc, "inventory": inv}, f, indent=2)

    print("Evidence selection (mean [95% CI])")
    for r in evidence:
        if r["metric"] in ("precision", "recall", "f1", "retention"):
            print(f"  {r['dataset']:<9} {r['method']:<28} {r['metric']:<10} "
                  f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print("\nContradiction detection")
    for r in contra:
        print(f"  {r['detector']:<34} {r['metric']:<9} {r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    for r in contra_groups:
        print(f"  {r['detector']:<34} {r['group']:<26} flag-rate {r['flag_rate']:.3f} (n={r['n']})")
    print(f"\nMultiNLI 3-way: acc {nli['accuracy'][0]:.3f}, macro-F1 {nli['macro_f1']:.3f}, "
          f"majority {nli['majority_baseline_accuracy'][0]:.3f}; confusion {nli['confusion']}")
    print(f"RAGTruth AUROC unsupported-token {halluc['unsupported_token_rate_auroc']}, "
          f"CAS verifier {halluc['cas_verifier_auroc']}, relations {halluc['cas_relation_counts']}")
    print("\nThreshold sweep")
    for r in sweep:
        print(f"  tau={r['tau']:.2f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} keep={r['retention']:.3f}")


if __name__ == "__main__":
    main()
