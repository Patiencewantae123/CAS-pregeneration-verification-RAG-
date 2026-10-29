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
import platform
import re
import sys
import time
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
    """Map raw passages onto the CAS Document schema.

    Relevance is BM25 divided by the pool maximum. (An earlier version min-max normalised, which
    forced the weakest passage to 0 and, for two-passage pools, always discarded one of them.)
    """
    raw = bm25_scores(query, passages)
    hi = max(raw)
    rel = [s / hi if hi > 0 else 0.5 for s in raw]
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


_NLI = None


def nli_verifier():
    """Shared DeBERTa-v3 NLI cross-encoder with an on-disk prediction cache."""
    global _NLI
    if _NLI is None:
        from services.nli_model_verifier import NLIModelVerifier
        os.makedirs(OUT_RESULTS, exist_ok=True)
        _NLI = NLIModelVerifier(cache_path=os.path.join(OUT_RESULTS, "nli_cache.jsonl"))
    return _NLI


def verifier_for(kind):
    if kind == "nli":
        return nli_verifier()
    if kind == "nli_cv":
        return nli_verifier().with_threshold(CV_THETA["penalty"]["__all__"])
    return NLIVerifier()


def select_cas(query, passages, tau=DEFAULT_TAU, use_graph=True, verifier="keyword", **_):
    docs, _ = to_documents(query, passages)
    edges = verifier_for(verifier).build_pairwise_matrix(docs) if use_graph else []
    graph = ConflictGraph(docs, edges)
    verified = TrustAwareSynthesizer(tau).filter_and_weight(graph)
    return {int(v.doc.id) for v in verified if not v.is_filtered}


def cas_preserving(query, passages, tau=DEFAULT_TAU):
    """Conflict-preserving CAS with the NLI model: returns (selected ids, contradicting id pairs)."""
    docs, _ = to_documents(query, passages)
    graph = ConflictGraph(docs, nli_verifier().build_pairwise_matrix(docs))
    verified, conflicts = TrustAwareSynthesizer(tau).filter_preserving_conflicts(graph)
    return {int(v.doc.id) for v in verified if not v.is_filtered}, [(int(a), int(b)) for a, b in conflicts]


def select_cas_preserving(query, passages, tau=DEFAULT_TAU, **_):
    return cas_preserving(query, passages, tau)[0]

# ------------------------------------------------------------
# Contradiction threshold theta, chosen by 5-fold cross-validation (see run_cv)
# ------------------------------------------------------------

THETA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
N_FOLDS = 5
CV_PATH = os.path.join(OUT_RESULTS, "cv_thresholds.json")
CV_THETA = {}   # kind -> {question: theta chosen on the other folds, "__all__": theta fit on all questions}
if os.path.exists(CV_PATH):
    with open(CV_PATH, encoding="utf-8") as _f:
        CV_THETA = json.load(_f)


def cas_nli_theta(query, passages, theta, preserve, tau=DEFAULT_TAU):
    docs, _ = to_documents(query, passages)
    graph = ConflictGraph(docs, nli_verifier().with_threshold(theta).build_pairwise_matrix(docs))
    synthesizer = TrustAwareSynthesizer(tau)
    if preserve:
        verified, conflicts = synthesizer.filter_preserving_conflicts(graph)
    else:
        verified, conflicts = synthesizer.filter_and_weight(graph), []
    return {int(v.doc.id) for v in verified if not v.is_filtered}, [(int(a), int(b)) for a, b in conflicts]


def theta_for(kind, query):
    """Held-out theta for a tuning question; the all-data theta for questions never used in tuning."""
    table = CV_THETA.get(kind, {})
    return table.get(query, table.get("__all__", 0.5))


def select_cas_nli_cv(query, passages, tau=DEFAULT_TAU, **_):
    return cas_nli_theta(query, passages, theta_for("penalty", query), False, tau)[0]


def select_cas_preserving_cv(query, passages, tau=DEFAULT_TAU, **_):
    return cas_nli_theta(query, passages, theta_for("preserve", query), True, tau)[0]


def cas_preserving_cv(query, passages, tau=DEFAULT_TAU):
    return cas_nli_theta(query, passages, theta_for("preserve", query), True, tau)


METHODS = {
    "Vanilla RAG (no filter)": select_vanilla,
    f"Relevance top-{TOP_K} (BM25)": select_topk,
    "CAS w/o conflict graph": lambda q, p, **kw: select_cas(q, p, use_graph=False, **kw),
    "CAS (full pipeline)": select_cas,
    "CAS-NLI (full pipeline)": lambda q, p, **kw: select_cas(q, p, verifier="nli", **kw),
    "CAS-NLI conflict-preserving": select_cas_preserving,
    "CAS-NLI θ-CV (penalty)": select_cas_nli_cv,
    "CAS-NLI θ-CV preserving": select_cas_preserving_cv,
}
NLI_METHODS = {"CAS-NLI (full pipeline)", "CAS-NLI conflict-preserving",
               "CAS-NLI θ-CV (penalty)", "CAS-NLI θ-CV preserving"}

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
    return float(auroc(scores, labels)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

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


def paired_test(a, b, n=10000, seed=SEED):
    """Mean of (a - b), bootstrap 95% CI, and two-sided paired sign-flip permutation p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    mean, lo, hi = boot_ci(d)
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    p = float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))
    return mean, lo, hi, p


def graph_diagnostics(items):
    """How often the conflict graph emits each relation, and who receives agreement credit."""
    rows = []
    for verifier in ("keyword", "nli", "nli_cv"):
        for ds, data in items.items():
            rel, agree, penalty = Counter(), {"gold": [], "distractor": []}, {"gold": [], "distractor": []}
            for q, passages, gold, _ in data:
                docs, _ = to_documents(q, passages)
                edges = verifier_for(verifier).build_pairwise_matrix(docs)
                rel.update(e.relation.value for e in edges)
                graph = ConflictGraph(docs, edges)
                for d in docs:
                    kind = "gold" if int(d.id) in gold else "distractor"
                    agree[kind].append(min(1.0, graph.get_agreement_score(d.id)))
                    penalty[kind].append(graph.get_contradiction_penalty(d.id))
            rows.append({"verifier": verifier, "dataset": ds, "pairs": sum(rel.values()),
                         "entailment_edges": rel["Entailment"], "contradiction_edges": rel["Contradiction"],
                         "neutral_edges": rel["Neutral"],
                         "mean_agreement_gold": float(np.mean(agree["gold"])),
                         "mean_agreement_distractor": float(np.mean(agree["distractor"])),
                         "mean_penalty_gold": float(np.mean(penalty["gold"])),
                         "mean_penalty_distractor": float(np.mean(penalty["distractor"]))})
    return rows


def run_evidence_selection(items):
    rows, per_item = [], {}
    for ds, data in items.items():
        for method, fn in METHODS.items():
            per = {"precision": [], "recall": [], "f1": [], "retention": [], "answer_coverage": []}
            start = time.perf_counter()
            for q, passages, gold, answers in data:
                sel = fn(q, passages)
                p, r, f = prf(sel, gold)
                per["precision"].append(p)
                per["recall"].append(r)
                per["f1"].append(f)
                per["retention"].append(len(sel) / len(passages))
                if answers:
                    per["answer_coverage"].append(answer_covered(sel, passages, answers))
            ms_per_query = 1000 * (time.perf_counter() - start) / len(data)
            per_item[(ds, method)] = per
            for metric, vals in per.items():
                m, lo, hi = boot_ci(vals)
                rows.append({"dataset": ds, "method": method, "metric": metric,
                             "mean": m, "ci_low": lo, "ci_high": hi, "n": len(vals),
                             "ms_per_query": ms_per_query})
    return rows, per_item


def run_timing(items, repeats=20):
    """Wall-clock latency of each selection method, repeated to estimate run-to-run variability."""
    rows = []
    for ds, data in items.items():
        for method, fn in METHODS.items():
            runs = []
            for _ in range(repeats):
                start = time.perf_counter()
                for q, passages, _, _ in data:
                    fn(q, passages)
                runs.append(1000 * (time.perf_counter() - start) / len(data))
            # Repeats of NLI methods hit the prediction cache, so add the model time measured at first computation
            model_ms = 0.0
            if method in NLI_METHODS:
                model_ms = 1000 * float(np.mean([
                    nli_verifier().inference_seconds(nli_verifier().document_pairs(to_documents(q, p)[0]))
                    for q, p, _, _ in data]))
            rows.append({"dataset": ds, "method": method, "repeats": repeats,
                         "ms_per_query_mean": float(np.mean(runs)) + model_ms,
                         "ms_per_query_sd": float(np.std(runs, ddof=1)),
                         "nli_model_ms_per_query": model_ms,
                         "candidates_per_query": len(data[0][1])})
    return rows


def run_error_analysis(items, tau=DEFAULT_TAU):
    """Attribute every full-pipeline decision on a passage to a cause, using the pipeline's own outputs."""
    rows = []
    for verifier, name in (("keyword", "CAS (full pipeline)"), ("nli", "CAS-NLI (full pipeline)")):
        for ds, data in items.items():
            c = Counter()
            for q, passages, gold, _ in data:
                full = select_cas(q, passages, tau=tau, verifier=verifier)
                no_graph = select_cas(q, passages, tau=tau, use_graph=False)
                top3 = select_topk(q, passages)
                for i in range(len(passages)):
                    if i in gold:
                        if i in full:
                            c["gold: retained"] += 1
                        elif i in no_graph:
                            c["gold: removed by contradiction penalty"] += 1
                        elif i in top3:
                            c["gold: missed, in BM25 top-3 but trust < tau"] += 1
                        else:
                            c["gold: missed, low lexical relevance (BM25 rank > 3)"] += 1
                    elif i in full:
                        key = "distractor: retained by relevance alone" if i in no_graph \
                            else "distractor: retained only via agreement bonus"
                        c[key] += 1
                    elif i in no_graph:
                        c["distractor: removed by contradiction penalty"] += 1
                    else:
                        c["distractor: filtered"] += 1
            for cat, n in sorted(c.items()):
                rows.append({"pipeline": name, "dataset": ds, "category": cat, "count": n})
    return rows


REFERENCES = ["CAS (full pipeline)", "CAS-NLI (full pipeline)", "CAS-NLI conflict-preserving",
              "CAS-NLI θ-CV (penalty)", "CAS-NLI θ-CV preserving"]


def run_cv(items):
    """5-fold CV for theta: each question's theta is chosen on the other folds (mean evidence F1).

    Folds are stratified by dataset (seeded shuffle). Ties go to the larger (stricter) theta.
    "__all__" is theta fit on every question; it is used only for data never seen in tuning (WikiContradict).
    """
    pooled = [(ds, q, p, g) for ds, data in items.items() for q, p, g, _ in data]
    rng = random.Random(SEED)
    fold = {}
    for ds in items:
        idx = [i for i, row in enumerate(pooled) if row[0] == ds]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            fold[i] = k % N_FOLDS
    rows, table = [], {}
    for kind, preserve in (("penalty", False), ("preserve", True)):
        f1 = {th: [prf(cas_nli_theta(q, p, th, preserve)[0], g)[2] for _, q, p, g in pooled] for th in THETA_GRID}
        for th in THETA_GRID:
            rows.append({"kind": kind, "fold": "all", "theta": th, "mean_f1": float(np.mean(f1[th]))})
        best = lambda ids: max(THETA_GRID, key=lambda th: (round(float(np.mean([f1[th][i] for i in ids])), 12), th))
        table[kind] = {"__all__": best(range(len(pooled)))}
        for k in range(N_FOLDS):
            train = [i for i in range(len(pooled)) if fold[i] != k]
            test = [i for i in range(len(pooled)) if fold[i] == k]
            th = best(train)
            for i in test:
                table[kind][pooled[i][1]] = th
            rows.append({"kind": kind, "fold": k, "theta": th,
                         "train_f1": float(np.mean([f1[th][i] for i in train])),
                         "heldout_f1": float(np.mean([f1[th][i] for i in test])), "n_test": len(test)})
    CV_THETA.clear()
    CV_THETA.update(table)
    with open(CV_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    return rows


def run_paired_tests(per_item):
    """Each CAS variant against every other method; one Holm family over all tests."""
    rows, seen = [], set()
    for ds in ("HotpotQA", "MuSiQue"):
        for ref in REFERENCES:
            for other in METHODS:
                if other == ref or (ds, frozenset((ref, other))) in seen:
                    continue
                seen.add((ds, frozenset((ref, other))))
                for metric in ("precision", "recall", "f1"):
                    m, lo, hi, p = paired_test(per_item[(ds, ref)][metric], per_item[(ds, other)][metric])
                    rows.append({"dataset": ds, "reference": ref, "comparator": other, "metric": metric,
                                 "mean_diff": m, "ci_low": lo, "ci_high": hi, "p_value": p})
    # Holm-Bonferroni step-down adjustment across the whole family of tests
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_value"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[i]["p_value"]))
        rows[i]["p_holm"] = running
    return rows


def run_threshold_sweep(items):
    taus = np.round(np.arange(0.10, 0.81, 0.05), 2)
    pooled = items["HotpotQA"] + items["MuSiQue"]
    rows = []
    for method in ("CAS (full pipeline)", "CAS-NLI (full pipeline)", "CAS-NLI conflict-preserving"):
        for tau in taus:
            per = {"precision": [], "recall": [], "f1": [], "retention": []}
            for q, passages, gold, _ in pooled:
                sel = METHODS[method](q, passages, tau=float(tau))
                p, r, f = prf(sel, gold)
                per["precision"].append(p)
                per["recall"].append(r)
                per["f1"].append(f)
                per["retention"].append(len(sel) / len(passages))
            rows.append({"method": method, "tau": float(tau), **{k: float(np.mean(v)) for k, v in per.items()}})
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


def cas_relation(a, b, verifier="keyword"):
    return verifier_for(verifier).compare_documents(
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


CV_DETECTOR = []


def cv_detector(pairs):
    """Contradiction predictions with theta chosen by 5-fold CV (label-stratified) on pooled-pair F1."""
    def predict(th, a, b):
        doc_a, doc_b = Document("a", a, "", 0, 0, 0), Document("b", b, "", 0, 0, 0)
        return int(nli_verifier().with_threshold(th).compare_documents(doc_a, doc_b).relation
                   == NLIRelation.CONTRADICTION)
    preds = {th: [predict(th, a, b) for a, b, _, _ in pairs] for th in THETA_GRID}
    y = [p[2] for p in pairs]
    rng = random.Random(SEED)
    fold = {}
    for label in (0, 1):
        idx = [i for i, v in enumerate(y) if v == label]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            fold[i] = k % N_FOLDS
    out = [0] * len(pairs)
    CV_DETECTOR.clear()
    for k in range(N_FOLDS):
        train = [i for i in fold if fold[i] != k]
        th = max(THETA_GRID, key=lambda t: (round(binary_prf([y[i] for i in train],
                                                             [preds[t][i] for i in train])[2], 12), t))
        CV_DETECTOR.append({"fold": k, "theta": th})
        for i in fold:
            if fold[i] == k:
                out[i] = preds[th][i]
    return out


def run_contradiction(pairs):
    y = [p[2] for p in pairs]
    detectors = {
        "Negation-cue baseline": [negation_baseline(a, b) for a, b, _, _ in pairs],
        "CAS NLIVerifier (as implemented)": [int(cas_relation(a, b) == NLIRelation.CONTRADICTION)
                                             for a, b, _, _ in pairs],
        "DeBERTa-v3 NLI (CAS-NLI)": [int(cas_relation(a, b, "nli") == NLIRelation.CONTRADICTION)
                                     for a, b, _, _ in pairs],
        "DeBERTa-v3 NLI θ-CV": cv_detector(pairs),
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
    out = {}
    keyword = [to_label[cas_relation(r["premise"], r["hypothesis"])] for r in data]
    # Standard single-direction NLI (premise -> hypothesis) for the model
    probs = nli_verifier().probabilities([(r["premise"], r["hypothesis"]) for r in data])
    model = [{"entailment": 0, "neutral": 1, "contradiction": 2}[max(p, key=p.get)] for p in probs]
    for name, pred in (("CAS NLIVerifier (as implemented)", keyword), ("DeBERTa-v3 NLI (CAS-NLI)", model)):
        out[name] = _nli_scores(y, pred, names)
    return out


def _nli_scores(y, pred, names):
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

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]


def nli_unsupported_score(context, output, window=3):
    """1 - support of the least-supported output sentence, where support is the max entailment
    probability over 3-sentence context windows (the context is too long for one 512-token pass)."""
    ctx = sentences(context)
    chunks = [" ".join(ctx[i:i + window]) for i in range(0, max(1, len(ctx)), window)] or [context]
    outs = sentences(output) or [output]
    probs = nli_verifier().probabilities([(c, s) for s in outs for c in chunks])
    support = [max(p["entailment"] for p in probs[k * len(chunks):(k + 1) * len(chunks)]) for k in range(len(outs))]
    return 1 - min(support)


def run_hallucination():
    data = load("ragtruth_hf_samples.json")
    labels, unsupported, cas_flag, nli_score = [], [], [], []
    for r in data:
        hl = r["hallucination_labels_processed"]
        labels.append(int(sum(hl.values()) > 0))
        ctx, out = set(tokens(r["context"])), tokens(r["output"])
        unsupported.append(sum(t not in ctx for t in out) / max(1, len(out)))
        rel = cas_relation(r["context"], r["output"])
        cas_flag.append({NLIRelation.CONTRADICTION: 1.0, NLIRelation.NEUTRAL: 0.5,
                         NLIRelation.ENTAILMENT: 0.0}[rel])
        nli_score.append(nli_unsupported_score(r["context"], r["output"]))
    return {
        "n": len(labels), "n_hallucinated": sum(labels),
        "unsupported_token_rate_auroc": boot_auroc(unsupported, labels),
        "cas_verifier_auroc": boot_auroc(cas_flag, labels),
        "nli_model_auroc": boot_auroc(nli_score, labels),
        "cas_relation_counts": dict(Counter(cas_flag)),
        "scores": [{"label": l, "unsupported_token_rate": u, "cas_score": c, "nli_score": s}
                   for l, u, c, s in zip(labels, unsupported, cas_flag, nli_score)],
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

    t0 = time.perf_counter()
    items = evidence_items()
    cv = run_cv(items)            # must run first: the θ-CV methods read the thresholds it stores
    evidence, per_item = run_evidence_selection(items)
    paired = run_paired_tests(per_item)
    diagnostics = graph_diagnostics(items)
    timing = run_timing(items)
    errors = run_error_analysis(items)
    sweep = run_threshold_sweep(items)
    contra, contra_groups = run_contradiction(contradiction_pairs())
    nli = run_nli()
    halluc = run_hallucination()
    inv = inventory()

    runtime_s = time.perf_counter() - t0
    write_csv(os.path.join(OUT_RESULTS, "evidence_selection.csv"), evidence)
    write_csv(os.path.join(OUT_RESULTS, "paired_tests.csv"), paired)
    write_csv(os.path.join(OUT_RESULTS, "graph_diagnostics.csv"), diagnostics)
    write_csv(os.path.join(OUT_RESULTS, "timing.csv"), timing)
    write_csv(os.path.join(OUT_RESULTS, "error_analysis.csv"), errors)
    write_csv(os.path.join(OUT_RESULTS, "threshold_sweep.csv"), sweep)
    write_csv(os.path.join(OUT_RESULTS, "contradiction_detection.csv"), contra)
    write_csv(os.path.join(OUT_RESULTS, "contradiction_by_group.csv"), contra_groups)
    write_csv(os.path.join(OUT_RESULTS, "dataset_inventory.csv"), inv)
    write_csv(os.path.join(OUT_RESULTS, "cv_threshold.csv"), cv)
    with open(os.path.join(OUT_RESULTS, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": {"seed": SEED, "bootstrap": N_BOOT, "permutations": 10000, "tau": DEFAULT_TAU,
                              "top_k": TOP_K, "provenance_prior": PROVENANCE,
                              "python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "processor": platform.processor(),
                              "total_runtime_s": runtime_s},
                   "evidence_selection": evidence, "paired_tests": paired,
                   "graph_diagnostics": diagnostics, "timing": timing, "error_analysis": errors,
                   "threshold_sweep": sweep,
                   "contradiction_detection": contra, "contradiction_by_group": contra_groups,
                   "nli": nli, "hallucination": halluc, "inventory": inv,
                   "cv_threshold": {"grid": THETA_GRID, "folds": N_FOLDS, "rows": cv,
                                    "theta_all": {k: v["__all__"] for k, v in CV_THETA.items()},
                                    "detector_folds": CV_DETECTOR}}, f, indent=2)

    print("Cross-validated contradiction threshold")
    for r in cv:
        print("  ", r)
    print("  detector folds:", CV_DETECTOR)
    print("Evidence selection (mean [95% CI])")
    for r in evidence:
        if r["metric"] in ("precision", "recall", "f1", "retention"):
            print(f"  {r['dataset']:<9} {r['method']:<28} {r['metric']:<10} "
                  f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print("\nPaired tests (reference minus comparator)")
    for r in paired:
        print(f"  {r['dataset']:<9} {r['reference']:<28} - {r['comparator']:<28} {r['metric']:<9} "
              f"{r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p_value']:.4f} holm={r['p_holm']:.4f}")
    print("\nConflict-graph diagnostics")
    for r in diagnostics:
        print("  ", r)
    print("\nRuntime ms/query")
    for r in evidence:
        if r["metric"] == "f1":
            print(f"  {r['dataset']:<9} {r['method']:<28} {r['ms_per_query']:.2f}")
    print(f"\nTotal runtime {runtime_s:.1f}s")
    print("\nContradiction detection")
    for r in contra:
        print(f"  {r['detector']:<34} {r['metric']:<9} {r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    for r in contra_groups:
        print(f"  {r['detector']:<34} {r['group']:<26} flag-rate {r['flag_rate']:.3f} (n={r['n']})")
    for name, s in nli.items():
        print(f"\nMultiNLI 3-way [{name}]: acc {s['accuracy'][0]:.3f} [{s['accuracy'][1]:.3f}, {s['accuracy'][2]:.3f}], "
              f"macro-F1 {s['macro_f1']:.3f}, majority {s['majority_baseline_accuracy'][0]:.3f}; confusion {s['confusion']}")
    print(f"\nRAGTruth AUROC unsupported-token {halluc['unsupported_token_rate_auroc']}, "
          f"CAS verifier {halluc['cas_verifier_auroc']}, NLI model {halluc['nli_model_auroc']}")
    print("\nThreshold sweep")
    for r in sweep:
        print(f"  {r['method']:<28} tau={r['tau']:.2f} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} keep={r['retention']:.3f}")
    print("\nError analysis")
    for r in errors:
        print("  ", r)


if __name__ == "__main__":
    main()
