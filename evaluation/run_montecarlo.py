"""
Study 4: Monte Carlo sensitivity analysis of the CAS selection layer, run as pre-registered in
PREREGISTRATION_montecarlo.md (3,064,800 question-level runs, six methods each).

No LLM and no new model inference: BM25 scores and DeBERTa-v3 NLI predictions are the real ones from studies 1-2,
read from results/nli_cache.jsonl. The selection code is a vectorised re-implementation of run_poison.py and
python/services/, and it must reproduce the stored study 1-2 selections exactly before anything is simulated.

Usage:  py evaluation/run_montecarlo.py
Output: evaluation/results/montecarlo_summary.json, montecarlo_e1.csv, montecarlo_e2.csv, montecarlo_e3.csv
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_detector_probe import labelled_pairs  # noqa: E402
from run_evaluation import DEFAULT_TAU, OUT_RESULTS, SEED, TOP_K, bm25_scores, load  # noqa: E402
from run_llm_generation import normalize  # noqa: E402
from run_poison import PRIMARY, THETA  # noqa: E402
from models.nli_edge import NLIRelation  # noqa: E402
from services.nli_model_verifier import NLIModelVerifier  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREGISTRATION_montecarlo.md")
STUDIES = {"1": "poison", "2": "poison2"}
METHODS = ["Vanilla RAG", "BM25 top-3", "Provenance + BM25 top-3", "CAS w/o graph", "CAS-NLI penalty", PRIMARY]
ENDPOINTS = ["silent_exposure", "exposure", "answer_available", "n_kept"]
LOWER_IS_BETTER = {"silent_exposure": True, "exposure": True, "answer_available": False}
COMPARATORS = {"A": ["Vanilla RAG", "BM25 top-3"], "B": ["Vanilla RAG", "BM25 top-3", "Provenance + BM25 top-3"]}
BEST_BASELINE = {"A": "BM25 top-3", "B": "Provenance + BM25 top-3"}
DEFAULT_W = (0.35, 0.25, 0.20, 0.20)

N_SETTINGS = 1250                                   # E1
SEPARATIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # E2
FLIPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
E2_DRAWS = 60
RECALLS = [round(0.1 * k, 1) for k in range(11)]    # E3
FALSE_ALARMS = [0.0, 0.10, 0.25, 0.50]
E3_DRAWS = 30
SYNTH_CONF = 0.99
N_BOOT = 2000

# ============================================================
# 1. FIXED PER-QUESTION INPUTS (real BM25 and real NLI)
# ============================================================

def text_pairs(passages):
    n = len(passages)
    return [(passages[a], passages[b]) for i in range(n) for j in range(i + 1, n) for a, b in ((i, j), (j, i))]


def cached_nli(items):
    """Real DeBERTa-v3 probabilities for every passage pair; fails rather than loading the model."""
    v = NLIModelVerifier()
    need = {v._key(a, b) for it in items for a, b in text_pairs(it["passages"])}
    found = {}
    with open(os.path.join(OUT_RESULTS, "nli_cache.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:   # another run may be appending to the cache
                continue
            if rec["key"] in need:
                found[rec["key"]] = rec["probs"]
    missing = need - found.keys()
    if missing:
        raise SystemExit(f"{len(missing)} NLI predictions are missing from the cache; run studies 1-2 first")
    return lambda a, b: found[v._key(a, b)]


class Question:
    """Everything about one question that stays fixed across draws."""

    def __init__(self, it, study, nli):
        p, n = it["passages"], len(it["passages"])
        self.id, self.study, self.n = it["id"], study, n
        self.raw = np.array(bm25_scores(it["question"], p), dtype=float)
        hi = self.raw.max()
        rank = np.empty(n)
        rank[sorted(range(n), key=lambda i: -self.raw[i])] = np.arange(n)
        self.rel = self.raw / hi if hi > 0 else np.full(n, 0.5)
        self.conf = 1 - rank / n
        self.order = np.argsort(-self.raw, kind="stable")
        self.poison = np.isin(np.arange(n), it["poison"])
        self.answer = np.array([not self.poison[i] and normalize(it["answer"]) in normalize(p[i]) for i in range(n)])
        self.provenance_B = np.array(it["provenance_B"], dtype=float)
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        index = {pair: k for k, pair in enumerate(self.pairs)}
        verifier = NLIModelVerifier(contradiction_threshold=THETA)
        rel_conf = [verifier._symmetric(nli(p[i], p[j]), nli(p[j], p[i])) for i, j in self.pairs]
        self.real_contra = np.array([r == NLIRelation.CONTRADICTION for r, _ in rel_conf])
        self.real_ent = np.array([r == NLIRelation.ENTAILMENT for r, _ in rel_conf])
        self.edge_conf = np.array([c for _, c in rel_conf])
        labelled = labelled_pairs(it)
        self.true_pairs = [index[(min(a, b), max(a, b))] for a, b, kind in labelled if kind == "conflict"]
        self.clean_pairs = [index[(a, b)] for a, b, kind in labelled if kind == "clean"]

# ============================================================
# 2. SELECTION, VECTORISED OVER DRAWS
# ============================================================

def select(q, prov, w=DEFAULT_W, tau=DEFAULT_TAU, contra=None):
    """Kept and conflict-marked passages of every method on D draws.

    prov: (D, n). w: a 4-tuple or (D, 4). tau: a scalar or (D,). contra: None for the real detector, or a (D, P)
    boolean synthetic contradiction mask. Returns ({method: (kept, marked)}, reported conflict pairs (D, P)).
    """
    D, n = prov.shape
    none = np.zeros((D, n), bool)
    out = {"Vanilla RAG": (np.ones((D, n), bool), none)}
    top = np.zeros((D, n), bool)
    top[:, q.order[:TOP_K]] = True
    out["BM25 top-3"] = (top, none)
    ok = prov >= 0.5
    first = np.argsort(np.where(ok, -q.raw, np.inf), axis=1, kind="stable")[:, :TOP_K]
    src = np.zeros((D, n), bool)
    np.put_along_axis(src, first, np.take_along_axis(ok, first, axis=1), axis=1)
    out["Provenance + BM25 top-3"] = (src, none)

    w = np.asarray(w, dtype=float)
    w = [w[..., k:k + 1] for k in range(4)]
    tau = np.asarray(tau, dtype=float).reshape(-1, 1)
    if contra is None:
        is_c = np.broadcast_to(q.real_contra, (D, len(q.pairs)))
        is_e = np.broadcast_to(q.real_ent, (D, len(q.pairs)))
        c_conf = q.edge_conf
    else:
        is_c, is_e = contra, q.real_ent & ~contra
        c_conf = np.full(len(q.pairs), SYNTH_CONF)
    agree, penalty = np.zeros((D, n)), np.zeros((D, n))
    for k, (i, j) in enumerate(q.pairs):   # edge order, as ConflictGraph sums them
        for x in (i, j):
            agree[:, x] += is_e[:, k] * q.edge_conf[k]
            penalty[:, x] += is_c[:, k] * c_conf[k]
    base = w[0] * q.rel + w[1] * prov + w[2] * q.conf
    trust = base + w[3] * np.minimum(1.0, agree)
    out["CAS w/o graph"] = (~(base + w[3] * 0.0 < tau), none)
    out["CAS-NLI penalty"] = (~(trust - 0.50 * penalty < tau), none)

    keep = ~(trust < tau)
    for k, (i, j) in enumerate(q.pairs):   # filter_preserving_conflicts: on-topic partners are kept, in edge order
        for a, b in ((i, j), (j, i)):
            if q.rel[b] >= 0.5:
                keep[:, b] |= is_c[:, k] & keep[:, a]
    reported = is_c & keep[:, [i for i, _ in q.pairs]] & keep[:, [j for _, j in q.pairs]]
    marked = np.zeros((D, n), bool)
    for k, (i, j) in enumerate(q.pairs):
        marked[:, i] |= reported[:, k]
        marked[:, j] |= reported[:, k]
    out[PRIMARY] = (keep, marked)
    return out, reported


def endpoints(q, kept, marked):
    return {"silent_exposure": (kept & ~marked)[:, q.poison].any(1), "exposure": kept[:, q.poison].any(1),
            "answer_available": kept[:, q.answer].any(1), "n_kept": kept.sum(1)}


def provenance(rng, poison, D, s=0.5, f=0.2):
    """Trusted ~ U(0.3 + s/2, 0.7 + s/2), untrusted ~ U(0.3 - s/2, 0.7 - s/2); each passage flipped with probability f.
    s = 0.5, f = 0.2 is the generator of studies 1-2 (condition B). s and f may be (D, 1) arrays."""
    trusted = ~poison ^ (rng.random((D, len(poison))) < f)
    return np.where(trusted, 0.5 + np.asarray(s) / 2, 0.5 - np.asarray(s) / 2) + rng.uniform(-0.2, 0.2, (D, len(poison)))

# ============================================================
# 3. VALIDATION GATE
# ============================================================

def validate(questions):
    checked = mismatches = 0
    for study, tag in STUDIES.items():
        with open(os.path.join(OUT_RESULTS, f"{tag}_selections.json"), encoding="utf-8") as f:
            stored = json.load(f)
        for q in (x for x in questions if x.study == study):
            for cond, prov in (("A", np.full(q.n, 0.5)), ("B", q.provenance_B)):
                sel, reported = select(q, prov[None])
                for m in METHODS:
                    key = f"{cond}|{m}|{q.id}"
                    if key not in stored:
                        continue
                    checked += 1
                    same = sorted(np.flatnonzero(sel[m][0][0]).tolist()) == stored[key]["selected"]
                    if m == PRIMARY:
                        pairs = [list(q.pairs[k]) for k in np.flatnonzero(reported[0])]
                        same &= pairs == stored[key]["conflicts"]
                    mismatches += not same
    return {"selections_checked": checked, "mismatches": mismatches}

# ============================================================
# 4. EXPERIMENTS
# ============================================================

def run(questions):
    nq = len(questions)
    rng1, rng2, rng3 = (np.random.default_rng(SEED + k) for k in (1, 2, 3))
    W = np.vstack([DEFAULT_W, rng1.dirichlet(np.ones(4), N_SETTINGS)])
    T = np.concatenate([[DEFAULT_TAU], rng1.uniform(0.30, 0.60, N_SETTINGS)])
    S = len(W)
    cells2 = [(s, f) for s in SEPARATIONS for f in FLIPS]
    s2 = np.repeat([c[0] for c in cells2], E2_DRAWS)[:, None]
    f2 = np.repeat([c[1] for c in cells2], E2_DRAWS)[:, None]
    cells3 = [(r, fa) for r in RECALLS for fa in FALSE_ALARMS]
    r3 = np.repeat([c[0] for c in cells3], E3_DRAWS)[:, None]
    fa3 = np.repeat([c[1] for c in cells3], E3_DRAWS)[:, None]

    e1 = {(c, m, e): np.zeros((S, nq), np.int16) for c in "AB" for m in METHODS for e in ENDPOINTS}
    e2 = {(m, e): np.zeros((len(cells2), nq)) for m in METHODS for e in ENDPOINTS}
    e3 = {(c, m, e): np.zeros((len(cells3), nq)) for c in "AB" for m in METHODS for e in ENDPOINTS}
    e3_det = {(c, k): np.zeros((len(cells3), nq)) for c in "AB" for k in ("recall", "false_alarm")}
    start = time.time()
    for k, q in enumerate(questions):
        # E1: random weights and tau, conditions A and B
        for cond in "AB":
            prov = np.full((S, q.n), 0.5) if cond == "A" else provenance(rng1, q.poison, S)
            sel, _ = select(q, prov, W, T)
            for m in METHODS:
                for e, v in endpoints(q, *sel[m]).items():
                    e1[(cond, m, e)][:, k] = v
        # E2: provenance quality grid, default parameters
        sel, _ = select(q, provenance(rng2, q.poison, len(s2), s2, f2))
        for m in METHODS:
            for e, v in endpoints(q, *sel[m]).items():
                e2[(m, e)][:, k] = v.reshape(len(cells2), E2_DRAWS).mean(1)
        # E3: synthetic detector of recall r and question-level false-alarm rate F, default parameters
        for cond in "AB":
            D = len(r3)
            contra = np.zeros((D, len(q.pairs)), bool)
            contra[:, q.true_pairs] = rng3.random((D, len(q.true_pairs))) < r3
            per_pair = 1 - (1 - fa3) ** (1 / len(q.clean_pairs))
            contra[:, q.clean_pairs] = rng3.random((D, len(q.clean_pairs))) < per_pair
            prov = np.full((D, q.n), 0.5) if cond == "A" else provenance(rng3, q.poison, D)
            sel, _ = select(q, prov, contra=contra)
            for m in METHODS:
                for e, v in endpoints(q, *sel[m]).items():
                    e3[(cond, m, e)][:, k] = v.reshape(len(cells3), E3_DRAWS).mean(1)
            e3_det[(cond, "recall")][:, k] = contra[:, q.true_pairs].any(1).reshape(len(cells3), E3_DRAWS).mean(1)
            e3_det[(cond, "false_alarm")][:, k] = contra[:, q.clean_pairs].any(1).reshape(len(cells3), E3_DRAWS).mean(1)
        if (k + 1) % 50 == 0:
            print(f"[simulation] {k + 1}/{nq} questions  elapsed {(time.time() - start) / 60:.1f} min", flush=True)
    runs = {"E1": 2 * S * nq, "E2": len(s2) * nq, "E3": 2 * len(r3) * nq}
    return W, T, cells2, cells3, e1, e2, e3, e3_det, runs

# ============================================================
# 5. PRE-REGISTERED SUMMARIES (unit of inference = question)
# ============================================================

def qboot(values):
    """Mean and 95% CI over 2,000 bootstrap resamples of questions."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))
    stats = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def scopes(questions):
    study = np.array([q.study for q in questions])
    return {"1": study == "1", "2": study == "2", "pooled": np.ones(len(questions), bool)}


def summarize_e1(questions, W, T, e1):
    rows, out = [], []
    for scope, mask in scopes(questions).items():
        for cond in "AB":
            means = {(m, e): e1[(cond, m, e)][:, mask].mean(1) for m in METHODS for e in ENDPOINTS}
            for s in range(len(W)):
                for m in METHODS:
                    rows.append({"setting": s, "w_relevance": W[s, 0], "w_provenance": W[s, 1], "w_retrieval": W[s, 2],
                                 "w_agreement": W[s, 3], "tau": T[s], "condition": cond, "study": scope, "method": m,
                                 **{e: means[(m, e)][s] for e in ENDPOINTS}})
            for comp in COMPARATORS[cond]:
                for e, lower in LOWER_IS_BETTER.items():
                    diff = means[(PRIMARY, e)] - means[(comp, e)]
                    default, rand = diff[0], diff[1:]
                    per_q = (e1[(cond, PRIMARY, e)][1:, mask] - e1[(cond, comp, e)][1:, mask]).mean(0)
                    mean, lo, hi = qboot(per_q)
                    better = rand < 0 if lower else rand > 0
                    same_sign = float((np.sign(rand) == np.sign(default)).mean())
                    out.append({"study": scope, "condition": cond, "comparator": comp, "endpoint": e,
                                "default_diff": float(default), "median_diff": float(np.median(rand)),
                                "p05_diff": float(np.percentile(rand, 5)), "p95_diff": float(np.percentile(rand, 95)),
                                "share_settings_cas_better": float(better.mean()),
                                "share_settings_same_sign_as_default": same_sign,
                                "robust_to_parameters": bool(np.sign(default) != 0 and same_sign >= 0.90),
                                "mean_diff_over_settings": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows), out


def cell_table(questions, cells, names, data, extra=None):
    rows = []
    for scope, mask in scopes(questions).items():
        for c, cell in enumerate(cells):
            for m in METHODS:
                row = {**dict(zip(names, cell)), "study": scope, "method": m}
                for e in ENDPOINTS:
                    row[e], row[f"{e}_ci_low"], row[f"{e}_ci_high"] = qboot(data[(m, e)][c, mask])
                for k, v in (extra or {}).items():
                    row[f"detector_{k}"] = float(v[c, mask].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_e2(t2):
    p = t2[t2["study"] == "pooled"].set_index(["flip", "separation", "method"])["silent_exposure"]
    out = []
    for f in FLIPS:
        row = {"flip": f}
        for m in ["Provenance + BM25 top-3", "CAS w/o graph", "CAS-NLI penalty", PRIMARY]:
            ok = [s for s in SEPARATIONS if p[(f, s, m)] <= 0.05]
            row[f"min_separation_silent_le_0.05 | {m}"] = ok[0] if ok else None
        ok = [s for s in SEPARATIONS if p[(f, s, PRIMARY)] <= p[(f, s, "Provenance + BM25 top-3")]]
        row["min_separation_cas_no_worse_than_source_filter"] = ok[0] if ok else None
        out.append(row)
    return out


def summarize_e3(t3):
    out = []
    p = t3[t3["study"] == "pooled"].set_index(["condition", "false_alarm", "recall", "method"])
    for cond in "AB":
        for fa in FALSE_ALARMS:
            ok = [r for r in RECALLS if p.loc[(cond, fa, r, PRIMARY), "silent_exposure"]
                  <= p.loc[(cond, fa, r, BEST_BASELINE[cond]), "silent_exposure"]]
            look = lambda r, m, e: float(p.loc[(cond, fa, r, m), e])  # noqa: E731
            out.append({"condition": cond, "false_alarm": fa, "comparator": BEST_BASELINE[cond],
                        "min_recall_cas_no_worse": ok[0] if ok else None,
                        "comparator_silent_exposure": look(0.0, BEST_BASELINE[cond], "silent_exposure"),
                        "penalty_answer_available_r0": look(0.0, "CAS-NLI penalty", "answer_available"),
                        "penalty_answer_available_r1": look(1.0, "CAS-NLI penalty", "answer_available"),
                        "preserve_answer_available_r0": look(0.0, PRIMARY, "answer_available"),
                        "preserve_answer_available_r1": look(1.0, PRIMARY, "answer_available"),
                        "realised_question_recall_at_min_r": look(ok[0], PRIMARY, "detector_recall") if ok else None,
                        "realised_false_alarm": look(0.0, PRIMARY, "detector_false_alarm")})
    return out


def note_vs_adoption():
    """E4 (exploratory): poison adoption by CAS-NLI preserve when the poison was or was not marked by a conflict note."""
    recs = []
    for tag in STUDIES.values():
        items = {it["id"]: it for it in load(f"hotpotqa_{tag}.json")}
        with open(os.path.join(OUT_RESULTS, f"{tag}_selections.json"), encoding="utf-8") as f:
            sel = json.load(f)
        rows = pd.read_csv(os.path.join(OUT_RESULTS, f"{tag}_rows.csv"), dtype={"id": str})
        for r in rows[(rows["method"] == PRIMARY) & (rows["poison_in_context"] == 1)].itertuples():
            pairs = sel[f"{r.condition}|{PRIMARY}|{r.id}"]["conflicts"]
            marked = bool(set(items[r.id]["poison"]) & {x for p in pairs for x in p})
            recs.append({"study": tag, "model": r.model, "marked": marked, "adoption": r.adoption})
    df = pd.DataFrame(recs)
    out = []
    for model, g in df.groupby("model"):
        for marked, h in g.groupby("marked"):
            m, lo, hi = qboot(h["adoption"])
            out.append({"model": model, "poison_marked_by_note": bool(marked), "n": len(h),
                        "adoption": m, "ci_low": lo, "ci_high": hi})
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    items = [(s, it) for s, tag in STUDIES.items() for it in load(f"hotpotqa_{tag}.json")]
    nli = cached_nli([it for _, it in items])
    questions = [Question(it, s, nli) for s, it in items]
    gate = validate(questions)
    print("validation:", gate, flush=True)
    if gate["mismatches"] or not gate["selections_checked"]:
        raise SystemExit("re-implementation does not reproduce the stored selections; nothing simulated")

    W, T, cells2, cells3, e1, e2, e3, e3_det, runs = run(questions)
    t1, e1_out = summarize_e1(questions, W, T, e1)
    t2 = cell_table(questions, cells2, ["separation", "flip"], e2)
    t3 = pd.concat([cell_table(questions, cells3, ["recall", "false_alarm"],
                               {(m, e): e3[(c, m, e)] for m in METHODS for e in ENDPOINTS},
                               {k: e3_det[(c, k)] for k in ("recall", "false_alarm")}).assign(condition=c)
                    for c in "AB"])
    # Where the real DeBERTa-v3 detector (θ = 0.99) sits on the E3 axes, at default parameters (E1 setting 0)
    real = {"pair_recall": float(np.mean([q.real_contra[k] for q in questions for k in q.true_pairs])),
            "question_recall": float(np.mean([q.real_contra[q.true_pairs].any() for q in questions])),
            "question_false_alarm": float(np.mean([q.real_contra[q.clean_pairs].any() for q in questions]))}
    for cond in "AB":
        for m in ("CAS-NLI penalty", PRIMARY):
            real[f"{cond} | {m} | silent_exposure"] = float(e1[(cond, m, "silent_exposure")][0].mean())
    with open(PREREG, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    summary = {"preregistration_sha256": sha, "validation": gate, "runs": runs, "total_runs": sum(runs.values()),
               "n_questions": len(questions), "theta": THETA, "tau_default": DEFAULT_TAU, "weights_default": DEFAULT_W,
               "real_detector": real,
               "E1_parameter_robustness": e1_out, "E2_provenance_quality": summarize_e2(t2),
               "E3_detector_quality": summarize_e3(t3), "E4_conflict_note_vs_adoption": note_vs_adoption(),
               "seconds": time.time() - t0}
    for name, t in (("e1", t1), ("e2", t2), ("e3", t3)):
        t.to_csv(os.path.join(OUT_RESULTS, f"montecarlo_{name}.csv"), index=False)
    with open(os.path.join(OUT_RESULTS, "montecarlo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(f"\n{summary['total_runs']:,} question-level runs ({runs}) in {summary['seconds'] / 60:.1f} min")
    print("\nE1: CAS-NLI preserve minus comparator over 1,250 random weight/tau settings (pooled, 400 questions)")
    for r in e1_out:
        if r["study"] == "pooled":
            print(f"  {r['condition']} vs {r['comparator']:<24} {r['endpoint']:<17} default {r['default_diff']:+.3f}  "
                  f"median {r['median_diff']:+.3f} [p5 {r['p05_diff']:+.3f}, p95 {r['p95_diff']:+.3f}]  "
                  f"CAS better in {r['share_settings_cas_better']:.1%}  robust={r['robust_to_parameters']}")
    print("\nE2:", *summary["E2_provenance_quality"], sep="\n  ")
    print("\nE3:", *summary["E3_detector_quality"], sep="\n  ")
    print("\nE4:", *summary["E4_conflict_note_vs_adoption"], sep="\n  ")


if __name__ == "__main__":
    main()
