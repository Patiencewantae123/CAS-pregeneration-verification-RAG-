"""
Exploratory: can the conflict detector be made accurate and cheap enough for day-to-day use on CPU?

Ground truth comes from the two poisoning benchmarks:
- gold vs poison passage pairs are true conflicts;
- pairs of original HotpotQA passages are treated as non-conflicting, so a flag there is a false alarm on a normal query.
WikiContradict gives 50 natural contradictions.

Detector families:
- pairwise NLI: DeBERTa-v3 or MiniLM; whole passage or the most question-relevant sentence; contradiction threshold θ;
  a lexical-overlap gate that skips unrelated pairs; one or both directions.
- answer-level: an extractive QA model answers the question from each passage; passages whose answers differ conflict.

Selection rule, fixed before results were seen: on study 1, maximise question-level recall subject to false alarms <= 10%.
Study 2 and WikiContradict are the check. Predictions and their CPU times are cached in results/detector_cache/.

Usage:  py evaluation/run_detector_probe.py
Output: evaluation/results/detector_probe.json
"""
import itertools
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import OUT_RESULTS, bm25_scores, load, nli_verifier, sentences, tokens  # noqa: E402
from run_llm_generation import normalize, token_f1  # noqa: E402
from services.nli_model_verifier import NLIModelVerifier  # noqa: E402

CACHE = os.path.join(OUT_RESULTS, "detector_cache")
MINI = "cross-encoder/nli-MiniLM2-L6-H768"
QA_MODELS = ["deepset/tinyroberta-squad2", "deepset/roberta-base-squad2"]
STUDIES = {"1": "hotpotqa_poison.json", "2": "hotpotqa_poison2.json"}
NLI_GRID = {"detector": ["deberta_passage", "deberta_sentence", "minilm_passage", "minilm_sentence"],
            "theta": [0.5, 0.9, 0.99], "gate": [0.0, 0.2, 0.3, 0.5], "directions": ["both", "one"]}
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
MAX_FALSE_ALARM = 0.10

# ============================================================
# 1. PAIRWISE NLI PREDICTIONS
# ============================================================

def overlap(a, b):
    x, y = set(tokens(a)), set(tokens(b))
    return len(x & y) / max(1, min(len(x), len(y)))


def best_sentence(question, passage):
    sents = [s for s in sentences(passage) if len(s.split()) >= 5] or [passage]
    scores = bm25_scores(question, sents)
    return sents[max(range(len(sents)), key=lambda i: scores[i])]


def labelled_pairs(item):
    """(i, j, kind): each poison vs its source gold passage ('conflict'); original vs original ('clean')."""
    poison, n = set(item["poison"]), len(item["passages"])
    out = [(max(item["gold"], key=lambda g: overlap(item["passages"][g], item["passages"][p])), p, "conflict")
           for p in item["poison"]]
    out += [(i, j, "clean") for i in range(n) for j in range(i + 1, n) if i not in poison and j not in poison]
    return out


def nli_records():
    verifiers = {"deberta_passage": nli_verifier(),
                 "deberta_sentence": NLIModelVerifier(cache_path=os.path.join(CACHE, "deberta_sentence.jsonl"),
                                                      batch_size=32),
                 "minilm": NLIModelVerifier(MINI, cache_path=os.path.join(CACHE, "minilm.jsonl"), batch_size=32)}

    def score(tag, a, b):
        v = verifiers["minilm" if tag.startswith("minilm") else tag]
        flat = [(a, b), (b, a)]
        ab, ba = v.probabilities(flat)
        return {"ab": ab, "ba": ba, "s": v.inference_seconds(flat)}

    records = []
    for study, name in STUDIES.items():
        for it in load(name):
            sent = {i: best_sentence(it["question"], p) for i, p in enumerate(it["passages"])}
            for i, j, kind in labelled_pairs(it):
                a, b = it["passages"][i], it["passages"][j]
                records.append({"study": study, "q": it["id"], "kind": kind, "overlap": overlap(a, b),
                                "deberta_passage": score("deberta_passage", a, b),
                                "deberta_sentence": score("deberta_sentence", sent[i], sent[j]),
                                "minilm_passage": score("minilm_passage", a, b),
                                "minilm_sentence": score("minilm_sentence", sent[i], sent[j])})
    for k, r in enumerate(load("wikicontradict_hf_samples.json")):
        a, b, q = r["context1"], r["context2"], r["question"]
        sa, sb = best_sentence(q, a), best_sentence(q, b)
        records.append({"study": "wiki", "q": k, "kind": "conflict", "overlap": overlap(a, b),
                        "group": r["contradictType"].split(" ")[0],
                        "deberta_passage": score("deberta_passage", a, b),
                        "deberta_sentence": score("deberta_sentence", sa, sb),
                        "minilm_passage": score("minilm_passage", a, b),
                        "minilm_sentence": score("minilm_sentence", sa, sb)})
    return records


def flagged(r, detector, theta, gate, directions):
    if r["overlap"] < gate:
        return False
    x = r[detector]
    p = x["ab"] if directions == "one" else {k: (x["ab"][k] + x["ba"][k]) / 2 for k in x["ab"]}
    return max(p, key=p.get) == "contradiction" and p["contradiction"] >= theta


def evaluate_nli(records, cfg):
    detector, theta, gate, directions = cfg
    out = {"detector": detector, "theta": theta, "gate": gate, "directions": directions}
    for study in STUDIES:
        q = defaultdict(lambda: {"hit": False, "fa": False, "s": 0.0, "pairs": 0})
        for r in records:
            if r["study"] != study:
                continue
            f = flagged(r, *cfg)
            if r["kind"] == "conflict":
                q[r["q"]]["hit"] |= f
            else:
                q[r["q"]]["fa"] |= f
                if r["overlap"] >= gate:   # cost on the original passages = the cost of a normal query
                    q[r["q"]]["s"] += r[detector]["s"] / (2 if directions == "one" else 1)
                    q[r["q"]]["pairs"] += 1
        n = len(q)
        out[f"s{study}_recall"] = sum(v["hit"] for v in q.values()) / n
        out[f"s{study}_false_alarm"] = sum(v["fa"] for v in q.values()) / n
        out[f"s{study}_sec_per_query"] = sum(v["s"] for v in q.values()) / n
        out[f"s{study}_pairs_per_query"] = sum(v["pairs"] for v in q.values()) / n
    wiki = [flagged(r, *cfg) for r in records if r["study"] == "wiki"]
    out["wiki_recall"] = sum(wiki) / len(wiki)
    return out

# ============================================================
# 2. ANSWER-LEVEL DETECTION (extractive QA)
# ============================================================

def qa_spans(model_name):
    path = os.path.join(CACHE, f"qa_{model_name.split('/')[-1]}.json")
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    jobs = [(f"{s}|{it['id']}", it["question"], it["passages"]) for s, name in STUDIES.items() for it in load(name)]
    jobs += [(f"wiki|{k}", r["question"], [r["context1"], r["context2"]])
             for k, r in enumerate(load("wikicontradict_hf_samples.json"))]
    missing = [j for j in jobs if j[0] not in cache]
    if missing:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        net = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()
        for key, question, passages in missing:
            t = time.perf_counter()
            with torch.inference_mode():
                enc = tok([question] * len(passages), passages, truncation="only_second", max_length=512,
                          padding=True, return_offsets_mapping=True, return_tensors="pt")
                offsets = enc.pop("offset_mapping")
                logits = net(**enc)
                ps, pe = logits.start_logits.softmax(-1), logits.end_logits.softmax(-1)
            spans = []
            for b, passage in enumerate(passages):
                ctx = [i for i, s in enumerate(enc.sequence_ids(b)) if s == 1]
                lo, hi = ctx[0], ctx[-1]
                grid = ps[b, lo:hi + 1, None] * pe[b, None, lo:hi + 1]
                grid = torch.triu(grid) - torch.triu(grid, diagonal=30)
                i, j = divmod(int(grid.argmax()), grid.shape[1])
                best, null = float(grid[i, j]), float(ps[b, 0] * pe[b, 0])
                spans.append({"text": "", "score": null} if null >= best else
                             {"text": passage[int(offsets[b, lo + i, 0]):int(offsets[b, lo + j, 1])], "score": best})
            cache[key] = {"spans": spans, "seconds": time.perf_counter() - t}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return cache


def evaluate_qa(model_name, cache, sigma):
    differ = lambda a, b: token_f1(a, b) < 0.5  # noqa: E731
    out = {"detector": f"answer-level ({model_name.split('/')[-1]})", "sigma": sigma}
    for study, name in STUDIES.items():
        hit = fa = 0
        sec = 0.0
        items = load(name)
        for it in items:
            c = cache[f"{study}|{it['id']}"]
            spans, poison = c["spans"], set(it["poison"])
            bearing = {i for i, s in enumerate(spans) if s["text"] and s["score"] >= sigma}
            clean = [i for i in bearing if i not in poison]
            gold = [i for i in clean if normalize(it["answer"]) in normalize(spans[i]["text"])
                    or token_f1(spans[i]["text"], it["answer"]) >= 0.5]
            hit += any(differ(spans[g]["text"], spans[p]["text"]) for g in gold for p in bearing & poison)
            fa += any(differ(spans[i]["text"], spans[j]["text"]) for i in clean for j in clean if i < j)
            sec += c["seconds"] * (len(it["passages"]) - len(it["poison"])) / len(it["passages"])
        out[f"s{study}_recall"], out[f"s{study}_false_alarm"] = hit / len(items), fa / len(items)
        out[f"s{study}_sec_per_query"] = sec / len(items)
    wiki = []
    for k in range(len(load("wikicontradict_hf_samples.json"))):
        a, b = cache[f"wiki|{k}"]["spans"]
        wiki.append(bool(a["text"] and b["text"] and min(a["score"], b["score"]) >= sigma and differ(a["text"], b["text"])))
    out["wiki_recall"] = sum(wiki) / len(wiki)
    return out

# ============================================================
# 3. MAIN
# ============================================================

def choose(rows):
    ok = [r for r in rows if r["s1_false_alarm"] <= MAX_FALSE_ALARM]
    return max(ok, key=lambda r: (r["s1_recall"], -r["s1_sec_per_query"])) if ok else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    records = nli_records()
    nli = [evaluate_nli(records, cfg) for cfg in itertools.product(*NLI_GRID.values())]
    qa = [evaluate_qa(m, qa_spans(m), s) for m in QA_MODELS for s in SIGMAS]
    current = next(r for r in nli if (r["detector"], r["theta"], r["gate"], r["directions"])
                   == ("deberta_passage", 0.99, 0.0, "both"))
    result = {"selection_rule": f"max study-1 recall with study-1 false alarms <= {MAX_FALSE_ALARM}",
              "current_cas_detector": current, "chosen_nli": choose(nli), "chosen_answer_level": choose(qa),
              "nli": nli, "answer_level": qa}
    with open(os.path.join(OUT_RESULTS, "detector_probe.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for name in ("current_cas_detector", "chosen_nli", "chosen_answer_level"):
        r = result[name]
        print(f"{name:<22} S1 recall {r['s1_recall']:.3f} FA {r['s1_false_alarm']:.3f} | "
              f"S2 recall {r['s2_recall']:.3f} FA {r['s2_false_alarm']:.3f} | wiki {r['wiki_recall']:.3f} | "
              f"{r['s1_sec_per_query']:.2f} s/query  {r}")


if __name__ == "__main__":
    main()
