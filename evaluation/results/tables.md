# Generated tables (from results.json, do not edit by hand)

## Table 1. Datasets

| File | Format | Records | Used for |
|---|---|---:|---|
| asqa_hf_samples.json | JSONL | 50 | Not used (see notes) |
| cas_pregeneration.json | JSON | 3 | Not used (see notes) |
| cofie.json | JSON | 1 | Not used (see notes) |
| conflictbank_hf_samples.json | JSONL | 50 | Not used (see notes) |
| fever.json | JSON | 2 | Not used (see notes) |
| halueval.json | JSON | 1 | Not used (see notes) |
| hotpotqa.json | JSON | 10 | Not used (see notes) |
| hotpotqa_hf_samples.json | JSONL | 50 | Evidence selection; consistent-pair negatives |
| merged_cleaned_dataset.json | JSON | 48 | Not used (see notes) |
| multinli.json | JSON | 20 | 3-way NLI; contradiction detection |
| musique.json | JSON | 20 | Evidence selection |
| ragtruth_hf_samples.json | JSONL | 50 | Hallucination detection |
| selfrag_train.json | JSON | 10 | Not used (see notes) |
| strategyqa.json | JSON | 1 | Not used (see notes) |
| strategyqa_hf_samples.json | JSONL | 50 | Not used (see notes) |
| test.json | JSON | 5 | Not used (see notes) |
| train.json | JSON | 38 | Not used (see notes) |
| truthful_qa_hf_samples.json | JSONL | 50 | Not used (see notes) |
| truthfulqa.json | JSON | 10 | Not used (see notes) |
| val.json | JSON | 5 | Not used (see notes) |
| verified_output.json | JSON | 1 | Not used (see notes) |
| wikicontradict_hf_samples.json | JSONL | 50 | Contradiction detection (positives) |

## Table 2. Configuration

| Setting | Value |
|---|---|
| seed | 42 |
| bootstrap | 2000 |
| permutations | 10000 |
| tau | 0.45 |
| top_k | 3 |
| provenance_prior | 0.50 |
| python | 3.14.6 |
| numpy | 2.5.1 |
| platform | Windows-11-10.0.26300-SP0 |
| processor | Intel64 Family 6 Model 198 Stepping 2, GenuineIntel |
| total_runtime_s | 4.98 |

## Table 3. Evidence selection (mean [95% bootstrap CI])

| Dataset | Method | Precision | Recall | F1 | Answer coverage | Retained | ms/query |
|---|---|---|---|---|---|---|---:|
| HotpotQA | Vanilla RAG (no filter) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 0.333 [0.333, 0.333] | 1.000 [1.000, 1.000] | 1.000 | 0.15 |
| HotpotQA | Relevance top-3 (BM25) | 0.493 [0.440, 0.547] | 0.740 [0.660, 0.820] | 0.592 [0.528, 0.656] | 0.761 [0.630, 0.870] | 0.300 | 0.36 |
| HotpotQA | CAS w/o conflict graph | 0.752 [0.657, 0.835] | 0.630 [0.550, 0.710] | 0.657 [0.582, 0.725] | 0.609 [0.457, 0.761] | 0.196 | 0.29 |
| HotpotQA | CAS (full pipeline) | 0.589 [0.502, 0.672] | 0.660 [0.580, 0.740] | 0.580 [0.516, 0.646] | 0.652 [0.500, 0.783] | 0.276 | 1.35 |
| MuSiQue | Vanilla RAG (no filter) | 0.100 [0.083, 0.120] | 1.000 [1.000, 1.000] | 0.180 [0.152, 0.211] | 0.733 [0.467, 0.933] | 1.000 | 0.27 |
| MuSiQue | Relevance top-3 (BM25) | 0.333 [0.222, 0.444] | 0.506 [0.322, 0.689] | 0.395 [0.262, 0.527] | 0.267 [0.067, 0.533] | 0.150 | 0.55 |
| MuSiQue | CAS w/o conflict graph | 0.351 [0.273, 0.430] | 0.639 [0.483, 0.789] | 0.417 [0.334, 0.503] | 0.333 [0.133, 0.600] | 0.187 | 0.54 |
| MuSiQue | CAS (full pipeline) | 0.300 [0.220, 0.385] | 0.639 [0.483, 0.789] | 0.369 [0.285, 0.458] | 0.333 [0.133, 0.600] | 0.240 | 4.49 |

## Table 4. Paired differences, CAS (full) minus comparator

| Dataset | Comparator | Metric | Δ mean [95% CI] | p (permutation) | p (Holm) |
|---|---|---|---|---:|---:|
| HotpotQA | Vanilla RAG (no filter) | precision | +0.389 [+0.302, +0.472] | 0.0001 | 0.0018 |
| HotpotQA | Vanilla RAG (no filter) | recall | -0.340 [-0.420, -0.260] | 0.0001 | 0.0018 |
| HotpotQA | Vanilla RAG (no filter) | f1 | +0.247 [+0.182, +0.313] | 0.0001 | 0.0018 |
| HotpotQA | Relevance top-3 (BM25) | precision | +0.095 [+0.021, +0.172] | 0.0194 | 0.1940 |
| HotpotQA | Relevance top-3 (BM25) | recall | -0.080 [-0.140, -0.020] | 0.0196 | 0.1940 |
| HotpotQA | Relevance top-3 (BM25) | f1 | -0.012 [-0.062, +0.040] | 0.6715 | 1.0000 |
| HotpotQA | CAS w/o conflict graph | precision | -0.163 [-0.235, -0.095] | 0.0001 | 0.0018 |
| HotpotQA | CAS w/o conflict graph | recall | +0.030 [+0.000, +0.070] | 0.2476 | 1.0000 |
| HotpotQA | CAS w/o conflict graph | f1 | -0.076 [-0.123, -0.033] | 0.0014 | 0.0168 |
| MuSiQue | Vanilla RAG (no filter) | precision | +0.200 [+0.127, +0.278] | 0.0001 | 0.0018 |
| MuSiQue | Vanilla RAG (no filter) | recall | -0.361 [-0.517, -0.211] | 0.0039 | 0.0429 |
| MuSiQue | Vanilla RAG (no filter) | f1 | +0.189 [+0.111, +0.276] | 0.0010 | 0.0130 |
| MuSiQue | Relevance top-3 (BM25) | precision | -0.034 [-0.115, +0.050] | 0.4587 | 1.0000 |
| MuSiQue | Relevance top-3 (BM25) | recall | +0.133 [-0.033, +0.333] | 0.3766 | 1.0000 |
| MuSiQue | Relevance top-3 (BM25) | f1 | -0.025 [-0.121, +0.074] | 0.6329 | 1.0000 |
| MuSiQue | CAS w/o conflict graph | precision | -0.052 [-0.098, -0.014] | 0.0640 | 0.5119 |
| MuSiQue | CAS w/o conflict graph | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS w/o conflict graph | f1 | -0.048 [-0.090, -0.013] | 0.0640 | 0.5119 |

## Table 5. Conflict-graph diagnostics

| Dataset | Pairs | Entailment | Contradiction | Neutral | Mean agreement (gold) | Mean agreement (distractor) |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 2250 | 111 | 0 | 2139 | 0.217 | 0.214 |
| MuSiQue | 2850 | 46 | 0 | 2804 | 0.088 | 0.179 |

## Table 6. Contradiction detection (pooled)

| Detector | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Negation-cue baseline | 0.643 [0.357, 0.889] | 0.161 [0.069, 0.269] | 0.257 [0.119, 0.391] | 0.567 [0.475, 0.650] |
| CAS NLIVerifier (as implemented) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.533 [0.442, 0.617] |

## Table 7. NLI and hallucination detection

| Task | Method | Metric | Value [95% CI] |
|---|---|---|---|
| MultiNLI 3-way (n=20) | CAS NLIVerifier | Accuracy | 0.500 [0.300, 0.700] |
| MultiNLI 3-way (n=20) | CAS NLIVerifier | Macro-F1 | 0.392 |
| MultiNLI 3-way (n=20) | Majority class (entailment) | Accuracy | 0.400 [0.200, 0.600] |
| RAGTruth (n=50, 12 hallucinated) | Unsupported-token rate | AUROC | 0.815 [0.686, 0.924] |
| RAGTruth (n=50, 12 hallucinated) | CAS NLIVerifier | AUROC | 0.500 [0.500, 0.500] |
