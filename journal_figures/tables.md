# Tables (generated from evaluation/results; do not edit by hand)

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
| total_runtime_s | 17.57 |

## Table 3. Evidence selection (mean [95% bootstrap CI])

| Dataset | Method | Precision | Recall | F1 | Answer coverage | Retained | ms/query |
|---|---|---|---|---|---|---:|---:|
| HotpotQA | Vanilla RAG (no filter) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 0.333 [0.333, 0.333] | 1.000 [1.000, 1.000] | 1.000 | 0.00 |
| HotpotQA | Relevance top-3 (BM25) | 0.493 [0.440, 0.547] | 0.740 [0.660, 0.820] | 0.592 [0.528, 0.656] | 0.761 [0.630, 0.870] | 0.300 | 0.26 |
| HotpotQA | CAS w/o conflict graph | 0.680 [0.595, 0.761] | 0.700 [0.630, 0.770] | 0.649 [0.585, 0.713] | 0.696 [0.565, 0.826] | 0.250 | 0.28 |
| HotpotQA | CAS (full pipeline) | 0.518 [0.444, 0.595] | 0.720 [0.650, 0.790] | 0.552 [0.499, 0.609] | 0.717 [0.587, 0.848] | 0.352 | 1.26 |
| HotpotQA | CAS-NLI (full pipeline) | 0.421 [0.301, 0.541] | 0.330 [0.240, 0.420] | 0.349 [0.257, 0.446] | 0.326 [0.196, 0.457] | 0.124 | 23355.54 |
| HotpotQA | CAS-NLI conflict-preserving | 0.650 [0.562, 0.731] | 0.700 [0.630, 0.770] | 0.630 [0.564, 0.692] | 0.717 [0.587, 0.848] | 0.272 | 23355.60 |
| HotpotQA | CAS-NLI θ-CV (penalty) | 0.473 [0.366, 0.579] | 0.490 [0.390, 0.600] | 0.447 [0.355, 0.537] | 0.500 [0.348, 0.652] | 0.190 | 23355.54 |
| HotpotQA | CAS-NLI θ-CV preserving | 0.651 [0.564, 0.731] | 0.700 [0.630, 0.770] | 0.632 [0.567, 0.694] | 0.696 [0.565, 0.826] | 0.268 | 23355.55 |
| MuSiQue | Vanilla RAG (no filter) | 0.100 [0.083, 0.120] | 1.000 [1.000, 1.000] | 0.180 [0.152, 0.211] | 0.733 [0.467, 0.933] | 1.000 | 0.00 |
| MuSiQue | Relevance top-3 (BM25) | 0.333 [0.222, 0.444] | 0.506 [0.322, 0.689] | 0.395 [0.262, 0.527] | 0.267 [0.067, 0.533] | 0.150 | 0.47 |
| MuSiQue | CAS w/o conflict graph | 0.350 [0.258, 0.443] | 0.672 [0.506, 0.833] | 0.421 [0.326, 0.523] | 0.400 [0.200, 0.667] | 0.207 | 0.52 |
| MuSiQue | CAS (full pipeline) | 0.303 [0.214, 0.397] | 0.672 [0.506, 0.833] | 0.376 [0.278, 0.474] | 0.400 [0.200, 0.667] | 0.263 | 4.27 |
| MuSiQue | CAS-NLI (full pipeline) | 0.241 [0.111, 0.397] | 0.317 [0.150, 0.500] | 0.239 [0.124, 0.364] | 0.333 [0.133, 0.600] | 0.103 | 84486.67 |
| MuSiQue | CAS-NLI conflict-preserving | 0.330 [0.230, 0.431] | 0.672 [0.506, 0.833] | 0.401 [0.300, 0.509] | 0.400 [0.200, 0.667] | 0.240 | 84486.65 |
| MuSiQue | CAS-NLI θ-CV (penalty) | 0.285 [0.175, 0.393] | 0.439 [0.272, 0.606] | 0.316 [0.198, 0.430] | 0.333 [0.133, 0.600] | 0.167 | 84486.62 |
| MuSiQue | CAS-NLI θ-CV preserving | 0.330 [0.230, 0.431] | 0.672 [0.506, 0.833] | 0.401 [0.300, 0.509] | 0.400 [0.200, 0.667] | 0.240 | 84486.67 |

## Table 4. Paired differences in evidence selection (reference − comparator)

| Dataset | Reference | Comparator | Metric | Δ mean [95% CI] | p | p (Holm) |
|---|---|---|---|---|---:|---:|
| HotpotQA | CAS (full pipeline) | Vanilla RAG (no filter) | precision | +0.318 [+0.244, +0.395] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | Vanilla RAG (no filter) | recall | -0.280 [-0.350, -0.210] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | +0.218 [+0.166, +0.276] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | Relevance top-3 (BM25) | precision | +0.025 [-0.045, +0.104] | 0.5421 | 1.0000 |
| HotpotQA | CAS (full pipeline) | Relevance top-3 (BM25) | recall | -0.020 [-0.090, +0.050] | 0.7825 | 1.0000 |
| HotpotQA | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | -0.040 [-0.091, +0.012] | 0.1300 | 1.0000 |
| HotpotQA | CAS (full pipeline) | CAS w/o conflict graph | precision | -0.162 [-0.232, -0.097] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | CAS w/o conflict graph | recall | +0.020 [+0.000, +0.050] | 0.4951 | 1.0000 |
| HotpotQA | CAS (full pipeline) | CAS w/o conflict graph | f1 | -0.098 [-0.145, -0.053] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | CAS-NLI (full pipeline) | precision | +0.097 [-0.026, +0.216] | 0.1326 | 1.0000 |
| HotpotQA | CAS (full pipeline) | CAS-NLI (full pipeline) | recall | +0.390 [+0.290, +0.490] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | CAS-NLI (full pipeline) | f1 | +0.203 [+0.105, +0.295] | 0.0003 | 0.0330 |
| HotpotQA | CAS (full pipeline) | CAS-NLI conflict-preserving | precision | -0.132 [-0.197, -0.072] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | CAS-NLI conflict-preserving | recall | +0.020 [+0.000, +0.050] | 0.4951 | 1.0000 |
| HotpotQA | CAS (full pipeline) | CAS-NLI conflict-preserving | f1 | -0.079 [-0.123, -0.039] | 0.0002 | 0.0238 |
| HotpotQA | CAS (full pipeline) | CAS-NLI θ-CV (penalty) | precision | +0.045 [-0.049, +0.138] | 0.3686 | 1.0000 |
| HotpotQA | CAS (full pipeline) | CAS-NLI θ-CV (penalty) | recall | +0.230 [+0.140, +0.330] | 0.0002 | 0.0238 |
| HotpotQA | CAS (full pipeline) | CAS-NLI θ-CV (penalty) | f1 | +0.105 [+0.017, +0.191] | 0.0238 | 1.0000 |
| HotpotQA | CAS (full pipeline) | CAS-NLI θ-CV preserving | precision | -0.133 [-0.198, -0.074] | 0.0001 | 0.0150 |
| HotpotQA | CAS (full pipeline) | CAS-NLI θ-CV preserving | recall | +0.020 [+0.000, +0.050] | 0.4951 | 1.0000 |
| HotpotQA | CAS (full pipeline) | CAS-NLI θ-CV preserving | f1 | -0.080 [-0.124, -0.041] | 0.0002 | 0.0238 |
| HotpotQA | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | precision | +0.221 [+0.101, +0.341] | 0.0006 | 0.0612 |
| HotpotQA | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | recall | -0.670 [-0.760, -0.580] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | +0.015 [-0.076, +0.113] | 0.7443 | 1.0000 |
| HotpotQA | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | precision | -0.072 [-0.195, +0.054] | 0.2886 | 1.0000 |
| HotpotQA | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | recall | -0.410 [-0.520, -0.290] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | -0.243 [-0.351, -0.134] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | CAS w/o conflict graph | precision | -0.259 [-0.377, -0.139] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | CAS w/o conflict graph | recall | -0.370 [-0.470, -0.270] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | -0.301 [-0.396, -0.205] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI conflict-preserving | precision | -0.228 [-0.347, -0.103] | 0.0005 | 0.0525 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI conflict-preserving | recall | -0.370 [-0.470, -0.270] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI conflict-preserving | f1 | -0.282 [-0.378, -0.182] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI θ-CV (penalty) | precision | -0.051 [-0.141, +0.027] | 0.2536 | 1.0000 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI θ-CV (penalty) | recall | -0.160 [-0.240, -0.090] | 0.0003 | 0.0330 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI θ-CV (penalty) | f1 | -0.098 [-0.166, -0.040] | 0.0035 | 0.3360 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI θ-CV preserving | precision | -0.230 [-0.348, -0.106] | 0.0005 | 0.0525 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI θ-CV preserving | recall | -0.370 [-0.470, -0.270] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI (full pipeline) | CAS-NLI θ-CV preserving | f1 | -0.283 [-0.379, -0.184] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | precision | +0.450 [+0.362, +0.531] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | recall | -0.300 [-0.370, -0.230] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | +0.297 [+0.231, +0.359] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | precision | +0.156 [+0.083, +0.228] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | recall | -0.040 [-0.110, +0.030] | 0.3963 | 1.0000 |
| HotpotQA | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | +0.038 [-0.011, +0.086] | 0.1275 | 1.0000 |
| HotpotQA | CAS-NLI conflict-preserving | CAS w/o conflict graph | precision | -0.031 [-0.071, -0.002] | 0.0632 | 1.0000 |
| HotpotQA | CAS-NLI conflict-preserving | CAS w/o conflict graph | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | -0.019 [-0.044, -0.002] | 0.0632 | 1.0000 |
| HotpotQA | CAS-NLI conflict-preserving | CAS-NLI θ-CV (penalty) | precision | +0.177 [+0.080, +0.278] | 0.0017 | 0.1700 |
| HotpotQA | CAS-NLI conflict-preserving | CAS-NLI θ-CV (penalty) | recall | +0.210 [+0.120, +0.310] | 0.0002 | 0.0238 |
| HotpotQA | CAS-NLI conflict-preserving | CAS-NLI θ-CV (penalty) | f1 | +0.184 [+0.096, +0.275] | 0.0002 | 0.0238 |
| HotpotQA | CAS-NLI conflict-preserving | CAS-NLI θ-CV preserving | precision | -0.001 [-0.003, +0.000] | 0.5004 | 1.0000 |
| HotpotQA | CAS-NLI conflict-preserving | CAS-NLI θ-CV preserving | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | CAS-NLI conflict-preserving | CAS-NLI θ-CV preserving | f1 | -0.001 [-0.004, +0.000] | 0.5004 | 1.0000 |
| HotpotQA | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | precision | +0.273 [+0.166, +0.379] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | recall | -0.510 [-0.610, -0.400] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | +0.113 [+0.022, +0.204] | 0.0231 | 1.0000 |
| HotpotQA | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | precision | -0.021 [-0.133, +0.091] | 0.7364 | 1.0000 |
| HotpotQA | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | recall | -0.250 [-0.370, -0.130] | 0.0004 | 0.0424 |
| HotpotQA | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | -0.145 [-0.250, -0.043] | 0.0087 | 0.7829 |
| HotpotQA | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | precision | -0.208 [-0.309, -0.110] | 0.0005 | 0.0525 |
| HotpotQA | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | recall | -0.210 [-0.310, -0.120] | 0.0002 | 0.0238 |
| HotpotQA | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | -0.203 [-0.295, -0.117] | 0.0002 | 0.0238 |
| HotpotQA | CAS-NLI θ-CV (penalty) | CAS-NLI θ-CV preserving | precision | -0.178 [-0.280, -0.082] | 0.0017 | 0.1700 |
| HotpotQA | CAS-NLI θ-CV (penalty) | CAS-NLI θ-CV preserving | recall | -0.210 [-0.310, -0.120] | 0.0002 | 0.0238 |
| HotpotQA | CAS-NLI θ-CV (penalty) | CAS-NLI θ-CV preserving | f1 | -0.185 [-0.276, -0.097] | 0.0002 | 0.0238 |
| HotpotQA | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | precision | +0.451 [+0.364, +0.531] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | recall | -0.300 [-0.370, -0.230] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | +0.299 [+0.233, +0.360] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | precision | +0.158 [+0.085, +0.229] | 0.0001 | 0.0150 |
| HotpotQA | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | recall | -0.040 [-0.110, +0.030] | 0.3963 | 1.0000 |
| HotpotQA | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | +0.040 [-0.009, +0.087] | 0.1110 | 1.0000 |
| HotpotQA | CAS-NLI θ-CV preserving | CAS w/o conflict graph | precision | -0.029 [-0.069, -0.001] | 0.1248 | 1.0000 |
| HotpotQA | CAS-NLI θ-CV preserving | CAS w/o conflict graph | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | -0.018 [-0.043, -0.001] | 0.1248 | 1.0000 |
| MuSiQue | CAS (full pipeline) | Vanilla RAG (no filter) | precision | +0.203 [+0.124, +0.288] | 0.0001 | 0.0150 |
| MuSiQue | CAS (full pipeline) | Vanilla RAG (no filter) | recall | -0.328 [-0.494, -0.167] | 0.0090 | 0.8009 |
| MuSiQue | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | +0.197 [+0.106, +0.294] | 0.0010 | 0.1010 |
| MuSiQue | CAS (full pipeline) | Relevance top-3 (BM25) | precision | -0.030 [-0.102, +0.041] | 0.4378 | 1.0000 |
| MuSiQue | CAS (full pipeline) | Relevance top-3 (BM25) | recall | +0.167 [+0.000, +0.367] | 0.2496 | 1.0000 |
| MuSiQue | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | -0.018 [-0.096, +0.062] | 0.6827 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS w/o conflict graph | precision | -0.047 [-0.090, -0.013] | 0.0322 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS w/o conflict graph | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS w/o conflict graph | f1 | -0.045 [-0.079, -0.014] | 0.0322 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI (full pipeline) | precision | +0.062 [-0.077, +0.201] | 0.4152 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI (full pipeline) | recall | +0.356 [+0.189, +0.533] | 0.0038 | 0.3610 |
| MuSiQue | CAS (full pipeline) | CAS-NLI (full pipeline) | f1 | +0.138 [+0.023, +0.257] | 0.0463 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI conflict-preserving | precision | -0.027 [-0.072, +0.008] | 0.2791 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI conflict-preserving | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI conflict-preserving | f1 | -0.025 [-0.060, +0.005] | 0.2489 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI θ-CV (penalty) | precision | +0.018 [-0.051, +0.102] | 0.7342 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI θ-CV (penalty) | recall | +0.233 [+0.100, +0.400] | 0.0323 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI θ-CV (penalty) | f1 | +0.061 [-0.009, +0.141] | 0.1394 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI θ-CV preserving | precision | -0.027 [-0.072, +0.008] | 0.2791 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI θ-CV preserving | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS (full pipeline) | CAS-NLI θ-CV preserving | f1 | -0.025 [-0.060, +0.005] | 0.2489 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | precision | +0.141 [+0.016, +0.294] | 0.0724 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | recall | -0.683 [-0.850, -0.500] | 0.0003 | 0.0330 |
| MuSiQue | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | +0.059 [-0.057, +0.182] | 0.3676 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | precision | -0.092 [-0.237, +0.060] | 0.2841 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | recall | -0.189 [-0.456, +0.100] | 0.2734 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | -0.156 [-0.315, +0.006] | 0.0963 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS w/o conflict graph | precision | -0.109 [-0.233, +0.011] | 0.1108 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS w/o conflict graph | recall | -0.356 [-0.533, -0.189] | 0.0038 | 0.3610 |
| MuSiQue | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | -0.182 [-0.292, -0.070] | 0.0115 | 0.9889 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI conflict-preserving | precision | -0.089 [-0.215, +0.033] | 0.2065 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI conflict-preserving | recall | -0.356 [-0.533, -0.189] | 0.0038 | 0.3610 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI conflict-preserving | f1 | -0.163 [-0.275, -0.050] | 0.0172 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI θ-CV (penalty) | precision | -0.044 [-0.167, +0.076] | 0.5751 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI θ-CV (penalty) | recall | -0.122 [-0.233, -0.033] | 0.1297 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI θ-CV (penalty) | f1 | -0.077 [-0.182, +0.018] | 0.1904 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI θ-CV preserving | precision | -0.089 [-0.215, +0.033] | 0.2065 | 1.0000 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI θ-CV preserving | recall | -0.356 [-0.533, -0.189] | 0.0038 | 0.3610 |
| MuSiQue | CAS-NLI (full pipeline) | CAS-NLI θ-CV preserving | f1 | -0.163 [-0.275, -0.050] | 0.0172 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | precision | +0.230 [+0.136, +0.326] | 0.0001 | 0.0150 |
| MuSiQue | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | recall | -0.328 [-0.494, -0.167] | 0.0090 | 0.8009 |
| MuSiQue | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | +0.222 [+0.120, +0.329] | 0.0017 | 0.1700 |
| MuSiQue | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | precision | -0.003 [-0.068, +0.062] | 0.9404 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | recall | +0.167 [+0.000, +0.367] | 0.2496 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | +0.007 [-0.059, +0.076] | 0.8521 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS w/o conflict graph | precision | -0.020 [-0.042, -0.002] | 0.1299 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS w/o conflict graph | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | -0.020 [-0.044, -0.002] | 0.1299 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS-NLI θ-CV (penalty) | precision | +0.045 [-0.018, +0.128] | 0.3111 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS-NLI θ-CV (penalty) | recall | +0.233 [+0.100, +0.400] | 0.0323 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS-NLI θ-CV (penalty) | f1 | +0.086 [+0.011, +0.172] | 0.0777 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS-NLI θ-CV preserving | precision | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS-NLI θ-CV preserving | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS-NLI conflict-preserving | CAS-NLI θ-CV preserving | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | precision | +0.185 [+0.082, +0.288] | 0.0041 | 0.3731 |
| MuSiQue | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | recall | -0.561 [-0.728, -0.394] | 0.0003 | 0.0330 |
| MuSiQue | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | +0.136 [+0.025, +0.248] | 0.0340 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | precision | -0.048 [-0.142, +0.042] | 0.2836 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | recall | -0.067 [-0.300, +0.167] | 0.8009 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | -0.079 [-0.202, +0.038] | 0.2145 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | precision | -0.065 [-0.152, +0.001] | 0.1095 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | recall | -0.233 [-0.400, -0.100] | 0.0323 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | -0.105 [-0.196, -0.027] | 0.0244 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | CAS-NLI θ-CV preserving | precision | -0.045 [-0.128, +0.018] | 0.3111 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | CAS-NLI θ-CV preserving | recall | -0.233 [-0.400, -0.100] | 0.0323 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV (penalty) | CAS-NLI θ-CV preserving | f1 | -0.086 [-0.172, -0.011] | 0.0777 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | precision | +0.230 [+0.136, +0.326] | 0.0001 | 0.0150 |
| MuSiQue | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | recall | -0.328 [-0.494, -0.167] | 0.0090 | 0.8009 |
| MuSiQue | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | +0.222 [+0.120, +0.329] | 0.0017 | 0.1700 |
| MuSiQue | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | precision | -0.003 [-0.068, +0.062] | 0.9404 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | recall | +0.167 [+0.000, +0.367] | 0.2496 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | +0.007 [-0.059, +0.076] | 0.8521 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV preserving | CAS w/o conflict graph | precision | -0.020 [-0.042, -0.002] | 0.1299 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV preserving | CAS w/o conflict graph | recall | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | -0.020 [-0.044, -0.002] | 0.1299 | 1.0000 |

## Table 5. Conflict-graph diagnostics

| Verifier | Dataset | Pairs | Entail. | Contra. | Neutral | Agreement gold / distr. | Penalty gold / distr. |
|---|---|---:|---:|---:|---:|---|---|
| keyword | HotpotQA | 2250 | 111 | 0 | 2139 | 0.217 / 0.214 | 0.000 / 0.000 |
| keyword | MuSiQue | 2850 | 46 | 0 | 2804 | 0.088 / 0.179 | 0.000 / 0.000 |
| nli | HotpotQA | 2250 | 8 | 340 | 1902 | 0.005 / 0.019 | 1.073 / 1.169 |
| nli | MuSiQue | 2850 | 24 | 273 | 2553 | 0.067 / 0.061 | 1.515 / 1.416 |
| nli_cv | HotpotQA | 2250 | 12 | 135 | 2103 | 0.008 / 0.027 | 0.459 / 0.559 |
| nli_cv | MuSiQue | 2850 | 34 | 80 | 2736 | 0.093 / 0.068 | 0.433 / 0.543 |

## Table 6. Contradiction detection (pooled, n = 120 pairs)

| Detector | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Negation-cue baseline | 0.643 [0.357, 0.889] | 0.161 [0.069, 0.269] | 0.257 [0.119, 0.391] | 0.567 [0.475, 0.650] |
| CAS NLIVerifier (as implemented) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.533 [0.442, 0.617] |
| DeBERTa-v3 NLI (CAS-NLI) | 0.833 [0.700, 0.946] | 0.536 [0.411, 0.673] | 0.652 [0.538, 0.760] | 0.733 [0.658, 0.808] |
| DeBERTa-v3 NLI θ-CV | 0.833 [0.700, 0.946] | 0.536 [0.411, 0.673] | 0.652 [0.538, 0.760] | 0.733 [0.658, 0.808] |

## Table 7. MultiNLI 3-way and RAGTruth hallucination detection

| Task | Method | Metric | Value [95% CI] |
|---|---|---|---|
| MultiNLI (n=20) | CAS NLIVerifier (as implemented) | Accuracy | 0.500 [0.300, 0.700] |
| MultiNLI (n=20) | CAS NLIVerifier (as implemented) | Macro-F1 | 0.392 |
| MultiNLI (n=20) | DeBERTa-v3 NLI (CAS-NLI) | Accuracy | 1.000 [1.000, 1.000] |
| MultiNLI (n=20) | DeBERTa-v3 NLI (CAS-NLI) | Macro-F1 | 1.000 |
| MultiNLI | Majority class | Accuracy | 0.400 [0.200, 0.600] |
| RAGTruth (n=50, 12 halluc.) | Unsupported-token rate | AUROC | 0.815 [0.686, 0.924] |
| RAGTruth (n=50, 12 halluc.) | CAS keyword verifier | AUROC | 0.500 [0.500, 0.500] |
| RAGTruth (n=50, 12 halluc.) | DeBERTa-v3 NLI | AUROC | 0.555 [0.367, 0.736] |

## Table 8. End-to-end answers from local LLMs (mean [95% CI])

| Dataset | Model | Method | Accuracy | Token F1 | Abstain | Ungrounded | Latency (s) |
|---|---|---|---|---|---|---|---:|
| HotpotQA | llama3.2:latest | Vanilla RAG (no filter) | 0.600 [0.460, 0.740] | 0.657 [0.526, 0.778] | 0.060 [0.000, 0.140] | 0.000 [0.000, 0.000] | 10.7 |
| HotpotQA | llama3.2:latest | Relevance top-3 (BM25) | 0.540 [0.400, 0.680] | 0.583 [0.449, 0.709] | 0.080 [0.020, 0.160] | 0.047 [0.000, 0.116] | 3.0 |
| HotpotQA | llama3.2:latest | CAS w/o conflict graph | 0.520 [0.380, 0.660] | 0.529 [0.393, 0.669] | 0.080 [0.020, 0.160] | 0.045 [0.000, 0.114] | 2.3 |
| HotpotQA | llama3.2:latest | CAS (full pipeline) | 0.500 [0.360, 0.640] | 0.493 [0.353, 0.633] | 0.120 [0.040, 0.220] | 0.024 [0.000, 0.072] | 3.0 |
| HotpotQA | llama3.2:latest | CAS-NLI (full pipeline) | 0.380 [0.240, 0.520] | 0.400 [0.264, 0.532] | 0.280 [0.160, 0.420] | 0.382 [0.235, 0.530] | 1.4 |
| HotpotQA | llama3.2:latest | CAS-NLI conflict-preserving | 0.520 [0.380, 0.660] | 0.529 [0.393, 0.669] | 0.100 [0.020, 0.180] | 0.047 [0.000, 0.116] | 2.1 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV (penalty) | 0.420 [0.280, 0.560] | 0.432 [0.292, 0.564] | 0.200 [0.100, 0.320] | 0.184 [0.079, 0.316] | 2.0 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV preserving | 0.520 [0.380, 0.660] | 0.529 [0.393, 0.669] | 0.080 [0.020, 0.160] | 0.045 [0.000, 0.114] | 2.4 |
| HotpotQA | gemma3:latest | Vanilla RAG (no filter) | 0.560 [0.420, 0.700] | 0.619 [0.487, 0.743] | 0.080 [0.020, 0.160] | 0.000 [0.000, 0.000] | 13.4 |
| HotpotQA | gemma3:latest | Relevance top-3 (BM25) | 0.540 [0.400, 0.680] | 0.515 [0.380, 0.647] | 0.260 [0.140, 0.380] | 0.028 [0.000, 0.083] | 3.9 |
| HotpotQA | gemma3:latest | CAS w/o conflict graph | 0.540 [0.400, 0.680] | 0.531 [0.399, 0.661] | 0.180 [0.080, 0.280] | 0.049 [0.000, 0.122] | 3.6 |
| HotpotQA | gemma3:latest | CAS (full pipeline) | 0.520 [0.380, 0.660] | 0.531 [0.399, 0.657] | 0.180 [0.080, 0.300] | 0.025 [0.000, 0.075] | 4.4 |
| HotpotQA | gemma3:latest | CAS-NLI (full pipeline) | 0.340 [0.200, 0.480] | 0.336 [0.212, 0.468] | 0.300 [0.180, 0.440] | 0.400 [0.257, 0.571] | 2.3 |
| HotpotQA | gemma3:latest | CAS-NLI conflict-preserving | 0.560 [0.420, 0.700] | 0.535 [0.405, 0.665] | 0.180 [0.080, 0.280] | 0.049 [0.000, 0.122] | 3.7 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV (penalty) | 0.440 [0.300, 0.580] | 0.433 [0.305, 0.561] | 0.240 [0.120, 0.360] | 0.263 [0.132, 0.395] | 3.1 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV preserving | 0.520 [0.380, 0.660] | 0.511 [0.380, 0.645] | 0.180 [0.080, 0.280] | 0.049 [0.000, 0.122] | 3.7 |
| HotpotQA | llama3.1:8b | Vanilla RAG (no filter) | 0.680 [0.560, 0.800] | 0.715 [0.592, 0.829] | 0.080 [0.020, 0.160] | 0.000 [0.000, 0.000] | 24.1 |
| HotpotQA | llama3.1:8b | Relevance top-3 (BM25) | 0.680 [0.540, 0.800] | 0.682 [0.552, 0.799] | 0.160 [0.060, 0.260] | 0.026 [0.000, 0.077] | 7.2 |
| HotpotQA | llama3.1:8b | CAS w/o conflict graph | 0.600 [0.460, 0.740] | 0.599 [0.473, 0.729] | 0.140 [0.060, 0.240] | 0.073 [0.000, 0.171] | 5.0 |
| HotpotQA | llama3.1:8b | CAS (full pipeline) | 0.600 [0.460, 0.740] | 0.609 [0.479, 0.740] | 0.220 [0.100, 0.340] | 0.054 [0.000, 0.135] | 7.2 |
| HotpotQA | llama3.1:8b | CAS-NLI (full pipeline) | 0.360 [0.240, 0.500] | 0.349 [0.226, 0.479] | 0.460 [0.320, 0.600] | 0.423 [0.231, 0.615] | 3.4 |
| HotpotQA | llama3.1:8b | CAS-NLI conflict-preserving | 0.600 [0.460, 0.740] | 0.599 [0.473, 0.729] | 0.160 [0.060, 0.260] | 0.075 [0.000, 0.175] | 4.1 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV (penalty) | 0.500 [0.360, 0.640] | 0.524 [0.384, 0.653] | 0.260 [0.140, 0.380] | 0.222 [0.083, 0.361] | 4.1 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV preserving | 0.600 [0.460, 0.740] | 0.599 [0.473, 0.729] | 0.160 [0.060, 0.260] | 0.075 [0.000, 0.175] | 4.4 |
| MuSiQue | llama3.2:latest | Vanilla RAG (no filter) | 0.333 [0.133, 0.600] | 0.300 [0.100, 0.533] | 0.333 [0.133, 0.600] | 0.000 [0.000, 0.000] | 21.9 |
| MuSiQue | llama3.2:latest | Relevance top-3 (BM25) | 0.133 [0.000, 0.333] | 0.067 [0.000, 0.200] | 0.600 [0.333, 0.867] | 0.000 [0.000, 0.000] | 3.0 |
| MuSiQue | llama3.2:latest | CAS w/o conflict graph | 0.133 [0.000, 0.333] | 0.067 [0.000, 0.200] | 0.600 [0.333, 0.867] | 0.000 [0.000, 0.000] | 3.4 |
| MuSiQue | llama3.2:latest | CAS (full pipeline) | 0.200 [0.000, 0.400] | 0.167 [0.000, 0.367] | 0.467 [0.200, 0.733] | 0.125 [0.000, 0.375] | 4.6 |
| MuSiQue | llama3.2:latest | CAS-NLI (full pipeline) | 0.133 [0.000, 0.333] | 0.067 [0.000, 0.200] | 0.800 [0.600, 1.000] | 0.000 [0.000, 0.000] | 2.3 |
| MuSiQue | llama3.2:latest | CAS-NLI conflict-preserving | 0.067 [0.000, 0.200] | 0.000 [0.000, 0.000] | 0.600 [0.333, 0.867] | 0.000 [0.000, 0.000] | 4.1 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV (penalty) | 0.133 [0.000, 0.333] | 0.100 [0.000, 0.267] | 0.600 [0.333, 0.867] | 0.000 [0.000, 0.000] | 3.2 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV preserving | 0.133 [0.000, 0.333] | 0.067 [0.000, 0.200] | 0.600 [0.333, 0.867] | 0.000 [0.000, 0.000] | 4.1 |
| MuSiQue | gemma3:latest | Vanilla RAG (no filter) | 0.200 [0.000, 0.400] | 0.164 [0.000, 0.351] | 0.600 [0.333, 0.867] | 0.000 [0.000, 0.000] | 24.1 |
| MuSiQue | gemma3:latest | Relevance top-3 (BM25) | 0.133 [0.000, 0.333] | 0.067 [0.000, 0.200] | 0.733 [0.467, 0.933] | 0.000 [0.000, 0.000] | 4.2 |
| MuSiQue | gemma3:latest | CAS w/o conflict graph | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.667 [0.400, 0.867] | 0.000 [0.000, 0.000] | 5.1 |
| MuSiQue | gemma3:latest | CAS (full pipeline) | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.600 [0.333, 0.867] | 0.000 [0.000, 0.000] | 6.5 |
| MuSiQue | gemma3:latest | CAS-NLI (full pipeline) | 0.133 [0.000, 0.333] | 0.053 [0.000, 0.160] | 0.867 [0.667, 1.000] | 0.000 [0.000, 0.000] | 3.3 |
| MuSiQue | gemma3:latest | CAS-NLI conflict-preserving | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.667 [0.400, 0.867] | 0.000 [0.000, 0.000] | 6.3 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV (penalty) | 0.133 [0.000, 0.333] | 0.087 [0.000, 0.214] | 0.733 [0.467, 0.933] | 0.000 [0.000, 0.000] | 5.0 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV preserving | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.667 [0.400, 0.867] | 0.000 [0.000, 0.000] | 6.2 |
| MuSiQue | llama3.1:8b | Vanilla RAG (no filter) | 0.333 [0.133, 0.600] | 0.253 [0.067, 0.467] | 0.533 [0.267, 0.800] | 0.000 [0.000, 0.000] | 49.0 |
| MuSiQue | llama3.1:8b | Relevance top-3 (BM25) | 0.133 [0.000, 0.333] | 0.067 [0.000, 0.200] | 0.733 [0.467, 0.933] | 0.000 [0.000, 0.000] | 7.3 |
| MuSiQue | llama3.1:8b | CAS w/o conflict graph | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.733 [0.467, 0.933] | 0.000 [0.000, 0.000] | 7.8 |
| MuSiQue | llama3.1:8b | CAS (full pipeline) | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.733 [0.467, 0.933] | 0.000 [0.000, 0.000] | 10.6 |
| MuSiQue | llama3.1:8b | CAS-NLI (full pipeline) | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.800 [0.600, 1.000] | 0.000 [0.000, 0.000] | 5.1 |
| MuSiQue | llama3.1:8b | CAS-NLI conflict-preserving | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.733 [0.467, 0.933] | 0.000 [0.000, 0.000] | 8.1 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV (penalty) | 0.133 [0.000, 0.333] | 0.087 [0.000, 0.214] | 0.800 [0.600, 1.000] | 0.000 [0.000, 0.000] | 7.3 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV preserving | 0.200 [0.000, 0.400] | 0.120 [0.000, 0.293] | 0.733 [0.467, 0.933] | 0.000 [0.000, 0.000] | 6.3 |

## Table 9. WikiContradict conflict handling (n = 50)

| Model | Method | Conflict flagged | Both answers | Passages kept |
|---|---|---|---|---:|
| llama3.2:latest | Vanilla RAG (no filter) | 0.480 [0.340, 0.620] | 0.480 [0.340, 0.620] | 2.00 |
| llama3.2:latest | Relevance top-3 (BM25) | 0.480 [0.340, 0.620] | 0.480 [0.340, 0.620] | 2.00 |
| llama3.2:latest | CAS w/o conflict graph | 0.280 [0.160, 0.400] | 0.300 [0.180, 0.420] | 1.44 |
| llama3.2:latest | CAS (full pipeline) | 0.300 [0.180, 0.440] | 0.340 [0.220, 0.480] | 1.62 |
| llama3.2:latest | CAS-NLI (full pipeline) | 0.200 [0.100, 0.320] | 0.160 [0.060, 0.260] | 0.78 |
| llama3.2:latest | CAS-NLI conflict-preserving | 0.440 [0.300, 0.580] | 0.300 [0.180, 0.420] | 1.56 |
| llama3.2:latest | CAS-NLI θ-CV (penalty) | 0.260 [0.140, 0.380] | 0.240 [0.120, 0.360] | 1.26 |
| llama3.2:latest | CAS-NLI θ-CV preserving | 0.320 [0.200, 0.460] | 0.300 [0.180, 0.420] | 1.48 |
| gemma3:latest | Vanilla RAG (no filter) | 0.700 [0.580, 0.820] | 0.480 [0.340, 0.620] | 2.00 |
| gemma3:latest | Relevance top-3 (BM25) | 0.700 [0.580, 0.820] | 0.480 [0.340, 0.620] | 2.00 |
| gemma3:latest | CAS w/o conflict graph | 0.320 [0.180, 0.460] | 0.400 [0.260, 0.540] | 1.44 |
| gemma3:latest | CAS (full pipeline) | 0.420 [0.280, 0.560] | 0.420 [0.280, 0.560] | 1.62 |
| gemma3:latest | CAS-NLI (full pipeline) | 0.600 [0.460, 0.740] | 0.260 [0.140, 0.380] | 0.78 |
| gemma3:latest | CAS-NLI conflict-preserving | 0.440 [0.300, 0.600] | 0.400 [0.260, 0.540] | 1.56 |
| gemma3:latest | CAS-NLI θ-CV (penalty) | 0.380 [0.240, 0.520] | 0.340 [0.200, 0.460] | 1.26 |
| gemma3:latest | CAS-NLI θ-CV preserving | 0.340 [0.220, 0.480] | 0.400 [0.260, 0.540] | 1.48 |
| llama3.1:8b | Vanilla RAG (no filter) | 0.800 [0.680, 0.900] | 0.420 [0.280, 0.560] | 2.00 |
| llama3.1:8b | Relevance top-3 (BM25) | 0.800 [0.680, 0.900] | 0.420 [0.280, 0.560] | 2.00 |
| llama3.1:8b | CAS w/o conflict graph | 0.540 [0.400, 0.680] | 0.300 [0.180, 0.440] | 1.44 |
| llama3.1:8b | CAS (full pipeline) | 0.640 [0.500, 0.760] | 0.320 [0.200, 0.460] | 1.62 |
| llama3.1:8b | CAS-NLI (full pipeline) | 0.280 [0.160, 0.400] | 0.140 [0.060, 0.240] | 0.78 |
| llama3.1:8b | CAS-NLI conflict-preserving | 0.620 [0.480, 0.760] | 0.280 [0.160, 0.420] | 1.56 |
| llama3.1:8b | CAS-NLI θ-CV (penalty) | 0.480 [0.340, 0.620] | 0.220 [0.120, 0.340] | 1.26 |
| llama3.1:8b | CAS-NLI θ-CV preserving | 0.580 [0.440, 0.720] | 0.280 [0.160, 0.420] | 1.48 |

## Table 10. LLM paired tests (reference − comparator)

| Dataset | Model | Reference | Comparator | Metric | Δ mean [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---:|---:|
| HotpotQA | llama3.2:latest | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | -0.164 [-0.282, -0.050] | 0.0085 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.100 [-0.220, +0.000] | 0.1816 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | -0.089 [-0.185, +0.000] | 0.1258 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS (full pipeline) | Relevance top-3 (BM25) | accuracy | -0.040 [-0.120, +0.040] | 0.6276 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS (full pipeline) | CAS w/o conflict graph | f1 | -0.036 [-0.092, +0.000] | 0.4983 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS (full pipeline) | CAS w/o conflict graph | accuracy | -0.020 [-0.060, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | -0.257 [-0.392, -0.120] | 0.0015 | 0.2910 |
| HotpotQA | llama3.2:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.220 [-0.360, -0.080] | 0.0074 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | -0.183 [-0.317, -0.052] | 0.0110 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | accuracy | -0.160 [-0.300, -0.020] | 0.0578 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | -0.129 [-0.256, -0.013] | 0.0483 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | accuracy | -0.140 [-0.280, -0.020] | 0.0916 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | -0.128 [-0.257, +0.001] | 0.0662 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | accuracy | -0.080 [-0.200, +0.040] | 0.3438 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | -0.053 [-0.133, +0.020] | 0.3805 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | accuracy | -0.020 [-0.080, +0.040] | 1.0000 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | -0.225 [-0.374, -0.083] | 0.0062 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | accuracy | -0.180 [-0.320, -0.040] | 0.0383 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | -0.151 [-0.271, -0.028] | 0.0314 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | accuracy | -0.120 [-0.240, +0.000] | 0.1141 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | -0.097 [-0.195, -0.007] | 0.0813 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | accuracy | -0.100 [-0.200, +0.000] | 0.1290 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | -0.128 [-0.257, +0.001] | 0.0662 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | accuracy | -0.080 [-0.200, +0.040] | 0.3438 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | -0.053 [-0.133, +0.020] | 0.3805 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | accuracy | -0.020 [-0.080, +0.040] | 1.0000 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.2:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | -0.088 [-0.220, +0.048] | 0.2768 | 1.0000 |
| HotpotQA | gemma3:latest | CAS (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.040 [-0.160, +0.080] | 0.7562 | 1.0000 |
| HotpotQA | gemma3:latest | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | +0.015 [-0.090, +0.115] | 0.8260 | 1.0000 |
| HotpotQA | gemma3:latest | CAS (full pipeline) | Relevance top-3 (BM25) | accuracy | -0.020 [-0.100, +0.060] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS (full pipeline) | CAS w/o conflict graph | f1 | +0.000 [-0.060, +0.060] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS (full pipeline) | CAS w/o conflict graph | accuracy | -0.020 [-0.100, +0.040] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | -0.283 [-0.404, -0.165] | 0.0001 | 0.0210 |
| HotpotQA | gemma3:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.220 [-0.340, -0.120] | 0.0008 | 0.1568 |
| HotpotQA | gemma3:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | -0.179 [-0.322, -0.042] | 0.0150 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | accuracy | -0.200 [-0.340, -0.060] | 0.0206 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | -0.195 [-0.313, -0.082] | 0.0029 | 0.5509 |
| HotpotQA | gemma3:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | accuracy | -0.200 [-0.320, -0.080] | 0.0062 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | -0.084 [-0.232, +0.056] | 0.3117 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | accuracy | +0.000 [-0.120, +0.120] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | +0.019 [-0.084, +0.123] | 0.7841 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | accuracy | +0.020 [-0.080, +0.120] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | +0.004 [-0.048, +0.060] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | accuracy | +0.020 [+0.000, +0.060] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | -0.185 [-0.329, -0.052] | 0.0182 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | accuracy | -0.120 [-0.260, +0.000] | 0.1518 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | -0.082 [-0.212, +0.042] | 0.2195 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | accuracy | -0.100 [-0.240, +0.020] | 0.2350 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | -0.097 [-0.184, -0.024] | 0.0326 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | accuracy | -0.100 [-0.180, -0.020] | 0.0665 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | -0.108 [-0.236, +0.012] | 0.1553 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | accuracy | -0.040 [-0.160, +0.060] | 0.7336 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | -0.005 [-0.114, +0.101] | 0.9179 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | accuracy | -0.020 [-0.120, +0.080] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | -0.020 [-0.060, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | gemma3:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | accuracy | -0.020 [-0.060, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | -0.105 [-0.201, -0.020] | 0.0239 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.080 [-0.160, -0.020] | 0.1268 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | -0.073 [-0.161, +0.020] | 0.1413 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS (full pipeline) | Relevance top-3 (BM25) | accuracy | -0.080 [-0.180, +0.000] | 0.2228 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS (full pipeline) | CAS w/o conflict graph | f1 | +0.010 [+0.000, +0.030] | 1.0000 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS (full pipeline) | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | -0.366 [-0.516, -0.222] | 0.0001 | 0.0210 |
| HotpotQA | llama3.1:8b | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.320 [-0.460, -0.200] | 0.0001 | 0.0210 |
| HotpotQA | llama3.1:8b | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | -0.333 [-0.475, -0.199] | 0.0001 | 0.0210 |
| HotpotQA | llama3.1:8b | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | accuracy | -0.320 [-0.460, -0.180] | 0.0001 | 0.0210 |
| HotpotQA | llama3.1:8b | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | -0.251 [-0.386, -0.127] | 0.0007 | 0.1386 |
| HotpotQA | llama3.1:8b | CAS-NLI (full pipeline) | CAS w/o conflict graph | accuracy | -0.240 [-0.380, -0.120] | 0.0015 | 0.2910 |
| HotpotQA | llama3.1:8b | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | -0.115 [-0.211, -0.030] | 0.0170 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | accuracy | -0.080 [-0.160, -0.020] | 0.1268 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | -0.083 [-0.168, +0.000] | 0.0917 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | accuracy | -0.080 [-0.180, +0.000] | 0.2228 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI conflict-preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | -0.191 [-0.323, -0.063] | 0.0081 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | accuracy | -0.180 [-0.300, -0.080] | 0.0051 | 0.9536 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | -0.158 [-0.281, -0.033] | 0.0195 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | accuracy | -0.180 [-0.320, -0.040] | 0.0254 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | -0.075 [-0.181, +0.024] | 0.1723 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | accuracy | -0.100 [-0.200, +0.000] | 0.1290 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | -0.115 [-0.211, -0.030] | 0.0170 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | accuracy | -0.080 [-0.160, -0.020] | 0.1268 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | -0.083 [-0.168, +0.000] | 0.0917 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | accuracy | -0.080 [-0.180, +0.000] | 0.2228 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| HotpotQA | llama3.1:8b | CAS-NLI θ-CV preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | -0.133 [-0.333, +0.033] | 0.3761 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.133 [-0.333, +0.000] | 0.4985 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | +0.100 [+0.000, +0.267] | 0.5022 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS (full pipeline) | Relevance top-3 (BM25) | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS (full pipeline) | CAS w/o conflict graph | f1 | +0.100 [+0.000, +0.267] | 0.5022 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS (full pipeline) | CAS w/o conflict graph | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | -0.233 [-0.500, +0.033] | 0.2232 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.200 [-0.467, +0.067] | 0.3787 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | -0.300 [-0.533, -0.100] | 0.0654 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | accuracy | -0.267 [-0.467, -0.067] | 0.1302 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | accuracy | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | accuracy | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | -0.200 [-0.400, -0.033] | 0.1250 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | accuracy | -0.200 [-0.400, +0.000] | 0.2483 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | +0.033 [-0.100, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | +0.033 [-0.100, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | -0.233 [-0.467, -0.033] | 0.1315 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | accuracy | -0.200 [-0.400, +0.000] | 0.2615 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.2:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | -0.044 [-0.133, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS (full pipeline) | Vanilla RAG (no filter) | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | +0.053 [+0.000, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS (full pipeline) | Relevance top-3 (BM25) | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS (full pipeline) | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS (full pipeline) | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | -0.111 [-0.267, +0.000] | 0.5032 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | -0.013 [-0.200, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI (full pipeline) | CAS w/o conflict graph | accuracy | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | -0.044 [-0.133, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | +0.053 [+0.000, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI conflict-preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | -0.078 [-0.189, +0.000] | 0.5032 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | accuracy | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | +0.020 [-0.080, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | -0.033 [-0.100, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | accuracy | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | -0.044 [-0.133, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | +0.053 [+0.000, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | gemma3:latest | CAS-NLI θ-CV preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS (full pipeline) | Vanilla RAG (no filter) | f1 | -0.133 [-0.333, +0.000] | 0.5082 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.133 [-0.333, +0.000] | 0.5082 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS (full pipeline) | Relevance top-3 (BM25) | f1 | +0.053 [+0.000, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS (full pipeline) | Relevance top-3 (BM25) | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS (full pipeline) | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS (full pipeline) | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | f1 | -0.133 [-0.400, +0.067] | 0.6247 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | accuracy | -0.133 [-0.400, +0.067] | 0.6247 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | f1 | +0.053 [-0.134, +0.253] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI (full pipeline) | Relevance top-3 (BM25) | accuracy | +0.067 [-0.133, +0.267] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI (full pipeline) | CAS w/o conflict graph | f1 | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI (full pipeline) | CAS w/o conflict graph | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | f1 | -0.133 [-0.333, +0.000] | 0.5082 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | accuracy | -0.133 [-0.333, +0.000] | 0.5082 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | f1 | +0.053 [+0.000, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI conflict-preserving | Relevance top-3 (BM25) | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI conflict-preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI conflict-preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | f1 | -0.167 [-0.367, +0.000] | 0.2540 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | accuracy | -0.200 [-0.400, +0.000] | 0.2540 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | f1 | +0.020 [-0.080, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV (penalty) | Relevance top-3 (BM25) | accuracy | +0.000 [-0.200, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | f1 | -0.033 [-0.100, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV (penalty) | CAS w/o conflict graph | accuracy | -0.067 [-0.200, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | f1 | -0.133 [-0.333, +0.000] | 0.5082 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | accuracy | -0.133 [-0.333, +0.000] | 0.5082 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | f1 | +0.053 [+0.000, +0.160] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV preserving | Relevance top-3 (BM25) | accuracy | +0.067 [+0.000, +0.200] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV preserving | CAS w/o conflict graph | f1 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| MuSiQue | llama3.1:8b | CAS-NLI θ-CV preserving | CAS w/o conflict graph | accuracy | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 |
| WikiContradict | llama3.2:latest | CAS (full pipeline) | Vanilla RAG (no filter) | conflict_flagged | -0.180 [-0.300, -0.060] | 0.0114 | 1.0000 |
| WikiContradict | llama3.2:latest | CAS (full pipeline) | Vanilla RAG (no filter) | both | -0.140 [-0.240, -0.060] | 0.0159 | 1.0000 |
| WikiContradict | llama3.2:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | conflict_flagged | -0.280 [-0.440, -0.120] | 0.0016 | 0.3072 |
| WikiContradict | llama3.2:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | both | -0.320 [-0.460, -0.200] | 0.0001 | 0.0210 |
| WikiContradict | llama3.2:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | conflict_flagged | -0.040 [-0.220, +0.140] | 0.8196 | 1.0000 |
| WikiContradict | llama3.2:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | both | -0.180 [-0.280, -0.080] | 0.0040 | 0.7559 |
| WikiContradict | llama3.2:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | conflict_flagged | -0.220 [-0.380, -0.080] | 0.0108 | 1.0000 |
| WikiContradict | llama3.2:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | both | -0.240 [-0.360, -0.120] | 0.0007 | 0.1386 |
| WikiContradict | llama3.2:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | conflict_flagged | -0.160 [-0.320, +0.000] | 0.0938 | 1.0000 |
| WikiContradict | llama3.2:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | both | -0.180 [-0.280, -0.080] | 0.0040 | 0.7559 |
| WikiContradict | gemma3:latest | CAS (full pipeline) | Vanilla RAG (no filter) | conflict_flagged | -0.280 [-0.420, -0.160] | 0.0001 | 0.0210 |
| WikiContradict | gemma3:latest | CAS (full pipeline) | Vanilla RAG (no filter) | both | -0.060 [-0.140, +0.000] | 0.2594 | 1.0000 |
| WikiContradict | gemma3:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | conflict_flagged | -0.100 [-0.260, +0.060] | 0.3284 | 1.0000 |
| WikiContradict | gemma3:latest | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | both | -0.220 [-0.340, -0.120] | 0.0014 | 0.2730 |
| WikiContradict | gemma3:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | conflict_flagged | -0.260 [-0.440, -0.100] | 0.0058 | 1.0000 |
| WikiContradict | gemma3:latest | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | both | -0.080 [-0.160, -0.020] | 0.1285 | 1.0000 |
| WikiContradict | gemma3:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | conflict_flagged | -0.320 [-0.480, -0.180] | 0.0004 | 0.0796 |
| WikiContradict | gemma3:latest | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | both | -0.140 [-0.240, -0.060] | 0.0165 | 1.0000 |
| WikiContradict | gemma3:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | conflict_flagged | -0.360 [-0.500, -0.220] | 0.0001 | 0.0210 |
| WikiContradict | gemma3:latest | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | both | -0.080 [-0.160, -0.020] | 0.1285 | 1.0000 |
| WikiContradict | llama3.1:8b | CAS (full pipeline) | Vanilla RAG (no filter) | conflict_flagged | -0.160 [-0.300, -0.040] | 0.0382 | 1.0000 |
| WikiContradict | llama3.1:8b | CAS (full pipeline) | Vanilla RAG (no filter) | both | -0.100 [-0.200, +0.000] | 0.1244 | 1.0000 |
| WikiContradict | llama3.1:8b | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | conflict_flagged | -0.520 [-0.660, -0.380] | 0.0001 | 0.0210 |
| WikiContradict | llama3.1:8b | CAS-NLI (full pipeline) | Vanilla RAG (no filter) | both | -0.280 [-0.400, -0.160] | 0.0003 | 0.0600 |
| WikiContradict | llama3.1:8b | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | conflict_flagged | -0.180 [-0.300, -0.040] | 0.0225 | 1.0000 |
| WikiContradict | llama3.1:8b | CAS-NLI conflict-preserving | Vanilla RAG (no filter) | both | -0.140 [-0.240, -0.040] | 0.0396 | 1.0000 |
| WikiContradict | llama3.1:8b | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | conflict_flagged | -0.320 [-0.460, -0.180] | 0.0002 | 0.0402 |
| WikiContradict | llama3.1:8b | CAS-NLI θ-CV (penalty) | Vanilla RAG (no filter) | both | -0.200 [-0.320, -0.100] | 0.0019 | 0.3629 |
| WikiContradict | llama3.1:8b | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | conflict_flagged | -0.220 [-0.360, -0.080] | 0.0079 | 1.0000 |
| WikiContradict | llama3.1:8b | CAS-NLI θ-CV preserving | Vanilla RAG (no filter) | both | -0.140 [-0.240, -0.040] | 0.0396 | 1.0000 |
