**TABLE_MAIN_A. Main results: answers on clean retrieval (study 6, pre-registered; HotpotQA n = 700, MuSiQue n = 200).** Share of questions answered correctly (acc.) and wrongly; the rest are abstentions. † significant against Vanilla RAG in the pre-registered test (Holm p < 0.05); bold = best in the column.

| Method | HotpotQA Llama 3.2 acc. ↑ | HotpotQA Llama 3.2 wrong ↓ | HotpotQA Gemma 3 acc. ↑ | HotpotQA Gemma 3 wrong ↓ | MuSiQue Llama 3.2 acc. ↑ | MuSiQue Llama 3.2 wrong ↓ | MuSiQue Gemma 3 acc. ↑ | MuSiQue Gemma 3 wrong ↓ |
|---|---|---|---|---|---|---|---|---|
| Vanilla RAG | 0.507 | 0.454 | 0.411 | 0.480 | 0.150 | 0.595 | 0.160 | 0.500 |
| Reranker top-3 | 0.441 | 0.433 | 0.370 | 0.400 | 0.120 | 0.465 | 0.150 | 0.390 |
| CAS v2 | 0.554† | **0.371†** | 0.470† | **0.373†** | 0.165 | **0.440†** | 0.170 | **0.360†** |
| CAS v2 + fallback | **0.574†** | 0.400 | **0.503†** | 0.417 | **0.195** | 0.600 | **0.230†** | 0.525 |


**TABLE_MAIN_B. Clean retrieval with every baseline (study 5, pre-registered; HotpotQA n = 200).** Bold = best in the column; for the full-evidence share, best among the filters, because Vanilla RAG keeps every passage by construction. † significant against Vanilla RAG in the pre-registered test (Holm p < 0.05).

| Method | Llama 3.2 acc. ↑ | Llama 3.2 wrong ↓ | Gemma 3 acc. ↑ | Gemma 3 wrong ↓ | Evidence F1 ↑ | All supporting kept ↑ |
|---|---|---|---|---|---|---|
| Vanilla RAG | 0.485 | 0.500 | 0.395 | 0.520 | 0.337 | 1.000 |
| BM25 top-3 | 0.450 | 0.450 | 0.350 | **0.420** | 0.625 | 0.580 |
| Reranker top-3 | 0.445 | 0.430 | 0.355 | 0.430 | 0.631 | 0.585 |
| CAS v1 | 0.415 | 0.430 | 0.315 | 0.440 | 0.647 | 0.485 |
| CAS v2 | **0.550** | **0.405†** | **0.450** | 0.440 | **0.744** | **0.910** |


**TABLE_MAIN_C. Poisoning (study 5, pre-registered; n = 150 per setting).** Adoption: the fake answer was given. Pre-registered decisions are in Table 2.

| Setting | Method | Llama 3.2 adoption ↓ | Llama 3.2 acc. ↑ | Gemma 3 adoption ↓ | Gemma 3 acc. ↑ |
|---|---|---|---|---|---|
| Copy attack, no metadata | Vanilla RAG | 0.273 | 0.247 | 0.340 | 0.213 |
|  | CAS v1 | 0.333 | 0.187 | 0.393 | 0.153 |
|  | CAS v2 | 0.000 | 0.073 | 0.000 | 0.073 |
| Copy attack, noisy metadata | Source filter | 0.047 | 0.313 | 0.047 | 0.247 |
|  | CAS v2 | 0.033 | 0.280 | 0.033 | 0.213 |
| Written attack, no metadata | Vanilla RAG | 0.307 | 0.107 | 0.333 | 0.087 |
|  | CAS v2 | 0.213 | 0.147 | 0.240 | 0.127 |
| Written attack, noisy metadata | Source filter | 0.113 | 0.253 | 0.140 | 0.220 |
|  | CAS v2 | 0.107 | 0.240 | 0.127 | 0.273 |


**TABLE_MAIN_D. Cost per question** (CPU only; means). Selection is the filter's own time; generation is the LLM call. For CAS v2 + fallback, prompt tokens and generation time include the second call when the model abstained.

| Method | Data | Selection (s) | Prompt tokens (Llama 3.2 / Gemma 3) | Generation (s) (Llama 3.2 / Gemma 3) |
|---|---|---|---|---|
| Vanilla RAG | HotpotQA, study 5 | < 0.0001 | 1398 / 1430 | 10.1 / 12.2 |
| BM25 top-3 | HotpotQA, study 5 | 0.0006 | 436 / 435 | 2.5 / 3.8 |
| Reranker top-3 | HotpotQA, study 5 | 0.18 | 426 / 425 | 2.5 / 3.7 |
| CAS v1 | HotpotQA, study 5 | 19.8 | 420 / 419 | 2.1 / 3.6 |
| CAS v2 | HotpotQA, study 5 | 0.0031 | 487 / 488 | 2.3 / 4.1 |
| Vanilla RAG | HotpotQA, study 6 | < 0.0001 | 1402 / 1429 | 10.0 / 12.2 |
| Reranker top-3 | HotpotQA, study 6 | 0.15 | 430 / 428 | 2.5 / 3.8 |
| CAS v2 | HotpotQA, study 6 | 0.0027 | 488 / 487 | 2.6 / 4.1 |
| CAS v2 + fallback | HotpotQA, study 6 | 0.0027 | 588 / 696 | 3.3 / 5.9 |
| Vanilla RAG | MuSiQue, study 6 | < 0.0001 | 2464 / 2542 | 19.9 / 21.5 |
| Reranker top-3 | MuSiQue, study 6 | 0.30 | 461 / 461 | 3.1 / 4.1 |
| CAS v2 | MuSiQue, study 6 | 0.0067 | 898 / 911 | 6.0 / 7.6 |
| CAS v2 + fallback | MuSiQue, study 6 | 0.0067 | 1872 / 2071 | 13.3 / 17.2 |
