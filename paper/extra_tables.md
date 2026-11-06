**Table 11. Standard answer metrics: exact match (EM) and token F1** (SQuAD normalisation, best over the gold aliases; exploratory). * CAS v2 or v2 + fallback differs from Vanilla RAG on the same questions (paired sign-flip test, Holm p < 0.05 within each study); bold = best within each block.

| Data | Method | Llama 3.2 EM ↑ | Llama 3.2 F1 ↑ | Gemma 3 EM ↑ | Gemma 3 F1 ↑ |
|---|---|---|---|---|---|
| HotpotQA, study 6 (n = 700) | Vanilla RAG | 0.469 | 0.603 | 0.387 | 0.506 |
|  | Reranker top-3 | 0.401 | 0.531 | 0.334 | 0.438 |
|  | CAS v2 | 0.514 | 0.652* | 0.434 | 0.556* |
|  | CAS v2 + fallback | **0.533*** | **0.678*** | **0.467*** | **0.596*** |
| MuSiQue, study 6 (n = 200) | Vanilla RAG | 0.135 | 0.220 | 0.145 | 0.205 |
|  | Reranker top-3 | 0.110 | 0.174 | 0.140 | 0.192 |
|  | CAS v2 | 0.150 | 0.216 | 0.165 | 0.217 |
|  | CAS v2 + fallback | **0.175** | **0.264** | **0.220*** | **0.288*** |
| HotpotQA, study 5 (n = 200) | Vanilla RAG | 0.470 | 0.623 | 0.385 | 0.517 |
|  | BM25 top-3 | 0.420 | 0.549 | 0.330 | 0.472 |
|  | Reranker top-3 | 0.430 | 0.556 | 0.325 | 0.469 |
|  | CAS v1 | 0.400 | 0.523 | 0.300 | 0.441 |
|  | CAS v2 | **0.515** | **0.662** | **0.430** | **0.595*** |


**Table 12. Abstention-aware scores** (study 6; exploratory). Coverage = share of questions answered; selective accuracy = accuracy among answered questions; net score = correct − wrong (a correct answer scores +1, an abstention 0, a wrong answer −1). * net score differs from Vanilla RAG's (paired sign-flip test, Holm p < 0.05 over the 10 comparisons with Vanilla RAG in studies 5 and 6).

| Data | Method | Llama 3.2 coverage | Llama 3.2 selective acc. ↑ | Llama 3.2 net score ↑ | Gemma 3 coverage | Gemma 3 selective acc. ↑ | Gemma 3 net score ↑ |
|---|---|---|---|---|---|---|---|
| HotpotQA, study 6 (n = 700) | Vanilla RAG | 0.961 | 0.527 | +0.053 | 0.891 | 0.462 | −0.069 |
|  | Reranker top-3 | 0.874 | 0.505 | +0.009 | 0.770 | 0.481 | −0.030 |
|  | CAS v2 | 0.926 | 0.599 | +0.183* | 0.843 | 0.558 | +0.097* |
|  | CAS v2 + fallback | 0.974 | 0.589 | +0.174* | 0.920 | 0.547 | +0.086* |
| MuSiQue, study 6 (n = 200) | Vanilla RAG | 0.745 | 0.201 | −0.445 | 0.660 | 0.242 | −0.340 |
|  | Reranker top-3 | 0.585 | 0.205 | −0.345 | 0.540 | 0.278 | −0.240 |
|  | CAS v2 | 0.605 | 0.273 | −0.275* | 0.530 | 0.321 | −0.190* |
|  | CAS v2 + fallback | 0.795 | 0.245 | −0.405 | 0.755 | 0.305 | −0.295 |


**Table 13. How each question's outcome moved from Vanilla RAG to CAS v2** (number of questions; exploratory). The last two columns ask whether v2's new abstentions fell on questions that Vanilla RAG got wrong. They compare Vanilla RAG's accuracy on the questions where v2 abstained and Vanilla RAG answered with its accuracy on every question it answered.

| Data | Model | Wrong → correct | Correct → wrong | Wrong → abstain | Correct → abstain | Abstain → correct / wrong | Net change in correct | Vanilla RAG accuracy on v2's new abstentions | Vanilla RAG accuracy where it answered |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA, study 6 (n = 700) | Llama 3.2 | 79 | 37 | 20 | 18 | 9 / 4 | +33 | 0.474 (n = 38) | 0.527 |
|  | Gemma 3 | 79 | 29 | 32 | 26 | 17 / 7 | +41 | 0.448 (n = 58) | 0.462 |
| MuSiQue, study 6 (n = 200) | Llama 3.2 | 11 | 5 | 30 | 6 | 3 / 5 | +3 | 0.167 (n = 36) | 0.201 |
|  | Gemma 3 | 12 | 5 | 32 | 12 | 7 / 11 | +2 | 0.273 (n = 44) | 0.242 |
| HotpotQA, study 5 (n = 200) | Llama 3.2 | 21 | 7 | 6 | 2 | 1 / 1 | +13 | 0.250 (n = 8) | 0.492 |
|  | Gemma 3 | 18 | 8 | 13 | 3 | 4 / 7 | +11 | 0.188 (n = 16) | 0.432 |


**Table 14. CAS v2 minus Vanilla RAG by question type** (study 6; same questions; 95% bootstrap CIs; exploratory, not corrected for multiplicity). The interval [0.000, 0.000] for Llama 3.2 on 4-hop questions means that the two methods answered exactly the same questions correctly.

| Data | Question type | n | Llama 3.2 Δ accuracy | Llama 3.2 Δ wrong | Gemma 3 Δ accuracy | Gemma 3 Δ wrong |
|---|---|---|---|---|---|---|
| HotpotQA | bridge | 555 | +0.029 [−0.009, +0.065] | −0.077 [−0.114, −0.040] | +0.065 [+0.029, +0.101] | −0.112 [−0.148, −0.074] |
|  | comparison | 145 | +0.117 [+0.048, +0.186] | −0.103 [−0.166, −0.041] | +0.034 [−0.041, +0.110] | −0.090 [−0.166, −0.021] |
| MuSiQue | 2 hops | 95 | +0.011 [−0.063, +0.084] | −0.189 [−0.284, −0.095] | +0.042 [−0.042, +0.126] | −0.116 [−0.221, −0.011] |
|  | 3 hops | 66 | +0.030 [−0.061, +0.121] | −0.121 [−0.242, −0.015] | −0.030 [−0.136, +0.076] | −0.152 [−0.288, −0.015] |
|  | 4 hops | 39 | 0.000 [0.000, 0.000] | −0.128 [−0.282, +0.026] | 0.000 [−0.154, +0.154] | −0.179 [−0.333, −0.026] |
