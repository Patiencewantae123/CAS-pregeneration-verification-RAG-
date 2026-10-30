# Pre-registration: study 2, a stronger poisoning attack (HotpotQA-Poison-2)

Written 2026-10-30, after study 1 (`PREREGISTRATION_poison.md`) was completed and **before** this benchmark was
built or any method was run on it. This is a separate study: study 1 is reported in full, and is not replaced.
The SHA-256 of this file is recorded in `results/poison2_summary.json`.

## Why a second study

In study 1 the models rarely adopted the substituted answer: 3.5–6.5% even with no filtering. The substitutes were
matched only on whether they contained digits, so they were often implausible. That left no room to measure a
reduction. Study 2 uses plausible substitutes and several poison passages per question, following the
PoisonedRAG black-box design (Zou et al. 2024), in which the poison text is prefixed with the question.

## Data

- Source: the HotpotQA dev set (distractor setting).
- Sample: **200 new questions**, seed 42. The filters are the same as in study 1, and the sample excludes the 50 earlier questions and the 200 questions of study 1.
- Substitute answer (type-matched):
  - Year (four digits, 1000–2099): the year shifted by a seeded random 3–15 years, up or down.
  - Other answers containing digits: every digit is replaced by a seeded random digit (the first digit is non-zero), giving a different value.
  - Otherwise: the title of a *distractor* paragraph from the same question, with any trailing parenthetical removed. Titles with word count within ±1 of the answer are preferred; otherwise any distractor title is used.
  - The substitute must differ from the answer and must not occur in any supporting paragraph.
- Poison passages, **3 per question**, all asserting the substitute:
  1. The first supporting paragraph that contains the answer, with every occurrence of the answer replaced.
  2. The question, followed by passage 1. This is the PoisonedRAG S⊕I form.
  3. The paragraph's title, followed by only the sentences of passage 1 that contain the substitute.
- Each poison passage is inserted at a seeded random position.
- In this regime the poison outnumbers the true passage 3 to 1, so methods that reward agreement are expected to favour the poison.

## Conditions, methods, models, endpoints, test, decision rule

## Amendment 1 (2026-10-30, after inspecting built items, before any method was run)

I inspected the first six built items. Two of the substitute rules produced implausible text: dates came out as
"January 37, 8376", and a stadium was replaced by a person's name. The rules are changed to the following:
- **Dates** (a month name with a four-digit year): the year is shifted by 3–15 years, and any day number is replaced by a random day from 1 to 28.
- **Other numbers:** the first number is multiplied by a seeded random factor between 0.5 and 1.8, and its comma format is kept.
- **Entity titles:** prefer distractor titles that share the answer's last word, for example "Stadium" or "Battle". If there are none, apply the word-count rule above.

Nothing else changes.

## Conditions, methods, models, endpoints, test, decision rule

All are identical to study 1:
- Conditions: A (no metadata) and B (noisy provenance, same ranges, 0.2 flip probability).
- Methods: the same seven, with τ = 0.45 and θ = 0.99. Nothing is tuned.
- Models: Llama 3.2 3B and Gemma 3 4B.
- Endpoints: primary is poison adoption; co-primary is accuracy.
- Test: 20 confirmatory paired tests with Holm correction.
- Decision rule: the same as study 1.

Every result is reported.
