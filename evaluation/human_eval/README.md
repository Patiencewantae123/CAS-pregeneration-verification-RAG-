# Human validation of the automatic answer metrics

Reviewers will ask whether string matching measures answer quality. This folder checks that against human judgement.

## How to label (about 2 hours)

1. Open `annotation_sheet.csv` in Excel. **Do not open `annotation_key.csv`**: it holds the automatic scores, and the labelling must be blind.
2. For each row, read the question, the reference answer and the model answer, then fill in:
   - **correct**: 1 if the model answer means the same as the reference answer, otherwise 0. Accept paraphrases, aliases and extra correct detail, for example "the 45th Infantry Division" for "45th Infantry Division". Mark 0 for wrong, partial-but-wrong, or "unknown".
   - **abstains**: 1 if the model declines to answer ("unknown", "not mentioned" and similar), otherwise 0.
   - **gives_fake_answer**: only on rows with a fake answer. Mark 1 if the model answer asserts the fake answer, otherwise 0.
   - **notes**: optional. Use it for unclear cases.
3. Judge from the text alone. Don't look anything up.
4. Save as CSV, keeping the same file name and format, then run:

```bash
py evaluation/human_eval/score_agreement.py
```

The script reports agreement and Cohen's κ, with a 95% CI, for each automatic metric. Results are written to `agreement.json`.

## Reporting

- κ ≥ 0.8 is usually reported as strong agreement. A lower κ must be reported too, and it limits how far the automatic metrics can be trusted.
- If a second person labels the same sheet independently, inter-annotator κ can also be reported. Ask me to add that.
