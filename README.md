# LLM Guardrail Evaluation Lab

Evaluating three input-guardrail approaches for an enterprise security assistant, plus a four-layer policy prompt design.

The assistant should answer legitimate security questions (API key rotation, redaction, secure development) and block attempts to extract secrets, bypass controls, steal credentials or expose private data.

## What this project does

| Component | Description |
|---|---|
| **Regex-only guardrail** | Intent-based patterns (verb + sensitive object + context), not bare keywords, so "How do I rotate an API key?" stays SAFE |
| **Semantic-only guardrail** | `all-MiniLM-L6-v2` embeddings; cosine similarity against 12 unsafe reference examples; threshold-based decision |
| **Combined guardrail** | Regex high-confidence hit -> UNSAFE; otherwise semantic threshold (0.55) |
| **Policy prompt layers** | `ETHICAL`, `BRAND`, `COMPLIANCE`, `RESPONSE` policies combined into one system prompt |
| **Evaluation** | TP/FP/TN/FN, precision, recall, F1, FPR, FNR, accuracy, threshold experiments, confusion matrices |

Dataset: 30 labelled prompts (12 UNSAFE, 18 SAFE) provided by the course instructor. Positive class = UNSAFE.

## Results

| Approach | Threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR | FNR | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex-only | N/A | 11 | 1 | 17 | 1 | 0.917 | 0.917 | 0.917 | 0.056 | 0.083 | 0.933 |
| Semantic-only | 0.40 | 12 | 6 | 12 | 0 | 0.667 | 1.000 | 0.800 | 0.333 | 0.000 | 0.800 |
| Semantic-only | 0.55 | 9 | 1 | 17 | 3 | 0.900 | 0.750 | 0.818 | 0.056 | 0.250 | 0.867 |
| Semantic-only | 0.70 | 3 | 0 | 18 | 9 | 1.000 | 0.250 | 0.400 | 0.000 | 0.750 | 0.700 |
| Combined | 0.55 | 12 | 2 | 16 | 0 | 0.857 | 1.000 | 0.923 | 0.111 | 0.000 | 0.933 |

![Threshold comparison](threshold_comparison.png)

### Key findings

- **Threshold trade-off:** raising the semantic threshold from 0.40 to 0.70 removed all false positives (6 -> 0) but missed attacks rose from 0 to 9. At 0.70 the guardrail looks "clean" only because it blocks almost nothing.
- **Combined was the strongest setup on this data:** it caught all 12 unsafe prompts with 2 false positives, while regex alone missed 1 attack and semantic alone (0.55) missed 3.
- **Accuracy alone is misleading:** regex-only and combined both score 0.933 accuracy, but they differ in what they miss. For a security assistant, a missed attack (potential credential leak) usually costs more than a blocked legitimate request, so recall and FNR matter alongside FPR.

## Policy prompt layers

`policy_prompts.py` defines the four layers, assembles them with `build_system_prompt()`, and includes one SAFE and one UNSAFE test per layer. With an OpenAI API key, it runs each test against a live model twice (single layer vs all four layers) and saves the replies to `policy_layer_live_results.json`.

> Policy prompts are not a security boundary. They can be overridden by prompt injection and behave non-deterministically, so they sit alongside input guardrails, output scanning and application-side authorisation.

## Project structure

```
.
├── guardrail_dataset.csv          # instructor-provided dataset
├── guardrail_evaluation.py        # regex, semantic, combined guardrails + metrics + charts
├── policy_prompts.py              # four policy layers + optional live LLM tests
├── guardrail_predictions.csv      # per-prompt predictions and scores
├── guardrail_metrics.csv          # metrics per approach / threshold
├── confusion_matrices/            # one PNG per approach
├── threshold_comparison.png       # recall and FPR vs semantic threshold
├── recommendation_report.md       # analysis and recommendation
├── requirements.txt
└── README.md
```

## Setup and run

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Mac/Linux:           source .venv/bin/activate

pip install -r requirements.txt

python guardrail_evaluation.py --backend minilm
python policy_prompts.py
```

The first MiniLM run downloads the model, so it needs internet access.

Optional live policy demo (uses a dummy-data test set; the key is read from an environment variable and never stored in the repo):

```powershell
# PowerShell
$env:OPENAI_API_KEY="your_key"
python policy_prompts.py
```

```bash
# Mac/Linux
export OPENAI_API_KEY="your_key"
python policy_prompts.py
```

## Limitations

- **Small dataset:** with 30 prompts, one misclassification shifts recall by about 8 points. Treat differences as suggestive, not statistically significant.
- **Possible overfitting:** the regex patterns and semantic reference examples were written while the dataset's style was visible. The references are paraphrases, not copies of test prompts, but a held-out set would give a fairer estimate.
- **Model-specific thresholds:** the 0.40/0.55/0.70 values apply to `all-MiniLM-L6-v2`; other embedding models score differently.
- **Input-side only:** there is no output guardrail yet.

## Next steps

- Evaluate on a larger, independently written dataset.
- Add an output guardrail that scans responses for secrets and PII.
- Add a "review" band between two thresholds instead of a hard block.
- Compare against an LLM-based classifier.

## Acknowledgements

Dataset and assignment brief provided by the course instructor.
