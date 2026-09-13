# HealthConnect Week 6

Wilson Moses | AnalystLab Africa | Data Science Track

This package advances the Week 5 baseline through model comparison, error analysis, feature refinement and local pipeline integration. The selected candidate is **no-history logistic regression**, chosen on internal temporal validation. The model is an educational prototype using fictional appointment records.

## Deliverables

| Location | Purpose |
|---|---|
| `reports/` | Portfolio report, project summary and AI-simulated integration evidence in Field Notes PDF format |
| `notebooks/03_model_improvement_and_validation.ipynb` | Executed analysis with methodology notes and learning checkpoints |
| `src/week6_analysis.py` | Reproduce the model comparison, metrics, figures and serialized candidate |
| `src/healthconnect.py` | Validate raw inputs and score with the saved pipeline |
| `models/candidate_pipeline.joblib` | Preprocessing, classifier, model version and scenario threshold together |
| `integration/` | Sample input/output, 17 interface checks and Week 7 test plan |
| `outputs/metrics/` | Fold definitions, model comparison, error profiles, threshold scenarios and metadata |
| `outputs/figures/` | Evidence charts |
| `data/raw/` | Unchanged appointment source |
| `reference/` | Preserved Week 5 comparison outputs |
| `LEARNING_GUIDE_WEEKS_4_TO_6.md` | Reasoning, IBM stage connections and exercises |

## Reproduce

Use Python 3.12 and install `requirements.txt` in a virtual environment. From this package root:

```bash
python -m pip install -r requirements.txt
python src/week6_analysis.py
python integration/test_interface.py
python src/healthconnect.py --input integration/sample_input.csv --model models/candidate_pipeline.joblib --output integration/scored_example.csv
```

Or run the notebook from its folder in PyCharm or Jupyter. Source data remains unchanged; generated Week 6 outputs are replaced on rerun. Notebook outputs were captured by sequential Python execution in one clean namespace. Only load trusted local joblib artifacts.

## Findings and limits

Internal mean ROC-AUC is 0.6481 for no-history logistic versus 0.6416 for the Week 5 baseline. On the **reused** 799-record comparison set, candidate AUC is 0.6653, recall 67.18% and precision 58.56% at 0.5. This is a trade-off, not superiority on every metric or independent final validation.

The serialized threshold 0.600 reflects an illustrative maximum 30% validation-workload scenario. On the reused comparison it flags 193 records, achieving precision 70.47% and recall 35.14%. The clinic has not approved this policy. The selected model does not require history fields.

Cross-track evidence is an **AI-simulated ML Engineering review requested by the user** with implemented code and testing changes. It is not a human intern exchange; the programme must confirm whether it meets its collaboration requirement. No operational deployment, attendance benefit, official form submission, Google Drive upload or GitHub commit is claimed.
