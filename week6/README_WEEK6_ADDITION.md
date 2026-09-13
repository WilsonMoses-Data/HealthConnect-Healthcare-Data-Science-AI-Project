## Week 6 model improvement and integration

Week 6 extends the Week 5 baseline through temporal validation, feature refinement, error analysis and a tested scoring interface. The selected candidate is **no-history logistic regression**, which avoids unverified history inputs and achieved the highest mean internal ROC-AUC (0.6481 versus 0.6416 for the Week 5 baseline).

On the previously examined 799-record comparison set, the candidate achieved ROC-AUC 0.6653 and recall 67.18% at threshold 0.5. It caught 14 more no-shows overall while producing 12 more false positives than the baseline. These are descriptive comparisons, not fresh validation or superiority on every metric.

The AI-simulated ML Engineering review produced a seven-field raw-input contract, a serialized preprocessing/model pipeline and **17 passing integration checks**. The scenario threshold of 0.600 is hypothetical and is not an approved clinic action policy. Programme acceptance of AI-simulated cross-track work remains to be confirmed.

- [Week 6 notebook](week6/notebooks/03_model_improvement_and_validation.ipynb)
- [Model improvement and validation report](week6/reports/healthconnect_week6_model_improvement_validation_report.pdf)
- [Project summary](week6/reports/healthconnect_week6_project_summary.pdf)
- [AI-simulated integration evidence](week6/reports/healthconnect_week6_cross_track_integration_evidence.pdf)
- [Week 7 testing requirements](week6/integration/WEEK7_TEST_PLAN.md)

From the repository root:

```bash
cd week6
python -m pip install -r requirements.txt
python src/week6_analysis.py
python integration/test_interface.py
```

Current phase: model validation and local integration. Next: fresh temporal evaluation, operational capacity agreement, cancellation-policy review and broader Week 7 testing. The project remains an educational prototype using fictional data.
