# Local Model Artifacts

This folder stores trained or generated model artifacts for local runs and
deploy branches. These files are intentionally ignored for GitHub pushes.

Common local files:

- `ltr_model.txt`
- `ltr_model.txt.weights.json`
- `transformer_encoder.pt`
- `bandit_model.pkl`

Train the LambdaMART model locally with:

```powershell
python backend/train_ltr.py --auto-export
```

The runtime can still start without a trained LTR model because
`backend/ltr.py` includes a deterministic fallback scorer.
