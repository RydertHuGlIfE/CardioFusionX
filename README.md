# CardioFusionX - ECG

## Patient History Risk Module

The separate patient-history module uses the UCI Heart Disease files in
`heart+disease/processed.{cleveland,hungarian,switzerland,va}.data`. The source
documentation is `heart+disease/heart-disease.names`. Each processed row has 13
predictors (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`,
`exang`, `oldpeak`, `slope`, `ca`, `thal`) and the original `num` target.

The saved model uses the documented binary interpretation `num > 0` as presence
versus `num == 0` as absence. Missing `?`, `-9`, and `-9.0` values are imputed
inside a leakage-safe scikit-learn `Pipeline`: median/scaled numeric features and
most-frequent/one-hot categorical features, followed by Logistic Regression.

Train it once from the project root:

```bash
venv/bin/python train_patient_history.py
```

Artifacts are written to `models/patient_history/`, including the joblib model,
feature metadata, metrics, training configuration, confusion matrix, and ROC plot.
The current reproducible holdout evaluation used a stratified 75/25 split (`n=230`
test rows): accuracy `0.848`, balanced accuracy `0.846`, sensitivity `0.866`,
specificity `0.825`, F1 `0.863`, ROC-AUC `0.909`, PR-AUC `0.912`, and Brier score
`0.119`. Five-fold CV ROC-AUC was `0.891 +/- 0.021`; these are dataset-based
research metrics, not clinical validation.

The Flask endpoint is `POST /predict-history`. It is independent of ECG waveform
inference and does not combine probabilities. The UI intentionally does not claim
to analyze free-text symptoms because the UCI data contains no such feature.
For severe or urgent symptoms, contact local emergency services rather than relying
on this application. Neither module is a medical diagnostic system.
