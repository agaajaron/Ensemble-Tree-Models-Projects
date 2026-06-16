# %% [markdown]
# # Model Evaluation and Comparison

# %%
from train import *
from utils import model_performance_classification_sklearn, confusion_matrix_sklearn

# ── Collect all models for comparison ────────────────────────────────────────
# %%
model_dict = {
    "DT Weighted":          (dt_weighted,         X_train, X_test, y_train, y_test),
    "DT Balanced":          (dt_balanced,          X_train, X_test, y_train, y_test),
    "DT Tuned Wide":        (dt_tuned_wide,        X_train, X_test, y_train, y_test),
    "DT Tuned Constrained": (dt_tuned_constrained, X_train, X_test, y_train, y_test),
    "RF":                   (rfo,                  X_train, X_test, y_train, y_test),
    "RF Tuned":             (rfo_tuned,            X_train, X_test, y_train, y_test),
    "RF (outlier-treated)": (rfo_,                 X_train3, X_test3, y_train3, y_test3),
    "RF Tuned (outlier)":   (rfo_tuned_,           X_train3, X_test3, y_train3, y_test3),
    "Bagging":              (bco,                  X_train, X_test, y_train, y_test),
    "Bagging Tuned":        (bco_tuned,            X_train, X_test, y_train, y_test),
    "AdaBoost":             (abco,                 X_train, X_test, y_train, y_test),
    "AdaBoost Tuned":       (abco_tuned,           X_train, X_test, y_train, y_test),
    "AdaBoost (outlier)":   (abco_,                X_train3, X_test3, y_train3, y_test3),
    "AdaBoost Tuned (out)": (abco_tuned_,          X_train3, X_test3, y_train3, y_test3),
    "GBM":                  (gbco,                 X_train, X_test, y_train, y_test),
    "GBM Tuned":            (gbco_tuned,           X_train, X_test, y_train, y_test),
    "GBM (outlier)":        (gbco_,                X_train3, X_test3, y_train3, y_test3),
    "GBM Tuned (out)":      (gbco_tuned_,          X_train3, X_test3, y_train3, y_test3),
    "XGBoost":              (xgbo,                 X_train, X_test, y_train, y_test),
    "XGBoost Tuned":        (xgbo_tuned,           X_train, X_test, y_train, y_test),
    "XGBoost (outlier)":    (xgbo_,                X_train3, X_test3, y_train3, y_test3),
    "XGBoost Tuned (out)":  (xgbo_tuned_,          X_train3, X_test3, y_train3, y_test3),
    "Stacking":             (stacking,             X_train, X_test, y_train, y_test),
    "Stacking (outlier)":   (stacking_,            X_train3, X_test3, y_train3, y_test3),
}

# %%
train_rows, test_rows = [], []
for name, (m, X_tr, X_te, y_tr, y_te) in model_dict.items():
    tr = model_performance_classification_sklearn(m, X_tr, y_tr)
    te = model_performance_classification_sklearn(m, X_te, y_te)
    tr.insert(0, "Model", name)
    te.insert(0, "Model", name)
    train_rows.append(tr)
    test_rows.append(te)

# %%
print("Training performance — all models:")
pd.concat(train_rows, ignore_index=True)

# %%
print("Testing performance — all models:")
pd.concat(test_rows, ignore_index=True)

# %%
print("Best model: Stacking — test confusion matrix:")
confusion_matrix_sklearn(stacking, X_test, y_test)

# %%
print("Classification report — Stacking:")
print(classification_report(y_test, stacking.predict(X_test)))
