# %% [markdown]
# # Model Evaluation and Comparison

# %%
from train import *
from utils import (
    model_performance_classification_statsmodels,
    model_performance_classification_sklearn,
    confusion_matrix_statsmodels,
    make_confusion_matrix,
    get_recall_score,
    get_f1_score,
)

# ── Statsmodels Logit performance ─────────────────────────────────────────────
# %%
print("Logit Full — train:")
model_performance_classification_statsmodels(logit_full, X_train1, y_train)

# %%
print("Logit Full — test:")
model_performance_classification_statsmodels(logit_full, X_test1, y_test)

# ── Optimal threshold via ROC curve ──────────────────────────────────────────
# %%
y_pred_proba = logit_full.predict(X_test1)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.3f}")

# ── Decision Tree performance ─────────────────────────────────────────────────
# %%
dt_models = {
    "DT Default": dt_default,
    "DT Depth=7": dt_depth7,
    "DT Depth=6": dt_depth6,
    "DT Balanced": dt_balanced,
    "DT Tuned": dt_tuned,
    "DT Pruned": dt_pruned,
}

train_rows, test_rows = [], []
for name, m in dt_models.items():
    tr = model_performance_classification_sklearn(m, X_train, y_train)
    te = model_performance_classification_sklearn(m, X_test, y_test)
    tr.insert(0, "Model", name)
    te.insert(0, "Model", name)
    train_rows.append(tr)
    test_rows.append(te)

# %%
print("Decision Tree — Train performance:")
pd.concat(train_rows, ignore_index=True)

# %%
print("Decision Tree — Test performance:")
pd.concat(test_rows, ignore_index=True)

# %%
print("Best model (DT Depth=7) test confusion matrix:")
make_confusion_matrix(dt_depth7, y_test, dt_depth7.predict(X_test))

# %%
print("Pruned tree best alpha:", best_alpha)
print(export_text(dt_pruned, feature_names=list(X_train.columns)))
