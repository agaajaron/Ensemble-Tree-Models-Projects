# %% [markdown]
# # Model Training

# %%
from model import *

# ── Logistic Regression (statsmodels) ────────────────────────────────────────
# %%
print("Training statsmodels Logistic Regression (full features)...")
logit_full = build_logit_full()
print(logit_full.summary())

# %%
print("Checking VIF for multicollinearity...")
vif_df = checking_vif(X_train1.drop(columns=['const'], errors='ignore'))
print(vif_df)

# Columns with VIF > 5 to drop (market_segment columns found to be multicollinear)
high_vif_cols = vif_df[vif_df["VIF"] > 5]["feature"].tolist()

# %%
print("Training statsmodels Logit — VIF-pruned...")
logit_vif, X_train_vif, X_test_vif = build_logit_vif_pruned(high_vif_cols)

# %%
print("Training statsmodels Logit — iterative p-value pruning...")
logit_pval, X_train_pval, X_test_pval = build_logit_pvalue_pruned()

# %%
print("Training statsmodels Logit — regularized...")
logit_reg = build_logit_regularized()

# ── Logistic Regression (sklearn) ────────────────────────────────────────────
# %%
print("Training sklearn LogisticRegression (full features)...")
logreg = build_logreg_sklearn()

print("Training sklearn LogisticRegression (VIF-pruned)...")
logreg_vif, X_train_lr_vif, X_test_lr_vif = build_logreg_sklearn_vif_pruned(high_vif_cols)

# ── Decision Trees ────────────────────────────────────────────────────────────
# %%
print("Training Decision Tree models...")
dt_default = build_decision_tree_default()
dt_depth7 = build_decision_tree_depth7()
dt_depth6 = build_decision_tree_depth6()
dt_balanced = build_decision_tree_balanced()

print("Tuning Decision Tree (GridSearchCV)...")
dt_tuned = build_decision_tree_tuned()

print("Pruning Decision Tree (cost complexity)...")
dt_pruned, best_alpha, train_f1s, test_f1s = build_decision_tree_pruned()

print("All models trained.")
