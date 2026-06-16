# %% [markdown]
# # Model Training — Visa Application Prediction
#
# Models are trained on two datasets:
#   • Regular dataset: X_train / X_test
#   • Outlier-treated dataset: X_train3 / X_test3
# GridSearchCV uses 50% subsamples for speed.

# %%
from preprocessing import *
from model import *

# ── Decision Trees ────────────────────────────────────────────────────────────
# %%
dt_weighted = build_dt_weighted()
dt_weighted.fit(X_train, y_train)
print("DT Weighted trained.")

dt_balanced = build_dt_balanced()
dt_balanced.fit(X_train, y_train)
print("DT Balanced trained.")

print("Tuning DT (wide search)...")
dt_tuned_wide = build_dt_tuned_wide(X_train_sub, y_train_sub)
dt_tuned_wide.fit(X_train, y_train)

print("Tuning DT (constrained)...")
dt_tuned_constrained = build_dt_tuned_constrained(X_train_sub, y_train_sub)
dt_tuned_constrained.fit(X_train, y_train)

# ── Random Forest ─────────────────────────────────────────────────────────────
# %%
rfo = build_rf()
rfo.fit(X_train, y_train)
print("RF trained.")

print("Tuning RF...")
rfo_tuned = build_rf_tuned(X_train_sub, y_train_sub)
rfo_tuned.fit(X_train, y_train)

# Outlier-treated RF
rfo_ = build_rf()
rfo_.fit(X_train3, y_train3)
print("RF (outlier-treated) trained.")

print("Tuning RF (outlier-treated)...")
rfo_tuned_ = build_rf_tuned(X_train3_sub, y_train3_sub)
rfo_tuned_.fit(X_train3, y_train3)

# ── Bagging ───────────────────────────────────────────────────────────────────
# %%
bco = build_bagging()
bco.fit(X_train, y_train)
print("Bagging trained.")

print("Tuning Bagging...")
bco_tuned = build_bagging_tuned(X_train_sub, y_train_sub)
bco_tuned.fit(X_train, y_train)

# ── AdaBoost ─────────────────────────────────────────────────────────────────
# %%
abco = build_adaboost()
abco.fit(X_train, y_train)
print("AdaBoost trained.")

print("Tuning AdaBoost...")
abco_tuned = build_adaboost_tuned(X_train_sub, y_train_sub)
abco_tuned.fit(X_train, y_train)

# Outlier-treated AdaBoost
abco_ = build_adaboost()
abco_.fit(X_train3, y_train3)

print("Tuning AdaBoost (outlier-treated)...")
abco_tuned_ = build_adaboost_tuned(X_train3_sub, y_train3_sub)
abco_tuned_.fit(X_train3, y_train3)

# ── Gradient Boosting ─────────────────────────────────────────────────────────
# %%
gbco = build_gbm()
gbco.fit(X_train, y_train)
print("GBM trained.")

print("Tuning GBM...")
gbco_tuned = build_gbm_tuned(X_train_sub, y_train_sub)
gbco_tuned.fit(X_train, y_train)

# Outlier-treated GBM
gbco_ = build_gbm()
gbco_.fit(X_train3, y_train3)

print("Tuning GBM (outlier-treated)...")
gbco_tuned_ = build_gbm_tuned(X_train3_sub, y_train3_sub)
gbco_tuned_.fit(X_train3, y_train3)

# ── XGBoost ──────────────────────────────────────────────────────────────────
# %%
xgbo = build_xgb()
xgbo.fit(X_train, y_train)
print("XGBoost trained.")

print("Tuning XGBoost...")
xgbo_tuned = build_xgb_tuned(X_train_sub, y_train_sub)
xgbo_tuned.fit(X_train, y_train)

# Outlier-treated XGBoost
xgbo_ = build_xgb()
xgbo_.fit(X_train3, y_train3)

print("Tuning XGBoost (outlier-treated)...")
xgbo_tuned_ = build_xgb_tuned(X_train3_sub, y_train3_sub)
xgbo_tuned_.fit(X_train3, y_train3)

# ── Stacking ─────────────────────────────────────────────────────────────────
# %%
print("Training Stacking Classifier (RF + GBM + DT → XGB)...")
stacking = build_stacking(rfo_tuned, gbco_tuned, dt_tuned_constrained, X_train, y_train)
print("Stacking trained.")

print("Training Stacking Classifier (outlier-treated)...")
stacking_ = build_stacking(rfo_tuned_, gbco_tuned_, dt_tuned_constrained, X_train3, y_train3)
print("All models trained.")
