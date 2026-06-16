# %% [markdown]
# # Model Definitions
#
# 14+ model configurations for visa certification prediction:
#
# ── Decision Tree ────────────────────────────────────────────────────────────
#   1. DecisionTreeClassifier(class_weight={0:0.67, 1:0.33})
#   2. DecisionTreeClassifier(class_weight="balanced")
#   3. DecisionTreeClassifier + GridSearchCV (max_depth 2-30, F1 scorer)
#   4. DecisionTreeClassifier + GridSearchCV (constrained, max_depth 5-7)
#
# ── Random Forest ────────────────────────────────────────────────────────────
#   5. RandomForestClassifier — default
#   6. RandomForestClassifier + GridSearchCV
#
# ── Bagging ─────────────────────────────────────────────────────────────────
#   7. BaggingClassifier — default
#   8. BaggingClassifier + GridSearchCV (DTree base_estimator)
#
# ── Boosting ─────────────────────────────────────────────────────────────────
#   9. AdaBoostClassifier — default (no overfitting, high scores)
#  10. AdaBoostClassifier + GridSearchCV (DTree base, max_depth 1-3)
#  11. GradientBoostingClassifier — default
#  12. GradientBoostingClassifier + GridSearchCV
#  13. XGBClassifier(eval_metric='logloss') — default
#  14. XGBClassifier + GridSearchCV  ← best individual model
#
# ── Stacking ─────────────────────────────────────────────────────────────────
#  15. StackingClassifier(RF + GBC + DTree → XGB) ← best overall
#
# All models repeated on outlier-treated dataset (variants with _o suffix)
# Tuning metric: F1-score

# %%
from preprocessing import *

# ── Decision Tree ─────────────────────────────────────────────────────────────

def build_dt_weighted():
    return DecisionTreeClassifier(class_weight={0: 0.67, 1: 0.33}, random_state=1)


def build_dt_balanced():
    return DecisionTreeClassifier(class_weight="balanced", random_state=1)


def build_dt_tuned_wide(X_tr, y_tr):
    scorer = metrics.make_scorer(metrics.f1_score)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=1),
        {'max_depth': np.arange(2, 30)},
        scoring=scorer, cv=5, n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best.fit(X_tr, y_tr)
    return best


def build_dt_tuned_constrained(X_tr, y_tr):
    scorer = metrics.make_scorer(metrics.f1_score)
    params = {
        'max_depth': [5, 6, 7],
        'min_samples_leaf': [5, 10, 20],
        'max_leaf_nodes': [5, 10, 20, 50],
        'min_impurity_decrease': [0.0001, 0.001],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=1), params, scoring=scorer, cv=5, n_jobs=-1
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best.fit(X_tr, y_tr)
    return best


# ── Random Forest ─────────────────────────────────────────────────────────────

def build_rf():
    return RandomForestClassifier(random_state=1)


def build_rf_tuned(X_tr, y_tr):
    scorer = metrics.make_scorer(metrics.f1_score)
    params = {
        'max_depth': list(np.arange(3, 15, 1)),
        'max_features': np.arange(0.6, 1.1, 0.1),
        'max_samples': np.arange(0.7, 1.1, 0.1),
        'min_samples_split': np.arange(2, 20, 5),
        'n_estimators': np.arange(30, 160, 20),
        'min_impurity_decrease': [0.0001, 0.001, 0.01, 0.1],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=1), params, scoring=scorer, cv=5, n_jobs=-1
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best.fit(X_tr, y_tr)
    return best


# ── Bagging ───────────────────────────────────────────────────────────────────

def build_bagging():
    return BaggingClassifier(random_state=1)


def build_bagging_tuned(X_tr, y_tr):
    scorer = metrics.make_scorer(metrics.f1_score)
    base = DecisionTreeClassifier(random_state=1)
    params = {
        'base_estimator__max_depth': [1, 2, 3],
        'max_samples': [0.7, 0.8, 0.9, 1],
        'max_features': [0.7, 0.8, 0.9, 1],
        'n_estimators': [10, 20, 30, 40, 50],
    }
    grid = GridSearchCV(
        BaggingClassifier(base_estimator=base, random_state=1),
        params, scoring=scorer, cv=5, n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best.fit(X_tr, y_tr)
    return best


# ── AdaBoost ─────────────────────────────────────────────────────────────────

def build_adaboost():
    return AdaBoostClassifier(random_state=1)


def build_adaboost_tuned(X_tr, y_tr):
    scorer = metrics.make_scorer(metrics.f1_score)
    base = DecisionTreeClassifier(random_state=1)
    params = {
        'base_estimator__max_depth': [1, 2, 3],
        'n_estimators': np.arange(50, 500, 50),
        'learning_rate': [0.01, 0.1, 0.05, 1],
    }
    grid = GridSearchCV(
        AdaBoostClassifier(base_estimator=base, random_state=1),
        params, scoring=scorer, cv=5, n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best.fit(X_tr, y_tr)
    return best


# ── Gradient Boosting ────────────────────────────────────────────────────────

def build_gbm():
    return GradientBoostingClassifier(random_state=1)


def build_gbm_tuned(X_tr, y_tr):
    scorer = metrics.make_scorer(metrics.f1_score)
    params = {
        'max_depth': np.arange(1, 5),
        'n_estimators': np.arange(50, 500, 50),
        'learning_rate': np.arange(0.1, 0.6, 0.1),
        'subsample': np.arange(0.5, 1.1, 0.1),
        'min_samples_leaf': [1, 2, 5, 10],
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=1), params, scoring=scorer, cv=5, n_jobs=-1
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best.fit(X_tr, y_tr)
    return best


# ── XGBoost ──────────────────────────────────────────────────────────────────

def build_xgb():
    return XGBClassifier(eval_metric='logloss', random_state=1)


def build_xgb_tuned(X_tr, y_tr):
    scorer = metrics.make_scorer(metrics.f1_score)
    params = {
        'n_estimators': np.arange(50, 500, 50),
        'scale_pos_weight': [1, 2, 5],
        'subsample': np.arange(0.5, 1.1, 0.1),
        'learning_rate': np.arange(0.01, 0.21, 0.05),
        'gamma': [0, 1, 3],
        'reg_lambda': [1, 2, 5],
        'max_depth': np.arange(1, 8),
    }
    grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=1),
        params, scoring=scorer, cv=5, n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    best.fit(X_tr, y_tr)
    return best


# ── Stacking ─────────────────────────────────────────────────────────────────

def build_stacking(rf_model, gbm_model, dt_model, X_tr, y_tr):
    """
    Stacking ensemble: RF + GBM + DTree base learners → XGBoost meta-learner.
    Best overall model.
    """
    estimators = [
        ("Random Forest", rf_model),
        ("Gradient Boosting", gbm_model),
        ("Decision Tree", dt_model),
    ]
    final_estimator = XGBClassifier(eval_metric='logloss', random_state=1)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=False,
    )
    stacking.fit(X_tr, y_tr)
    return stacking
