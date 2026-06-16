# %% [markdown]
# # Model Definitions
#
# Models trained for hotel booking cancellation (binary classification):
#
# ── Logistic Regression (statsmodels) ───────────────────────────────────────
#   1. sm.Logit — full feature set
#   2. sm.Logit — VIF-pruned features (removed multicollinear market_segment cols)
#   3. sm.Logit — iterative p-value pruning
#   4. sm.Logit — regularized (L1, alpha=1)
#
# ── Logistic Regression (sklearn) ───────────────────────────────────────────
#   5. LogisticRegression — full feature set
#   6. LogisticRegression — VIF-pruned features
#
# ── Decision Tree ───────────────────────────────────────────────────────────
#   7. DecisionTreeClassifier — default
#   8. DecisionTreeClassifier(max_depth=7) — best F1 / generalized
#   9. DecisionTreeClassifier(max_depth=6)
#  10. DecisionTreeClassifier(class_weight="balanced")
#  11. DecisionTreeClassifier + GridSearchCV (recall scorer)
#  12. DecisionTreeClassifier + cost complexity pruning
#
# Tuning metric: F1-score (balanced recall and precision)
# Root node of best tree: lead_time (>150 days → ~75% canceled)

# %%
from preprocessing import *
from utils import checking_vif


def build_logit_full():
    return sm.Logit(y_train, X_train1).fit()


def build_logit_vif_pruned(vif_cols_to_drop):
    X_tr = X_train1.drop(columns=vif_cols_to_drop, errors='ignore')
    X_te = X_test1.drop(columns=vif_cols_to_drop, errors='ignore')
    model = sm.Logit(y_train, X_tr).fit()
    return model, X_tr, X_te


def build_logit_pvalue_pruned(threshold_pvalue=0.05):
    cols = list(X_train1.columns)
    while True:
        model = sm.Logit(y_train, X_train1[cols]).fit()
        pvals = model.pvalues.drop('const', errors='ignore')
        max_pval = pvals.max()
        if max_pval > threshold_pvalue:
            cols.remove(pvals.idxmax())
        else:
            break
    X_tr = X_train1[cols]
    X_te = X_test1[cols]
    return model, X_tr, X_te


def build_logit_regularized():
    return sm.Logit(y_train, X_train1).fit_regularized(alpha=1, L1_wt=1)


def build_logreg_sklearn():
    model = LogisticRegression(max_iter=1000, random_state=1)
    model.fit(X_train, y_train)
    return model


def build_logreg_sklearn_vif_pruned(vif_cols_to_drop):
    X_tr = X_train.drop(columns=vif_cols_to_drop, errors='ignore')
    X_te = X_test.drop(columns=vif_cols_to_drop, errors='ignore')
    model = LogisticRegression(max_iter=1000, random_state=1)
    model.fit(X_tr, y_train)
    return model, X_tr, X_te


def build_decision_tree_default():
    m = DecisionTreeClassifier(random_state=1)
    m.fit(X_train, y_train)
    return m


def build_decision_tree_depth7():
    m = DecisionTreeClassifier(max_depth=7, random_state=1)
    m.fit(X_train, y_train)
    return m


def build_decision_tree_depth6():
    m = DecisionTreeClassifier(max_depth=6, random_state=1)
    m.fit(X_train, y_train)
    return m


def build_decision_tree_balanced():
    m = DecisionTreeClassifier(class_weight="balanced", random_state=1)
    m.fit(X_train, y_train)
    return m


def build_decision_tree_tuned():
    dtree = DecisionTreeClassifier(random_state=1)
    params = {
        'max_depth': np.arange(2, 10),
        'min_samples_leaf': [5, 7, 10, 15],
        'max_leaf_nodes': [2, 3, 5, 10, 15],
        'min_impurity_decrease': [0.0001, 0.001, 0.01, 0.1],
    }
    scorer = metrics.make_scorer(metrics.f1_score)
    grid = GridSearchCV(dtree, params, scoring=scorer, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best.fit(X_train, y_train)
    return best


def build_decision_tree_pruned():
    path = DecisionTreeClassifier(random_state=1).cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[::5]  # sample every 5th
    scorer = metrics.make_scorer(metrics.f1_score)
    trees = []
    for alpha in ccp_alphas:
        m = DecisionTreeClassifier(random_state=1, ccp_alpha=alpha)
        m.fit(X_train, y_train)
        trees.append(m)
    train_scores = [metrics.f1_score(y_train, m.predict(X_train)) for m in trees]
    test_scores = [metrics.f1_score(y_test, m.predict(X_test)) for m in trees]
    best_idx = np.argmax(test_scores)
    return trees[best_idx], ccp_alphas[best_idx], train_scores, test_scores
