# %% [markdown]
# # Model Definitions
#
# Six models for diabetes risk prediction (binary classification):
#   1. DecisionTreeClassifier (default)
#   2. DecisionTreeClassifier (tuned with GridSearchCV, class_weight)
#   3. RandomForestClassifier (default)
#   4. RandomForestClassifier (tuned with GridSearchCV, class_weight)
#   5. BaggingClassifier (default)
#   6. BaggingClassifier (tuned with GridSearchCV)
#
# Tuning metric: Recall (maximize detection of true diabetics)

# %%
from preprocessing import *


def build_decision_tree():
    return DecisionTreeClassifier(random_state=1)


def build_decision_tree_tuned(X_train, y_train):
    dtree = DecisionTreeClassifier(class_weight={0: 0.35, 1: 0.65}, random_state=1)
    parameters = {
        'max_depth': np.arange(2, 10),
        'min_samples_leaf': [5, 7, 10, 15],
        'max_leaf_nodes': [2, 3, 5, 10, 15],
        'min_impurity_decrease': [0.0001, 0.001, 0.01, 0.1],
    }
    scorer = metrics.make_scorer(metrics.recall_score)
    grid = GridSearchCV(dtree, parameters, scoring=scorer, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best.fit(X_train, y_train)
    return best


def build_random_forest():
    return RandomForestClassifier(random_state=1)


def build_random_forest_tuned(X_train, y_train):
    rf = RandomForestClassifier(class_weight={0: 0.35, 1: 0.65}, random_state=1)
    parameters = {
        'max_depth': list(np.arange(3, 10, 1)),
        'max_features': np.arange(0.6, 1.1, 0.1),
        'max_samples': np.arange(0.7, 1.1, 0.1),
        'min_samples_split': np.arange(2, 20, 5),
        'n_estimators': np.arange(30, 160, 20),
        'min_impurity_decrease': [0.0001, 0.001, 0.01, 0.1],
    }
    scorer = metrics.make_scorer(metrics.recall_score)
    grid = GridSearchCV(rf, parameters, scoring=scorer, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best.fit(X_train, y_train)
    return best


def build_bagging():
    return BaggingClassifier(random_state=1)


def build_bagging_tuned(X_train, y_train):
    bg = BaggingClassifier(random_state=1)
    parameters = {
        'max_samples': [0.7, 0.8, 0.9, 1],
        'max_features': [0.7, 0.8, 0.9, 1],
        'n_estimators': [10, 20, 30, 40, 50],
    }
    scorer = metrics.make_scorer(metrics.recall_score)
    grid = GridSearchCV(bg, parameters, scoring=scorer, cv=5)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best.fit(X_train, y_train)
    return best
