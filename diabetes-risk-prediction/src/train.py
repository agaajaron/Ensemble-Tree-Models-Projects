# %% [markdown]
# # Model Training

# %%
from preprocessing import *
from model import (
    build_decision_tree, build_decision_tree_tuned,
    build_random_forest, build_random_forest_tuned,
    build_bagging, build_bagging_tuned,
)

# %%
d_tree = build_decision_tree()
d_tree.fit(X_train, y_train)
print("Decision Tree trained.")

rf_estimator = build_random_forest()
rf_estimator.fit(X_train, y_train)
print("Random Forest trained.")

bagging_classifier = build_bagging()
bagging_classifier.fit(X_train, y_train)
print("Bagging Classifier trained.")

# %%
print("Tuning Decision Tree (GridSearchCV)...")
dtree_estimator = build_decision_tree_tuned(X_train, y_train)

print("Tuning Random Forest (GridSearchCV)...")
rf_tuned = build_random_forest_tuned(X_train, y_train)

print("Tuning Bagging Classifier (GridSearchCV)...")
bagging_estimator_tuned = build_bagging_tuned(X_train, y_train)

print("All models trained.")
