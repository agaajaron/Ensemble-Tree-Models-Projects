# %% [markdown]
# # Model Evaluation and Comparison

# %%
from train import *
from utils import model_performance_classification_sklearn, confusion_matrix_sklearn

model_names = [
    "Decision Tree", "Decision Tree Tuned",
    "Random Forest", "Random Forest Tuned",
    "Bagging Classifier", "Bagging Tuned",
]
models = [d_tree, dtree_estimator, rf_estimator, rf_tuned, bagging_classifier, bagging_estimator_tuned]

# %%
train_perfs, test_perfs = [], []
for name, m in zip(model_names, models):
    tr = model_performance_classification_sklearn(m, X_train, y_train)
    te = model_performance_classification_sklearn(m, X_test, y_test)
    train_perfs.append(tr.T.rename(columns={0: name}))
    test_perfs.append(te.T.rename(columns={0: name}))

# %%
print("Training performance comparison:")
pd.concat(train_perfs, axis=1)

# %%
print("Testing performance comparison:")
pd.concat(test_perfs, axis=1)

# %%
# Feature importance — tuned decision tree (best model)
feature_names = list(X_train.columns)
print(tree.export_text(dtree_estimator, feature_names=feature_names, show_weights=True))
