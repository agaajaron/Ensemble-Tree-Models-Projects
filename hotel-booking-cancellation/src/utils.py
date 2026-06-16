# %%
from config import *


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins)
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")


def labeled_barplot(data, feature, perc=False, n=None):
    total = len(data[feature])
    count = data[feature].nunique()
    plt.figure(figsize=((n or count) + 1, 5))
    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data, x=feature, palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )
    for p in ax.patches:
        label = "{:.1f}%".format(100 * p.get_height() / total) if perc else p.get_height()
        ax.annotate(label, (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="center", size=12, xytext=(0, 5),
                    textcoords="offset points")
    plt.show()


# ── Statsmodels helpers ──────────────────────────────────────────────────────
def model_performance_classification_statsmodels(model, predictors, target, threshold=0.5):
    pred_probas = model.predict(predictors)
    pred_labels = (pred_probas > threshold).astype(int)
    acc = accuracy_score(target, pred_labels)
    recall = recall_score(target, pred_labels)
    precision = precision_score(target, pred_labels)
    f1 = f1_score(target, pred_labels)
    auc_score = roc_auc_score(target, pred_probas)
    df = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1, "AUC-ROC": auc_score},
        index=[0],
    )
    return df


def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    pred_probas = model.predict(predictors)
    pred_labels = (pred_probas > threshold).astype(int)
    cm = confusion_matrix(target, pred_labels)
    labels = np.asarray(
        [["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
         for item in cm.flatten()]
    ).reshape(2, 2)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


# ── Sklearn helpers ──────────────────────────────────────────────────────────
def model_performance_classification_sklearn(model, predictors, target):
    pred = model.predict(predictors)
    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred)
    precision = precision_score(target, pred)
    f1 = f1_score(target, pred)
    auc_score = roc_auc_score(target, model.predict_proba(predictors)[:, 1])
    return pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1, "AUC-ROC": auc_score},
        index=[0],
    )


def make_confusion_matrix(model, y_actual, y_predicted, labels=None):
    if labels is None:
        labels = ["Not Canceled (0)", "Canceled (1)"]
    cm = confusion_matrix(y_actual, y_predicted)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.show()


def get_recall_score(model, flag=True, threshold=0.5):
    if flag:
        y_pred = (model.predict_proba(X_train)[:, 1] >= threshold).astype(int)
        score = recall_score(y_train, y_pred)
    else:
        y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        score = recall_score(y_test, y_pred)
    return round(score, 2)


def get_f1_score(model, flag=True, threshold=0.5):
    if flag:
        y_pred = (model.predict_proba(X_train)[:, 1] >= threshold).astype(int)
        score = f1_score(y_train, y_pred)
    else:
        y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        score = f1_score(y_test, y_pred)
    return round(score, 2)


def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i)
        for i in range(predictors.shape[1])
    ]
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif
