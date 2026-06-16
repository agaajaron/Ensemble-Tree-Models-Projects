# %% [markdown]
# # Visualization — EDA and Model Results

# %%
from evaluate import *
from utils import histogram_boxplot, labeled_barplot, stacked_barplot, confusion_matrix_sklearn

# ── EDA — distributions ───────────────────────────────────────────────────────
# %%
for col in ['no_of_employees', 'prevailing_wage', 'wage_fin']:
    histogram_boxplot(data, col)

# %%
for col in ['continent', 'education_of_employee', 'has_job_experience',
            'requires_job_training', 'region_of_employment',
            'full_time_position', 'case_status']:
    labeled_barplot(data, col, perc=True)

# %%
# Stacked barplots: outcome rate by categorical feature
for col in ['continent', 'education_of_employee', 'has_job_experience',
            'requires_job_training', 'region_of_employment', 'full_time_position']:
    stacked_barplot(data, col, "case_status")

# ── Correlation heatmap ───────────────────────────────────────────────────────
# %%
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=False, vmin=-1, vmax=1, cmap="Spectral")
plt.title("Correlation Matrix")
plt.show()

# ── Outlier detection — before/after ─────────────────────────────────────────
# %%
for col in ['no_of_employees', 'prevailing_wage', 'wage_fin']:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=data, x=col, ax=axes[0]).set_title(f"{col} — raw")
    sns.boxplot(data=data3, x=col, ax=axes[1]).set_title(f"{col} — outlier-treated")
    plt.tight_layout()
    plt.show()

# ── Feature importance — top models ─────────────────────────────────────────
# %%
for name, model in [("XGBoost Tuned", xgbo_tuned), ("RF Tuned", rfo_tuned), ("GBM Tuned", gbco_tuned)]:
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)[:15]
    feat_imp.plot(kind='barh', figsize=(10, 7), title=f'Top 15 Feature Importances — {name}', color='violet')
    plt.gca().invert_yaxis()
    plt.show()

# ── Confusion matrices — key models ─────────────────────────────────────────
# %%
for name, (m, X_tr, X_te, y_tr, y_te) in [
    ("Stacking", model_dict["Stacking"]),
    ("XGBoost Tuned", model_dict["XGBoost Tuned"]),
    ("AdaBoost", model_dict["AdaBoost"]),
]:
    print(f"\n{name} — test confusion matrix:")
    confusion_matrix_sklearn(m, X_te, y_te)
