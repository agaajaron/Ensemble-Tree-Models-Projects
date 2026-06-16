# %% [markdown]
# # Visualization — EDA and Model Results

# %%
from evaluate import *
from utils import histogram_boxplot, labeled_barplot, boxplot, confusion_matrix_sklearn

# ── EDA ──────────────────────────────────────────────────────────────────────
# %%
for col in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']:
    histogram_boxplot(data, col)

labeled_barplot(data, "Class", perc=True)
labeled_barplot(data, "Pregnancies", perc=True)

# %%
plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, cmap="Spectral")
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(data=data, hue="Class")
plt.show()

for col in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']:
    boxplot(data, col)

# ── Confusion matrices ────────────────────────────────────────────────────────
# %%
for name, m in zip(model_names, models):
    print(f"\n{name} — test confusion matrix:")
    confusion_matrix_sklearn(m, X_test, y_test)

# ── Feature importance — best model ─────────────────────────────────────────
# %%
importances = dtree_estimator.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12, 12))
plt.title('Feature Importances — Tuned Decision Tree')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
