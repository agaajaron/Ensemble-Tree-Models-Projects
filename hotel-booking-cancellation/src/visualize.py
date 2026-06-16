# %% [markdown]
# # Visualization — EDA and Model Results

# %%
from evaluate import *
from utils import histogram_boxplot, labeled_barplot, confusion_matrix_statsmodels

# ── EDA — distributions ───────────────────────────────────────────────────────
# %%
for col in ['lead_time', 'no_of_special_requests', 'avg_price_per_room',
            'no_of_week_nights', 'no_of_weekend_nights',
            'no_of_adults', 'no_of_children', 'no_of_previous_cancellations']:
    histogram_boxplot(data, col)

# %%
for col in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type',
            'booking_status', 'repeated_guest']:
    labeled_barplot(data, col, perc=True)

# %%
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap="Spectral", fmt=".1f")
plt.title("Correlation Matrix")
plt.show()

# ── EDA — business insights ───────────────────────────────────────────────────
# %%
# Lead time vs cancellation rate
cancel_bins = pd.cut(data['lead_time'], bins=[0, 50, 100, 150, 200, 500])
cancel_rate = data.groupby(cancel_bins)['Target'].mean()
cancel_rate.plot(kind='bar', figsize=(8, 4), title='Cancellation Rate by Lead Time Bin')
plt.ylabel('Cancellation Rate')
plt.show()

# %%
# Special requests vs cancellation
sns.boxplot(data=data, x='Target', y='no_of_special_requests', palette='pastel')
plt.title("Special Requests vs Booking Status")
plt.xticks([0, 1], ['Canceled', 'Not Canceled'])
plt.show()

# ── ROC curves ────────────────────────────────────────────────────────────────
# %%
fpr_full, tpr_full, _ = roc_curve(y_test, logit_full.predict(X_test1))
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_depth7.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_full, tpr_full, label=f"Logit Full (AUC={roc_auc_score(y_test, logit_full.predict(X_test1)):.3f})")
plt.plot(fpr_dt, tpr_dt, label=f"DT Depth=7 (AUC={roc_auc_score(y_test, dt_depth7.predict_proba(X_test)[:,1]):.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves")
plt.legend()
plt.show()

# ── Feature importance — Decision Tree ───────────────────────────────────────
# %%
importances = dt_depth7.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)[:15]
feat_imp.plot(kind='barh', figsize=(10, 7), title='Top 15 Feature Importances — DT Depth=7', color='violet')
plt.gca().invert_yaxis()
plt.show()

# ── Cost complexity pruning curve ────────────────────────────────────────────
# %%
plt.figure(figsize=(10, 5))
plt.title("F1 Score vs CCP Alpha")
plt.xlabel("alpha")
plt.ylabel("F1")
plt.plot(np.linspace(0, max(train_f1s), len(train_f1s)), train_f1s, marker='o', label='train', color='blue')
plt.plot(np.linspace(0, max(train_f1s), len(test_f1s)), test_f1s, marker='o', label='test', color='orange')
plt.legend()
plt.show()
