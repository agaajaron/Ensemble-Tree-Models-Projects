# %% [markdown]
# # Data Preprocessing

# %%
from data_loader import *

# %%
# Replace biologically impossible 0 values with column median
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    data.loc[data[col] == 0, col] = data[col].median()

# %%
# Split features and target
X = data.drop('Class', axis=1)
y = data['Class']

# %%
# Stratified 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
print("Train:", X_train.shape, "  Test:", X_test.shape)
print(y.value_counts(normalize=True))
