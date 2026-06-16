# %% [markdown]
# # Data Preprocessing

# %%
from data_loader import *
from utils import treat_outliers_all

# ── Feature engineering ────────────────────────────────────────────────────────
# %%
wage_map = {"Hour": 2080, "Week": 52, "Month": 12, "Year": 1}
data["wage_fin"] = data["prevailing_wage"] * data["unit_of_wage"].map(wage_map)

# ── Encode target ─────────────────────────────────────────────────────────────
# %%
data["case_status"] = (data["case_status"] == "Certified").astype(int)

# ── One-hot encode categoricals ───────────────────────────────────────────────
# %%
cat_cols = [
    "continent", "education_of_employee", "has_job_experience",
    "requires_job_training", "region_of_employment",
    "unit_of_wage", "full_time_position",
]
df = pd.get_dummies(data, columns=cat_cols, drop_first=True)
df.columns = df.columns.str.lower().str.replace(" ", "_")

# ── Feature / target split ────────────────────────────────────────────────────
# %%
X = df.drop("case_status", axis=1)
y = df["case_status"]

# ── Regular train-test split (70/30, stratified) ─────────────────────────────
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1, stratify=y
)
print("Train:", X_train.shape, "  Test:", X_test.shape)

# ── Outlier-treated dataset ───────────────────────────────────────────────────
# %%
numeric_cols = ["no_of_employees", "prevailing_wage", "wage_fin"]
data3 = data.copy()
data3 = treat_outliers_all(data3, numeric_cols)
df3 = pd.get_dummies(data3, columns=cat_cols, drop_first=True)
df3.columns = df3.columns.str.lower().str.replace(" ", "_")

X3 = df3.drop("case_status", axis=1)
y3 = df3["case_status"]

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X3, y3, test_size=0.30, random_state=1, stratify=y3
)
print("Outlier-treated — Train:", X_train3.shape)

# ── 50% subsample for faster GridSearchCV ────────────────────────────────────
# %%
X_train_sub, _, y_train_sub, _ = train_test_split(
    X_train, y_train, test_size=0.50, random_state=1, stratify=y_train
)
X_train3_sub, _, y_train3_sub, _ = train_test_split(
    X_train3, y_train3, test_size=0.50, random_state=1, stratify=y_train3
)
print("Subsample size:", X_train_sub.shape)
