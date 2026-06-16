# %% [markdown]
# # Data Preprocessing

# %%
from data_loader import *

# %%
# Remove duplicates
data.drop_duplicates(inplace=True)
print("After dedup:", data.shape)

# %%
# Encode target: 1 = Not_Canceled, 0 = Canceled
data["Target"] = np.where(data["booking_status"] == "Not_Canceled", 1, 0)
data.drop(columns=["booking_status"], inplace=True)

# %%
# One-hot encode categorical features
df = pd.get_dummies(
    data,
    columns=["type_of_meal_plan", "room_type_reserved", "market_segment_type"],
    drop_first=True,
)
df.columns = df.columns.str.lower().str.replace(" ", "_")

# %%
# Feature / target split
X = df.drop("target", axis=1)
y = df["target"]

# %%
# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
print("Train:", X_train.shape, "  Test:", X_test.shape)

# %%
# statsmodels requires an explicit intercept column
X_train1 = sm.add_constant(X_train)
X_test1 = sm.add_constant(X_test)
