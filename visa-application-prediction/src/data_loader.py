# %% [markdown]
# # Data Loading - EasyVisa US Visa Applications
#
# Source: EasyVisa.csv — OFLC (Office of Foreign Labor Certification)
# ~25,000 rows (after filtering), 12 columns
# Target: case_status (Certified / Denied)

# %%
from config import *

visa = pd.read_csv("EasyVisa.csv")
data = visa.copy()

# Drop surrogate key
data.drop(columns=["case_id"], inplace=True)

# Remove impossible employee counts
data = data[data["no_of_employees"] > 0].reset_index(drop=True)

print("Shape:", data.shape)
print(data.dtypes)
data.describe().T
