# %% [markdown]
# # Data Loading - Pima Indians Diabetes Dataset
#
# Source: pima-diabetes.csv (768 rows, 9 columns)
# Target: Class (0 = not diabetic, 1 = diabetic)

# %%
from config import *

pima = pd.read_csv("pima-diabetes.csv")
data = pima.copy()

print(data.shape)
print(data.dtypes)
data.describe().T
