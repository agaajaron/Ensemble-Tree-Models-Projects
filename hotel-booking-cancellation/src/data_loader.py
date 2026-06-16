# %% [markdown]
# # Data Loading - Hotel Booking Cancellation Dataset
#
# Source: StarHotelsGroup.csv
# 56,926 rows → 42,576 after removing duplicates
# Target: booking_status (Canceled / Not_Canceled)

# %%
from config import *

hotel0 = pd.read_csv("StarHotelsGroup.csv")
data = hotel0.copy()

print("Shape:", data.shape)
print(data.dtypes)
data.head()
