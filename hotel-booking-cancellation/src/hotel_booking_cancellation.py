# %% [markdown]
# #  Hotel Management Project
# 
# ## Context
# 
# A significant number of hotel bookings are called-off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. T
# 
# ## Objective
# The increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled. 
# 
# * analyze the data provided to find which factors have a high influence on booking cancellations
# * build a predictive model that can predict which booking is going to be canceled in advance
# * formulate strategy and profitable policies for cancellations and refunds.
# 
# ## Data Description
# The data contains the different attributes of customers' booking details. The detailed data dictionary is given below.
# 
# 
# **Data Dictionary**
# 
# * no_of_adults: Number of adults
# * no_of_children: Number of Children
# * no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
# * no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
# * type_of_meal_plan: Type of meal plan booked by the customer:
#     * Not Selected â€“ No meal plan selected
#     * Meal Plan 1 â€“ Breakfast
#     * Meal Plan 2 â€“ Half board (breakfast and one other meal)
#     * Meal Plan 3 â€“ Full board (breakfast, lunch, and dinner)
# * required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
# * room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by Star Hotels.
# * lead_time: Number of days between the date of booking and the arrival date
# * arrival_year: Year of arrival date
# * arrival_month: Month of arrival date
# * arrival_date: Date of the month
# * market_segment_type: Market segment designation.
# * repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
# * no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
# * no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer prior to the current booking
# * avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
# * no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
# * booking_status: Flag indicating if the booking was canceled or not.

# %% [markdown]
# ## Importing necessary libraries and data

# %%
# this will help in making the Python code more structured automatically (good coding practice)
# %load_ext nb_black

# Library to suppress warnings or deprecation notes
import warnings

warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# Libraries to build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV

# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    plot_confusion_matrix,
    make_scorer,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)


# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics

# %%
hotel = pd.read_csv("StarHotelsGroup.csv")

# %%
hotel.head()

# %%
hotel.describe()

# %%
hotel.shape

# %% [markdown]
# Data is in form of matrix/table with 56926 rows and 18 columns

# %%
hotel.info()

# %% [markdown]
# Observation: Most of the data-types are either int64 or float64.
# 4 columns - are having data-types as an object, this means we need to convert these into suitable data-type before we feed our data into the model.

# %% [markdown]
# ## Data Overview
# 
# - Observations
# - Sanity checks

# %%
hotel[hotel.duplicated()].count()

# %%
hotel0 = hotel.drop_duplicates()

# %%
hotel0

# %%
56926 - 42576  # checking row counts in dataframes - to make sure I did not delete too much

# %% [markdown]
# There are 14350 duplicate values in the dataset so ok.

# %% [markdown]
# Now I check for unique values in all columns:

# %%
for col in hotel0.columns:
    print("Number of unique values in ", col, len(hotel0[col].unique()))

# %% [markdown]
# This printout shows unique values for each column. It will help visualization and analysis.

# %%
nulldata = hotel0.isnull().any(axis=1)

# %%
for n in nulldata.value_counts().sort_index().index:
    if n > 0:
        print(f"For the rows with exactly {n} missing values, NAs are found in:")
        n_miss_per_col = hotel[nulldata == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print("\n\n")

# %% [markdown]
# ## Conclusion: No missing data.

# %%
data = hotel0

# %%
data

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# 
# - EDA is an important part of any project involving data.
# - It is important to investigate and understand the data better before building a model with it.
# - A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
# - A thorough analysis of the data, in addition to the questions mentioned below, should be done.

# %% [markdown]
# **Questions**:
# 1. What are the busiest months in the hotel?
# 2. Which market segment do most of the guests come from?
# 3. Hotel rates are dynamic and change according to demand and customer demographics. What are the differences in room prices in different market segments?
# 4. What percentage of bookings are canceled? 
# 5. Repeating guests are the guests who stay in the hotel often and are important to brand equity. What percentage of repeating guests cancel?
# 6. Many guests have special requirements when booking a hotel room. Do these requirements affect booking cancellation?

# %%
cancel=data[data['booking_status']=='Canceled']


# %%
cancel

# %%
noncancel = data[data["booking_status"] == "Not_Canceled"]

# %%
noncancel

# %%
# customized boxplot+histogram with mean and median values
def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="yellow"
    )  
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    ) 
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  


# %%
sns.histplot(data=hotel, x="no_of_adults", bins=5, palette="winter")

# %%
# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

# %%
histogram_boxplot(hotel, "arrival_month")

# %%
histogram_boxplot(data, "arrival_month")

# %% [markdown]
# ## COMMENT: 
# The 2 above figures show significance of duplicated rows. even on such figure we can see thedifference eg. for counts for June and July. 
# It can affect answering the questions and building the model. From now on I only use the dataframe called data - obtained after removing duplicates. 

# %%
labeled_barplot(data, "arrival_month", perc=True)

# %% [markdown]
# 
# ### Q1. What are the busiest months in the hotel?

# %% [markdown]
# ### Answer Q2: 
# Based on histogram: The busiest month is August (12.5%) then July (11.1%) and May (10.2%). 

# %% [markdown]
# ### Q2. Which market segment do most of the guests come from?

# %%
sns.histplot(data=data, x="market_segment_type", bins=5, palette="winter")

# %% [markdown]
# ### Answer Q2: 
# Based on histogram : Most guests come from Online market segment.

# %% [markdown]
# ### Q3. Hotel rates are dynamic and change according to demand and customer demographics. What are the differences in room prices in different market segments?

# %%
sns.displot(
    data,
    x="avg_price_per_room",
    hue="market_segment_type",
    bins=30,
    palette="winter",
    kde=True,
    stat="probability",
)

# %%
sns.violinplot(
    x="market_segment_type", y="avg_price_per_room", data=data, height=5, aspect=5.5
)

# %%
sns.boxplot(x="market_segment_type", y="avg_price_per_room", data=data)

# %%
data.groupby("market_segment_type")["avg_price_per_room"].mean()

# %%
data.groupby("market_segment_type")["avg_price_per_room"].mean().plot(kind="bar")

# %%
data.groupby("market_segment_type")["avg_price_per_room"].median()

# %%
3

# %%
data.groupby("market_segment_type")["avg_price_per_room"].median().plot(kind="bar")

# %% [markdown]
# ## Answer Q3: 
# 
# ### 1. Observations:
# 
# 1. As can be seen on the plots the "complementary" category is different. Median price is 0  (which agrees with the name of the category). The mean value is not representative in this case.
# 
# 2. Aviation has the mean 103.2 and median 95.0 (there is smaller numbr of outliers but on the whole this category is small.)
# 
# 3. Corporate mean price 82.5 and median 75.0 
# 
# 4. Offline mean price 87.7 and median 83.8
# 
# 5. Online mean price 119.9 and median 115.0 
# 
# All market segments except "complimentary" have many outliers for average price distribution. 
# All distriutions  except for "complimentary" look similar (barplot, violiplot).
# Histogram shows slighlty skewed distrbution but overall the shape resembles normal distribution (for all except "complimentary"). 
#     
# ### 2. Discussion of Mean and Median average price values for market segmenst     
# Suprisingly 'online' market segment has the highest mena and medina price. One could expect the customers to be able to find 
#     the cheaper price and promotions but the data does not show it.  
#     
#  Offline and Corporate have the lowest prices. 
#     
# Interestingly the difference for Online customer to go Offline means saving 30%. It would be interesting to know more about the offline 
#     group. It is second largest group in dataset and there is no indication why they price is so advantageous.
#     (Are these group bookings? )
#      
# One can try to understand the corporate market segment price to be lower - as often the lower 
# price is negotiated for a series of bookings (eg. rereats, workshops, conferences etc.)
#     
# ### 3. Below are plots that illustrate the size of each market segment in the full dataset     

# %%
labeled_barplot(data, "market_segment_type", perc=False, n=None)

# %%
labeled_barplot(data, "market_segment_type", perc=True, n=None)

# %% [markdown]
# ### Q4. What percentage of bookings are canceled? 

# %% [markdown]
# There are 42576 number of total bookings. 14487 canceled bookings.

# %%
cancel.info()

# %%
14487 / 42567

# %% [markdown]
# ###  Answer Q4: Percentage of cancelled bookings is equal to 34%

# %% [markdown]
# ### Q5: Repeating guests are the guests who stay in the hotel often and are important to brand equity. What percentage of repeating guests cancel?

# %%
labeled_barplot(data, "repeated_guest", perc=True)

# %% [markdown]
# Repeated guests are 3% of total guests.

# %%
repeat = data[data["repeated_guest"] == 1]

# %% [markdown]
# I create dataframe with data ony for repeated guests

# %%
repeat

# %%
labeled_barplot(repeat, "booking_status", perc=True, n=None)

# %% [markdown]
# ### Answer Q5: This plot shows that 0.8% of repeated guests cancel.

# %% [markdown]
# ###  Q6. Many guests have special requirements when booking a hotel room. Do these requirements affect booking cancellation?

# %%
data.groupby("booking_status")["no_of_special_requests"].median()


# %%
labeled_barplot(noncancel, "no_of_special_requests", perc=True)

# %% [markdown]
# Conclusion: 62.7% of non-canceled bookings had 1 or more special requestes.

# %%
labeled_barplot(cancel, "no_of_special_requests", perc=True)

# %% [markdown]
# Conclusion: Smaller range - only from 0 to 2. And 60.4% did not include special requests in their bookings. 

# %% [markdown]
# ### Answer Q6: 
# 
# Cancelled bookings median is 0 for number of special requests. 
# For non-canceled bookins the median is 1. 
# So guests who did not cancel the boikings tookcare to optimize their stay and included special requests in their bookings. 
# While the guests who canceled did not included special requests.
# It might be a signature that they did were not sure they will use the booking. They did not finilize their plans or did not 
# treat the booking "seriously" (for some reasons).
# 
# The conclusionis if a customer has specialrequests in the booking it is less probable he/she will cancel the booking.
# I will now show  some plots for this question.
# 
# More than 60% of noncalcelled  bokings had 1 or more special requests. 
# More than 60% of cancelled booings had 0 speial requests.
# 
# The range: The number of special requests in non-canceled bookings ranges from 0 to 5 when for canceled bookings only from 0 to 2. 

# %% [markdown]
# ## Additional analysis EDA

# %% [markdown]
# I start with pairplot as it often shows interesting relationships between variables/data.

# %%
sns.pairplot(data, hue="booking_status")

# %% [markdown]
# ## Main conclusions
#     
# Cancelled group has 
# * longer lead time (100+ days)
# * higher price
# * is comprised of new guests ("returning guests" did not show cancellations in pairplot)
# * smaller number of special requests (when planning is done carefully and there is strong intent - it is logical the hotel guest tries to plan and optimize the stay by carefully checking the posibilities and using special requests. BTW Individualizing the stay by publicising the options of 'special requests' can be one of the ways how to motivate guest against cancelling - and chosing different hotel.)
# * "0" in "previous_booking_not_cancelled"
# * arrival_time_distribution is narrower and centers around middle of the e year- Summer months.
#     
#     
# Different arrival_year data shows slightly different trend (2018,2019 - visualization looks different than for 2017).  
# 
# I will now analyse different variables and check in detail these general trends shown in pairplot.

# %% [markdown]
# ### Analysis of effect of the variable room_type_reserved:

# %%
labeled_barplot(data, "room_type_reserved", perc=True)

# %%
data.groupby("room_type_reserved")["avg_price_per_room"].mean()

# %%
69.8 + 22

# %% [markdown]
# ### Conclusion: 
# 91.8% of bookings were for Room type1 and Room type4. these room types are not the cheapest or most expensive.  

# %%
labeled_barplot(cancel, "room_type_reserved", perc=True)

# %%
labeled_barplot(noncancel, "room_type_reserved", perc=True)

# %%
cancel.groupby("room_type_reserved")["avg_price_per_room"].mean()

# %%
noncancel.groupby("room_type_reserved")["avg_price_per_room"].mean()

# %% [markdown]
# ### Conclusion: 
# The cancelled bookings had higher average price for every room type.

# %% [markdown]
# ### Analysis of the type_of_meal_plan variable

# %%
labeled_barplot(data, "type_of_meal_plan", perc=True)

# %%
labeled_barplot(cancel, "type_of_meal_plan", perc=True)

# %%
labeled_barplot(noncancel, "type_of_meal_plan", perc=True)

# %% [markdown]
# ### Conclusion: 
# Meal plan seems ot be not very important variable for cancellations.

# %% [markdown]
# ### Analysis of variables arrival_month and lead_time 

# %%
sns.displot(cancel, x="lead_time", kde=True, hue="arrival_month")

# %%
sns.displot(cancel, x="lead_time", kde=True)

# %%
sns.displot(noncancel, x="lead_time", kde=True, hue="arrival_month")

# %%
sns.displot(noncancel, x="lead_time", kde=True)

# %%
data.groupby("booking_status")["lead_time"].mean()

# %%
data.groupby("booking_status")["lead_time"].median()

# %% [markdown]
# Conclusion: Very different distributions for canceled and non-canceled bookings. 
# Canceled bookings have been made with lead time longer by 2 months. 
# Canceled bookings were made on average 4 months before palnned hotel stay.
# Non canceled bookings were made on average 1-2 months ahead of the stay.

# %%
cancel.groupby("arrival_month")["lead_time"].mean()

# %%
cancel.groupby("arrival_month")["lead_time"].median()

# %%
noncancel.groupby("arrival_month")["lead_time"].mean()

# %%
noncancel.groupby("arrival_month")["lead_time"].median()

# %% [markdown]
# I now will display thedifference of average lead time for each month.

# %%
cancel.groupby("arrival_month")["lead_time"].mean() - noncancel.groupby(
    "arrival_month"
)["lead_time"].mean()

# %% [markdown]
# Conclusion: The customers who cancelled bookings have much longer lead time. Maybe one can use this fact -  that they have time to cancel ahead of time  so that the hotel does not have many losses. 
#     

# %%
labeled_barplot(cancel, "arrival_month", perc=True)

# %%
labeled_barplot(noncancel, "arrival_month", perc=True)

# %% [markdown]
# Conclusion: One can see that the most of the cancellations are from April to August.
#     
# It might be relevant information and one can add additional screening for those bookings especially in April and May (and additional email/text 
# reminders and confirmations to offer guests earlier cancellations that would not lead to revenue losses). One can offer a gradual return for cancellation 
# is early enough to reuse.
# 
# The cancelations are corresponding to longer "lead_time" (100+ days). So in principlethe customers have time to cancel ahead and 
# do not cause lossess to the hotels.  

# %%
labeled_barplot(cancel, "no_of_previous_cancellations", perc=True)

# %%
labeled_barplot(noncancel, "no_of_previous_cancellations", perc=True)

# %%
largelead = data[data["lead_time"] > 100]

# %%
labeled_barplot(largelead, "booking_status", perc=True)

# %%
vlargelead = data[data["lead_time"] > 150]

# %%
vlargelead.info()

# %%
labeled_barplot(vlargelead, "booking_status", perc=True)

# %% [markdown]
# Conclusion: Three Fourth of the bookings with lead time longer than 150 days were cancelled. 

# %%
g = sns.jointplot(
    data=data, x="lead_time", y="avg_price_per_room", hue="booking_status"
)

# %%
g = sns.jointplot(
    data=cancel, x="lead_time", y="avg_price_per_room", hue="market_segment_type"
)

# %%
g = sns.jointplot(
    data=cancel, x="lead_time", y="avg_price_per_room", hue="required_car_parking_space"
)

# %% [markdown]
# ### Conclusion
# * guests who require car parking space among the canceled bookings - have lead time longer than 100 days. 
# * only very few guests who cancelled booking - requested parking

# %%
g = sns.jointplot(
    data=noncancel,
    x="lead_time",
    y="avg_price_per_room",
    hue="required_car_parking_space",
)

# %% [markdown]
# ### Conclusion
# Guests who required car parking space among the non-canceled bookings have lead time shorter than 150 days.

# %%
g = sns.jointplot(
    data=cancel, x="lead_time", y="avg_price_per_room", hue="repeated_guest"
)

# %%
g = sns.jointplot(data=data, x="lead_time", y="avg_price_per_room", hue="arrival_year")

# %%
sns.boxplot(x="arrival_year", y="avg_price_per_room", data=data)

# %% [markdown]
# ### Conclusion
# One can see effect of inflation - the later year the higher the mean price.

# %%
labeled_barplot(vlargelead, "market_segment_type", perc=True)

# %% [markdown]
# ### EDA Conclusions: Please note that for each relevant or interesting result I included conclusion and I do not repeat it here.
#  (it is close to plot that is related to the conclusion).    

# %% [markdown]
# ## Data Preprocessing
# 
# - Missing value treatment (if needed)
# - Feature engineering (if needed)
# - Outlier detection and treatment (if needed)
# - Preparing data for modeling 
# - Any other preprocessing steps (if needed)

# %% [markdown]
# ### Data preprocessing  - here is the summary:

# %% [markdown]
# 1.Missing values: There are no missing values. So no imputing needed.

# %% [markdown]
# 2.Outliers: Decision trees are not sensitive to outliers. So we do not have to worry about it forthe model of the Decision Tree. I will reconsider if the model does not work, but I hesitate to remove additional data after removing many duplicate rows. Logistic Regression models are not much impacted due to the presence of outliers because the sigmoid function tapers the outliers. But the presence of extreme outliers may somehow affect the performance of the model and lowering the performance.
# The rest of the notebook will follow this - so section for logistic regression will also include a model with otliers removed. 

# %% [markdown]
# 3.Object type variables need encoding.

# %% [markdown]
# 4.No scaling needed as decision trees are not sensitive. (the only variable that one can cosider scaling is the price)

# %% [markdown]
# 5.Duplicates removed.  

# %% [markdown]
# Further steps: 
#     1) Removing fetaures due to multicolinearity - for logistic regression (decision tree can handle multicollinearity). 
#     2) After EDA I would like to consider model build only for Online market segment. 

# %%
data.head()

# %% [markdown]
# Encoding the object type variables:

# %%
cols1 = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]

# %%
df = pd.get_dummies(data=data, columns=cols1, drop_first=True)

# %%
df

# %% [markdown]
# ## Checking Multicollinearity
# 

# %% [markdown]
# First - manually encoding the "booking_status"

# %%
df["Target"] = np.where(df["booking_status"].str.contains("Not_Canceled"), 1, 0)

# %% [markdown]
# 
# Checking the dataframe
# 

# %%
df

# %%
df.drop("booking_status", axis=1, inplace=True)

# %% [markdown]
# Preparation of  training and testing data sets

# %%
X = df.drop(["Target"], axis=1)
y = df["Target"]

# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# %%
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))

# %% [markdown]
# ### Comment:
# The classes are imbalanced. 

# %% [markdown]
# ### Adding constant to statsmodel Logit model

# %%
from statsmodels.tools.tools import add_constant

# %%
# we have to add the constant manually
x_train1 = sm.add_constant(x_train)
# adding constant to the test data
x_test1 = sm.add_constant(x_test)

logittest = sm.Logit(y_train, x_train1).fit()
print(logittest.summary())

# %% [markdown]
# Conclusion: Model preditcs that non_calcelation is positively correlated with 
# * no-of_adults, 
# * required_parking_space
# * repeated_guest
# * no_of_special_requests, 
# * Complementary market segment 
# * Offline market segment
# 
# It negatively correlated with 
# * lead_time
# * no_of_children 
# * no_of_previous_cancelations 
# * average price per room.
# 
# 
# 
# The prediciton of model agrees with EDA. 
# 
# As reminder based on ETA the following features were correlated in the visualizations:
# *  lead time 
# *  price
# * "repeated guest" did not show cancelations in pairplot
# * number of special requests 
# * "previous_booking_not_cancelled"
# * arrival_time
# 
# The Logistic Regression allows to compare the importance of the features. 
# 

# %%
print(logittest.params)

# %% [markdown]
# Check for regularized fit

# %%
logittest2 = sm.Logit(y_train, x_train1).fit_regularized()
print(logittest2.summary())

# %% [markdown]
# Comment: LL-p value is 0. It means model is not bad.

# %% [markdown]
# ### Test for VIF (multi colinearity)

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# we will define a function to check VIF
def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i)
        for i in range(len(predictors.columns))
    ]
    return vif

# %%
checking_vif(x_train1)

# %% [markdown]
# Conclusion: "Market segment" variable (3 out of 4 encodes) seem to have large VIF. 
#     I will consider 2 versions with and without this variable. 

# %%
vifdropcols=["market_segment_type_Complementary","market_segment_type_Offline","market_segment_type_Online","market_segment_type_Corporate"]

# %%
# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification_statsmodels(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf

# %% [markdown]
# ## Building a Logistic Regression model

# %% [markdown]
# ### Building logistic regression using statsmodels implementation

# %% [markdown]
# I have started the Logistic Regression model from  statsmodels above. Here I will remove features with high VIF and high p-values, check performance scores. Then I will build sklearn Logistic Regression model for comparison. 

# %%
print("Training performance:")
model_performance_classification_statsmodels(logittest, x_train1, y_train)

# %%
print("Test set performance")
model_performance_classification_statsmodels(logittest, x_test1, y_test)

# %% [markdown]
# Conclusion: The scores are very good. 

# %%
# defining a function to plot the confusion_matrix 

def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

# %% [markdown]
# How to reduce the mistakes in classificaiton of canceled bookings - need to reduce False Negatives.
# Also f1_score should be maximized, the greater the f1_score higher the chances of identifying both the classes correctly.
# Let us check for False Negatives in confusion matrix.

# %%
confusion_matrix_statsmodels(logittest, x_train1, y_train)

# %%
confusion_matrix_statsmodels(logittest, x_test1, y_test),

# %% [markdown]
# Test and train sets differ by around 0.08% in False Negatives. False positives agree very well. 
# Without doing anything thisis already pretty good model.

# %% [markdown]
# ### Remove multicolinearity (high VIF columns plus one with lower VIF - after forum advice)

# %%
x_train2 = x_train1.drop(
    [
        "market_segment_type_Complementary",
        "market_segment_type_Offline",
        "market_segment_type_Online",
        "market_segment_type_Corporate",
    ],
    axis=1,
)

logit2 = sm.Logit(y_train, x_train2)
lg2 = logit2.fit()

print(lg2.summary())

# %%
checking_vif(x_train2)

# %% [markdown]
# ### p-values removing loop

# %% [markdown]
# Now I do the following:
# 
# 1) Build a model, check the p-values of the variables, and drop the column with the highest p-value.
# 2) Create a new model without the dropped feature, check the p-values of the variables, and drop the column with the highest p-value.
# 3) Repeat the above two steps till there are no columns with p-value > 0.05.
# 

# %% [markdown]
# ### Remove "no_of_previous_bookings_not_canceled"

# %%
x_train3 = x_train2.drop(["no_of_previous_bookings_not_canceled"], axis=1,)

logit3 = sm.Logit(y_train, x_train3)
lg3 = logit3.fit()

print(lg3.summary())

# %% [markdown]
# ### Remove "type_of_meal_plan_Meal Plan 3"

# %%
x_train4 = x_train3.drop(["type_of_meal_plan_Meal Plan 3"], axis=1,)

logit4 = sm.Logit(y_train, x_train4)
lg4 = logit4.fit()

print(lg4.summary())

# %% [markdown]
# ### Removing "Room_type 2 & 3 & 4" 

# %%
x_train5 = x_train4.drop(
    [
        "room_type_reserved_Room_Type 3",
        "room_type_reserved_Room_Type 4",
        "room_type_reserved_Room_Type 2",
    ],
    axis=1,
)

logit5 = sm.Logit(y_train, x_train5)
lg5 = logit5.fit()

print(lg5.summary())

# %% [markdown]
# ### Convert coefficients to odds
# 
# The coefficients of the logistic regression model are in terms of log(odd), to find the odds we have to take the exponential of the coefficients.
# 
# Therefore, odds = exp(b)
# 
# The percentage change in odds is given as odds = (exp(b) - 1) * 100

# %%
# converting coefficients to odds
odds = np.exp(lg5.params)

# finding the percentage change
perc_change_odds = (np.exp(lg5.params) - 1) * 100

# removing limit from number of columns to display
pd.set_option("display.max_columns", None)

# adding the odds to a dataframe
pd.DataFrame({"Odds": odds, "Change_odd%": perc_change_odds}, index=x_train5.columns).T

# %%
perc_change_odds

# %% [markdown]
# Due to the nature of exponential function the features that have positive correation have large odds when the negative are simply small and seemto have small effect. 
# But for canceled bokings these features are important.

# %% [markdown]
# ### Logistic Regression statsmodels conclusions:
# 1. The model confirms conclusions from EDA: important fetaures "repeated_guest","no_of_special_requests"        
# 2. "required_car_parking_space" seems very important and was not so important in ETA for non_cancelation.
# 3. "types of room 6 and 7" means higher price per room for non_canceled bookings (so when guest paid more - he/she did not want to risk cancellation?) but one can see in general also inverse relationship with average prce per room (the cheaper the room the more probable non-caleation).
# 4. Noncaceled bookings aremade by guests who plan in detail their stay - a) select meal plan b) have special requests c) require car space
# 

# %% [markdown]
# ## Model performance evaluation

# %% [markdown]
# ### Statsmodel final Logistic Regression model scores:

# %%
print("Training performance:")
model_performance_classification_statsmodels(lg5, x_train5, y_train)

# %%
confusion_matrix_statsmodels(lg5, x_train5, y_train)

# %% [markdown]
# Unfortunately this model is worse than default before removing "p-value" columns and higher VIF columns.
# Accoding to internet both p-values and VIF consideration for Logistic regression is much more subtle (I found opinion that VIF only remove when it is very large and do not remove higher p-values as long as LL values is small). 

# %%
logit_roc_auc_train = roc_auc_score(y_train, lg5.predict(x_train5))
fpr, tpr, thresholds = roc_curve(y_train, lg5.predict(x_train5))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic for final model")
plt.legend(loc="lower right")
plt.show()

# %%
logit_roc_auc_train = roc_auc_score(y_train, logittest.predict(x_train1))
fpr, tpr, thresholds = roc_curve(y_train, logittest.predict(x_train1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic for initial model")
plt.legend(loc="lower right")
plt.show()

# %%
y_scores = lg5.predict(x_train5)
prec, rec, tre = precision_recall_curve(y_train, y_scores,)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("Recall vs precision curve for final model")
    plt.ylim([0, 1])


plt.figure(figsize=(10, 7))
plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()

# %%
y_scores = logittest.predict(x_train1)
prec, rec, tre = precision_recall_curve(y_train, y_scores,)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("Recall vs precision curve for initial model")
    plt.ylim([0, 1])


plt.figure(figsize=(10, 7))
plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()

# %% [markdown]
# Both models predict similar thresholds.

# %%
# Optimal threshold as per AUC-ROC curve
# The optimal cut off would be where tpr is high and fpr is low

fpr, tpr, thresholds = roc_curve(y_train, lg5.predict(x_train5))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auc_roc1 = thresholds[optimal_idx]
print(optimal_threshold_auc_roc1)

# %%
fpr, tpr, thresholds = roc_curve(y_train, logittest.predict(x_train1))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auc_roc = thresholds[optimal_idx]
print(optimal_threshold_auc_roc)

# %% [markdown]
# Checking the scores and confusion matrix for the optimalthreshold for the 2 models:

# %%
# creating confusion matrix
confusion_matrix_statsmodels(
    lg5, x_train5, y_train, threshold=optimal_threshold_auc_roc1
)

# %%
# creating confusion matrix
confusion_matrix_statsmodels(
    logittest, x_train1, y_train, threshold=optimal_threshold_auc_roc
)

# %%
# checking model performance for this model
log_reg_model_train_perf_threshold_auc_roc5 = model_performance_classification_statsmodels(
    lg5, x_train5, y_train, threshold=optimal_threshold_auc_roc1
)
print("Training performance:")
log_reg_model_train_perf_threshold_auc_roc5

# %%
# checking model performance for this model
log_reg_model_train_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    logittest, x_train1, y_train, threshold=optimal_threshold_auc_roc
)
print("Training performance:")
log_reg_model_train_perf_threshold_auc_roc

# %%
This is the best model. It has high recall precision and F1 and smallest number of false positives (it best predicts the non_canceled bookings)

# %%
The original model (with 0.5 threshold) had better false negatives percentage. 

# %% [markdown]
# 
# 
# ### Sklearn Logistic regression (initial data set)
# 
# 
# 

# %%
from sklearn import metrics

from sklearn.linear_model import LogisticRegression

# Fit the model on train
modelSLR1 = LogisticRegression(random_state=1)
modelSLR1.fit(x_train, y_train)
# predict on test
y_predict = modelSLR1.predict(x_test)


coef_df = pd.DataFrame(modelSLR1.coef_)
coef_df["intercept"] = modelSLR1.intercept_
print(coef_df)

# %%
model_score1 = modelSLR1.score(x_test, y_test)
print(model_score1)

# %%
cm = metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])

df_cm = pd.DataFrame(
    cm,
    index=[i for i in ["Actual 1", " Actual 0"]],
    columns=[i for i in ["Predict 1", "Predict 0"]],
)
plt.figure(figsize=(7, 5))
sns.heatmap(df_cm, annot=True, fmt="g")
plt.show()

# %% [markdown]
# Values are similar to statsmodel Logistic Regression model. 

# %% [markdown]
# At this point it is rather matter of preferences which sklearn or statsmodel. 

# %% [markdown]
# ### Sklearn Logistic Regression with the data set with removed multicolinearity and high p-value columns

# %%
x_train6 = x_train5.drop(["const"], axis=1)

# %% [markdown]
# ### Dropping the columns from the test set that were dropped from the training set

# %%
x_test6 = x_test[x_train6.columns].astype(float)

# %%
x_test6

# %%
# Fit the model on train
modelSLR2 = LogisticRegression(random_state=1)
modelSLR2.fit(x_train6, y_train)
# predict on test
y_predict2 = modelSLR2.predict(x_test6)


coef_df2 = pd.DataFrame(modelSLR2.coef_)
coef_df2["intercept"] = modelSLR2.intercept_
print(coef_df2)

# %%
model_score2 = modelSLR2.score(x_test6, y_test)
print(model_score2)

# %%
cm2 = metrics.confusion_matrix(y_test, y_predict2, labels=[1, 0])

df_cm2 = pd.DataFrame(
    cm2,
    index=[i for i in ["Actual 1", " Actual 0"]],
    columns=[i for i in ["Predict 1", "Predict 0"]],
)
plt.figure(figsize=(7, 5))
sns.heatmap(df_cm2, annot=True, fmt="g")
plt.show()

# %% [markdown]
# ### Conclusion
# The score is lower than the full data set. Coefficients are similar.

# %% [markdown]
# ## Final Model Summary

# %% [markdown]
# * Original model performed very well. 
# * After removing high VIF columns and high p-value columns the model worked worse. 
# * Removing the variables might have cause removing too much information (it is nonlinear model so even variable that seems not very significant might support finding the right model).

# %% [markdown]
# ### Best performing model parameters: 
#     

# %% [markdown]
# Best performing model is the model Sklearn Logistic Regressionon for full data set. Accuracy 0.787 and the FN and FP are the smallest.

# %% [markdown]
# ### Conclusions based on the logistic Regression:
#     

# %% [markdown]
# 
# * we get lightly different models but they perform similarly and all perform relatively well. 
# * the following features are imporant: no_of_adults, no_of_children, no_of_weekend_nights,no_of_week_nights,required_car_parking_space,lead_time, arrival_year, arrival_month, arrival_date, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,avg_price_per_room, no_of_special_requests.
# * non_calcelled bookings are carefully prepared - guests have "parkingrequests" and "special requests"
# * repeated guest has higher probabaility of non_calling the booking
# * bookings with most expenisve rooms are also less likely to be canceled
# 
#     
# 
#  

# %% [markdown]
# ## Building a Decision Tree model

# %% [markdown]
# ### Model 0 (Balanced classes, defaults)

# %%
dTree = DecisionTreeClassifier(criterion="gini", random_state=1)
dTree.fit(x_train, y_train)

# %%
print("Accuracy on training set : ", dTree.score(x_train, y_train))
print("Accuracy on test set : ", dTree.score(x_test, y_test))

# %%
## Function to create confusion matrix
def make_confusion_matrix(model, y_actual, labels=[1, 0]):
    """
    model : classifier to predict values of X
    y_actual : ground truth  
    
    """
    y_predict = model.predict(x_test)
    cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(
        cm,
        index=[i for i in ["Actual - No", "Actual - Yes"]],
        columns=[i for i in ["Predicted - No", "Predicted - Yes"]],
    )
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

# %%
##  Function to calculate recall score
def get_recall_score(model):
    """
    model : classifier to predict values of X

    """
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    print("Recall on training set : ", metrics.recall_score(y_train, pred_train))
    print("Recall on test set : ", metrics.recall_score(y_test, pred_test))

# %%
make_confusion_matrix(dTree, y_test)

# %%
# Recall on train and test
get_recall_score(dTree)

# %%
feature_names = list(X.columns)
print(feature_names)

# %%
# Checking number of positives
y.sum(axis=0)

# %%
plt.figure(figsize=(20, 30))
tree.plot_tree(
    dTree,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
# Text report showing the rules of a decision tree -

print(tree.export_text(dTree, feature_names=feature_names, show_weights=True))

# %%
print(
    pd.DataFrame(
        dTree.feature_importances_, columns=["Imp"], index=x_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
importances = dTree.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %% [markdown]
# ### Conclusion: 
# The decision tree confirms the EDA conlclusions that lead time, price, number of special requests and arrival month mater for the cancelations. 

# %%
##  Function to calculate recall score
def get_f1_score(model):
    """
    model : classifier to predict values of X

    """
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    print("F1 on training set : ", metrics.f1_score(y_train, pred_train))
    print("F1 on test set : ", metrics.f1_score(y_test, pred_test))

# %%
get_f1_score(dTree)

# %% [markdown]
# ### Conclusion
# There is overfitting. It will need pruning.

# %%
make_confusion_matrix(dTree, y_test)

# %% [markdown]
# ## Pruning the tree

# %% [markdown]
# ### Prepruning by limiting the max_depth.

# %% [markdown]
# ### Model 1 (max_depth = 7)

# %%
dTree1 = DecisionTreeClassifier(criterion="gini", max_depth=7, random_state=1)
dTree1.fit(x_train, y_train)

# %%
make_confusion_matrix(dTree1, y_test)

# %%
# Accuracy on train and test
print("Accuracy on training set : ", dTree1.score(x_train, y_train))
print("Accuracy on test set : ", dTree1.score(x_test, y_test))
# Recall on train and test
get_recall_score(dTree1)

# %%
get_f1_score(dTree1)

# %% [markdown]
# ### Conclusions: 
# This tree scores look very good. There is no overfitting but accurcy is ery good and the recall is very high.

# %%
plt.figure(figsize=(15, 10))

tree.plot_tree(
    dTree1,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
print(tree.export_text(dTree1, feature_names=feature_names, show_weights=True))

# %%
importances = dTree1.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %% [markdown]
# ### Conclusion: 
# * EDA visualization allowed for the recognising the importance of lead time around 150 days. It is also the root node for tree. 
# 
# * The rest of the nodes also seem to coverge with EDA conclusions. 
# * It is also evident the "Online" is an important feature. Since the bookings made online, with large lead time can be canceled ahead of time in starightforward way - I would like to later build model only for online market segment.  

# %% [markdown]
# ### Model 2 Smaller tree (max_depth=6)

# %%
dTree2 = DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=1)
dTree2.fit(x_train, y_train)

# %%
# Accuracy on train and test
print("Accuracy on training set : ", dTree2.score(x_train, y_train))
print("Accuracy on test set : ", dTree2.score(x_test, y_test))
# Recall on train and test
get_recall_score(dTree2)

# %%
get_f1_score(dTree2)

# %%
plt.figure(figsize=(20, 30))
tree.plot_tree(
    dTree2,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
importances = dTree2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(tree.export_text(dTree2, feature_names=feature_names, show_weights=True))

# %% [markdown]
# ### Model 3: Tree with class_weight =balanced 

# %%
dTreeI = DecisionTreeClassifier(
    class_weight="balanced", criterion="gini", max_depth=7, random_state=1
)
dTreeI.fit(x_train, y_train)

# %%
# Accuracy on train and test
print("Accuracy on training set : ", dTreeI.score(x_train, y_train))
print("Accuracy on test set : ", dTreeI.score(x_test, y_test))
# Recall on train and test
get_recall_score(dTreeI)

# %% [markdown]
# Comment: Accuracy is lower but Recall is higher as compared to "balanced" tree. 

# %%
get_f1_score(dTreeI)

# %%
plt.figure(figsize=(20, 30))
tree.plot_tree(
    dTreeI,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
importances = dTreeI.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(tree.export_text(dTreeI, feature_names=feature_names, show_weights=True))

# %% [markdown]
# ## Model 4 - with  hyperparameter tuning - gridsearch CV (for default weight of classes)

# %%
# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_depth": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "min_samples_split": [40, 50, 60, 70],
    "criterion": ["entropy", "gini"],
    "splitter": ["best", "random"],
    "min_impurity_decrease": [0.00001, 0.0001, 0.01],
}


# Type of scoring used to compare parameter combinations
scorer = make_scorer(recall_score)

# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(x_train, y_train)

# %%
# Accuracy on train and test
print("Accuracy on training set : ", estimator.score(x_train, y_train))
print("Accuracy on test set : ", estimator.score(x_test, y_test))
# Recall on train and test
get_recall_score(estimator)

# %%
make_confusion_matrix(estimator, y_test)

# %%
get_f1_score(estimator)

# %%
plt.figure(figsize=(15, 10))

tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
# importance of features in the tree building ( The importance of a feature is computed as the
# (normalized) total reduction of the 'criterion' brought by that feature. It is also known as the Gini importance )

print(
    pd.DataFrame(
        estimator.feature_importances_, columns=["Imp"], index=x_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# Here we will see that importance of features has increased

# %%
print(tree.export_text(estimator, feature_names=feature_names, show_weights=True))

# %%


# %% [markdown]
# ## Model 5 - with  hyperparameter tuning - gridsearch CV (for weight of classes ="balanced")

# %%
# Choose the type of classifier.
estimator5 = DecisionTreeClassifier(random_state=1, class_weight="balanced")

# Grid of parameters to choose from
parameters = {
    "max_depth": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "min_samples_split": [40, 50, 60, 70],
    "criterion": ["entropy", "gini"],
    "splitter": ["best", "random"],
    "min_impurity_decrease": [0.00001, 0.0001, 0.01],
}


# Type of scoring used to compare parameter combinations
scorer = make_scorer(recall_score)

# Run the grid search
grid_obj = GridSearchCV(estimator5, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
estimator5 = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator5.fit(x_train, y_train)

# %%
# Accuracy on train and test
print("Accuracy on training set : ", estimator5.score(x_train, y_train))
print("Accuracy on test set : ", estimator5.score(x_test, y_test))
# Recall on train and test
get_recall_score(estimator5)
get_f1_score(estimator5)

# %%
make_confusion_matrix(estimator5, y_test)

# %%
plt.figure(figsize=(15, 10))

tree.plot_tree(
    estimator5,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
# importance of features in the tree building ( The importance of a feature is computed as the
# (normalized) total reduction of the 'criterion' brought by that feature. It is also known as the Gini importance )

print(
    pd.DataFrame(
        estimator5.feature_importances_, columns=["Imp"], index=x_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# Here we will see that importance of features has increased

# %% [markdown]
# Comment: This model agrees with EDA.

# %%
print(tree.export_text(estimator5, feature_names=feature_names, show_weights=True))

# %%


# %% [markdown]
# ## Model 6 - Decision tree with default class weight and cost complexity pruning

# %% [markdown]
# Cost Complexity Pruning
# 
# The DecisionTreeClassifier provides parameters such as min_samples_leaf and max_depth to prevent a tree from overfiting. Cost complexity pruning provides another option to control the size of a tree. In DecisionTreeClassifier, this pruning technique is parameterized by the cost complexity parameter, ccp_alpha. Greater values of ccp_alpha increase the number of nodes pruned. Here we only show the effect of ccp_alpha on regularizing the trees and how to choose a ccp_alpha based on validation scores.
# Total impurity of leaves vs effective alphas of pruned tree
# 
# Minimal cost complexity pruning recursively finds the node with the "weakest link". The weakest link is characterized by an effective alpha, where the nodes with the smallest effective alpha are pruned first. To get an idea of what values of ccp_alpha could be appropriate, scikit-learn provides DecisionTreeClassifier.cost_complexity_pruning_path that returns the effective alphas and the corresponding total leaf impurities at each step of the pruning process. As alpha increases, more of the tree is pruned, which increases the total impurity of its leaves.

# %%
clf = DecisionTreeClassifier(random_state=1)
path = clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# %%
pd.DataFrame(path)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()

# %%
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

# %%
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

# %%
train_scores = [clf.score(x_train, y_train) for clf in clfs]
test_scores = [clf.score(x_test, y_test) for clf in clfs]

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# %%
index_best_model = np.argmax(test_scores)
best_model = clfs[index_best_model]
print(best_model)
print("Training accuracy of best model: ", best_model.score(x_train, y_train))
print("Test accuracy of best model: ", best_model.score(x_test, y_test))

# %% [markdown]
# ### Looking for tree with best recall

# %%
recall_train = []
for clf in clfs:
    pred_train2 = clf.predict(x_train)
    values_train = metrics.recall_score(y_train, pred_train2)
    recall_train.append(values_train)

# %%
recall_test = []
for clf in clfs:
    pred_test2 = clf.predict(x_test)
    values_test = metrics.recall_score(y_test, pred_test2)
    recall_test.append(values_test)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("Recall")
ax.set_title("Recall vs alpha for training and testing sets")
ax.plot(ccp_alphas, recall_train, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, recall_test, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# %%
# creating the model where we get highest train and test recall
index_best_model = np.argmax(recall_test)
best_model = clfs[index_best_model]
print(best_model)

# %%
make_confusion_matrix(best_model, y_test)

# %%
# Recall on train and test
get_recall_score(best_model)

# %%
# Recall on train and test
get_f1_score(best_model)

# %%
# Accuracy on train and test
print("Accuracy on training set : ", best_model.score(x_train, y_train))
print("Accuracy on test set : ", best_model.score(x_test, y_test))
# Recall on train and test

# %%
plt.figure(figsize=(17, 15))

tree.plot_tree(
    best_model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
# Text report showing the rules of a decision tree -

print(tree.export_text(best_model, feature_names=feature_names, show_weights=True))

# %%
# importance of features in the tree building ( The importance of a feature is computed as the
# (normalized) total reduction of the 'criterion' brought by that feature. It is also known as the Gini importance )

print(
    pd.DataFrame(
        best_model.feature_importances_, columns=["Imp"], index=x_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# ### Comment: 
#  * This tree seems too simple. 
#  * This model might be giving the highest recall but a buisness would not be able to use it to actually target the potential customers.

# %% [markdown]
# It is interesting that in the Decision Tree Classification the most important is "lead time" when in Logistic Regression/Classification there were other variables that were more or similarly important.

# %% [markdown]
# ## Model Performance Comparison and Conclusions

# %%
comparison_frame = pd.DataFrame(
    {
        "Model": [
            "default decision tree",
            "decision tree (depth 7)",
            "decision tree (depth 6)",
            "decision tree balanced classes weight(depth 7)",
            "default decision treee,GridSearchCV(5)",
            "decision tree,balanced_weight_clas.,GridSearch(6)",
            "default decision tree, cost complexity pruning",
        ],
        "Train_Recall": [0.995, 0.9091, 0.8756, 0.8823, 0.9538, 0.8712, 0.9389],
        "Test_Recall": [0.8326, 0.9075, 0.8733, 0.8814, 0.9546, 0.87, 0.9369],
        "Train Accuracy": [0.9967, 0.8286, 0.8216, 0.8266, 0.7626, 0.81894, 0.7513],
        "Test Accuracy": [0.7876, 0.827, 0.821, 0.8275, 0.7582, 0.81868, 0.7453],
        "Train F1": [0.9975, 0.8752, 0.8662, 0.8706, 0.8416, 0.8642, 0.833],
        "Test F1": [0.8373, 0.8732, 0.86495, 0.8703, 0.8382, 0.863, 0.8285],
    }
)

comparison_frame

# %% [markdown]
# ### Conclusions on comparion of decision tree models

# %% [markdown]
# * It seems model 3 offers the best F1 score with high recall and accuracy. 

# %% [markdown]
# * In principle it is possible the Grid Search model 5 is very close - only there the scorer was set up as recallso the Grid Search maximized the recall.

# %% [markdown]
# * all models performed very well 

# %% [markdown]
# ## Actionable Insights and Recommendations
# 
# - What profitable policies for cancellations and refunds can the hotel adopt?
# - What other recommedations would you suggest to the hotel?

# %% [markdown]
# 1. Most guests come from Online market segment. This makes direct communication easier and one can start campaign of engaging the cusomer in "personalization" of his/her stay (explain below why).
# 
# 2. It is profitable to maintain connections to "repeated_guests" as they are more likely not to cancel bookings (on average 34% of bookings is canceled but forrepeated guests only 0.8%)

# %% [markdown]
# 3. "lead_time" information can be used to minimize losses: canceled bookings have much longer lead_time. 
# 
# * EDA visualization allowed for the recognising the importance of lead time around 150 days. It is also the root node for decision trees.
# * Hotel can try to communicate during this time (several months!) and offer additional personalization to the booking - repeat the information about the options like "meal plan", "special requests", "car parking". Or offer even some of such services awith discounts or as complimentary 

# %% [markdown]
# 4. "average price per room" and "parking space" and "special requests" can help predict/classify bookings. In both the Logistic regression and Decision tree models they came up as more important variables for non-canceling. 

# %% [markdown]
# 5. The busiest month is August (12.5% of bookings) then July (11.1%) and May (10.2%). 
# But also the most cancellations are for bookings or stays during those months (April to August).
# It might be relevant information and one can add additional screening for those bookings especially for April and May as then the cancellations are relatively more often.The cancelations are corresponding to longer "lead_time" (100+ days). So in principlethe customers have time to cancel ahead and do not cause lossess to the hotels. 
# 
# 
# One can plan of the correct strategy - e.g. additional email/text reminders and confirmations or special additional offers or offer discount for future stay, if the guests who consider cancellation - to cancel earlier (canceled bookings correspond to higher average price and the price might be a factor in the cancelation). early cancelation would prevent the revenue losses. One can offer a gradual return for cancellation. 
# 
# 
# 

# %% [markdown]
# 6. 62.7% of non-canceled bookings had 1 or more special requests.
# Cancelled bookings had median - 0 - number of special requests. For non-canceled bookins the median is 1. 
# So guests who did not cancel the bookings tookcare to optimize their stay and included special requests in their 
# bookings. While the guests who canceled did not included special requests. It might be a signature that they were not 
# sure they will use the booking. They did not finalize their plans or did not treat the booking "seriously" (for some reasons).The conclusion is if a customer has special requests in the booking it is less probable he/she will cancel the booking. 
# 
# 

# %% [markdown]
# 7. Different arrival_year data shows slightly different trend (2018,2019 - visualization looks different than for 2017). So it is necessary to follow the trends in real time and implement amore flexible strategy (algorithm similar to decision tree structure)

# %% [markdown]
# 8.Number of previous cancelation migt play a role and represent a group of customers who make a game out of booking and who decided to book quickly and cancel when they find a better offer. For such customers one can offer a "locking the cheaper price" if they agree to no-return in case of cancelation (like cheapest fare in airlines)

# %%


