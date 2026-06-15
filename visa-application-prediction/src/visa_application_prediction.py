# %% [markdown]
# # EasyVisa Project
# 
# 
# ## Objective:
# 
# In FY 2016, the OFLC processed 775,979 employer applications for 1,699,957 positions for temporary and permanent labor certifications. This was a nine percent increase in the overall number of processed applications from the previous year. The process of reviewing every case is becoming a tedious task as the number of applicants is increasing every year.
# 
# 
# ## Data Description
# 
# The data contains the different attributes of the employee and the employer. The detailed data dictionary is given below.
# 
# * case_id: ID of each visa application
# * continent: Information of continent the employee
# * education_of_employee: Information of education of the employee
# * has_job_experience: Does the employee has any job experience? Y= Yes; N = No
# * requires_job_training: Does the employee require any job training? Y = Yes; N = No 
# * no_of_employees: Number of employees in the employer's company
# * yr_of_estab: Year in which the employer's company was established
# * region_of_employment: Information of foreign worker's intended region of employment in the US.
# * prevailing_wage:  Average wage paid to similarly employed workers in a specific occupation in the area of intended employment. The purpose of the prevailing wage is to ensure that the foreign worker is not underpaid compared to other workers offering the same or similar service in the same area of employment. 
# * unit_of_wage: Unit of prevailing wage. Values include Hourly, Weekly, Monthly, and Yearly.
# * full_time_position: Is the position of work full-time? Y = Full Time Position; N = Part Time Position
# * case_status:  Flag indicating if the Visa was certified or denied

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

# Libraries to import decision tree classifier and different ensemble classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier


#
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# ## Data Overview
# 
# - Observations
# - Sanity checks

# %%
dfr = pd.read_csv("EasyVisa.csv")

# %%
dfr

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# 
# 

# %%
dfr.describe()

# %%
dfr[dfr.duplicated()].count()

# %% [markdown]
# Comment: No duplicated rows!

# %% [markdown]
# Comment:Dropping the "case id" column.

# %%
dfr.drop(["case_id"], axis=1, inplace=True)

# %%
dfr

# %%
for col in dfr.columns:
    print("Number of unique values in ", col, len(dfr[col].unique()))

# %% [markdown]
# Comment: "Sanity check" for different values.

# %%
nulldata = dfr.isnull().any(axis=1)

# %%
for n in nulldata.value_counts().sort_index().index:
    if n > 0:
        print(f"For the rows with exactly {n} missing values, NAs are found in:")
        n_miss_per_col = dfr[nulldata == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print("\n\n")

# %%
##check count of NAN
count_nan = len(dfr) - dfr.count()
print(count_nan)

# %% [markdown]
# Comment:No missing values. 

# %%
dfr["unit_of_wage"].unique()

# %% [markdown]
# Summary before vizualizations:
# 
# 1.Missing values: There are no missing values. So no imputing needed.
# 
# 2.Outliers: Decision trees are not sensitive to outliers. 
# So we do not have to worry about it forthe model of the Decision Tree. 
# I will reconsider if the model does not work.
# The sigmoid function tapers the outliers. 
# But the presence of extreme outliers may somehow affect the performance of the model and lowering the performance. 
# 
# 3.No scaling needed as decision trees are not sensitive. 
# 
# 4.No duplicate rows.
# 
# 5. Case_id column removed. 
# 
# 
# Further steps:
# 
# 1. ETA
# 1. Encoding variables
# 
# 2. Decision tree can handle multicollinearity so I will skip this step to see if themodels work.  (Multicolinearity could be used to remove fetures and make hyperparameter tuning faster)

# %%
case1 = dfr["unit_of_wage"].str.contains("Hour")
case2 = dfr["unit_of_wage"].str.contains("Year")
case3 = dfr["unit_of_wage"].str.contains("Week")
case4 = dfr["unit_of_wage"].str.contains("Month")
dfr["wage_fin"] = np.select(
    [case1, case2, case3, case4],
    [
        dfr["prevailing_wage"] * 7 * 5 * 43,
        dfr["prevailing_wage"],
        dfr["prevailing_wage"] * 43,
        dfr["prevailing_wage"] * 12,
    ],
)

# %%
dfr.describe()

# %%
dfr.head()

# %%
dfr.tail()

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
histogram_boxplot(dfr, "no_of_employees")

# %% [markdown]
# It might be useful to bin this column into 0-99,100-999,1000-Max

# %%
histogram_boxplot(dfr, "wage_fin")

# %%
g = sns.catplot(
    x="wage_fin",
    y="case_status",
    row="education_of_employee",
    kind="box",
    orient="h",
    height=1.5,
    aspect=4,
    data=dfr,
)
g.set(xscale="log")

# %% [markdown]
# Comment: Log scale plots for yearly wage (calculated) do not show different values or characteristics of distribtions for th different levels of educaiton and  certified or denied visas.
# 
# BUT the salaries for certified visas are never larger than the denied.they are slightly lower for BS, Phd, MS education level.  

# %%
g = sns.catplot(
    x="wage_fin",
    y="case_status",
    row="continent",
    kind="box",
    orient="h",
    height=1.5,
    aspect=4,
    data=dfr,
)
g.set(xscale="log")

# %% [markdown]
# Comment: Similarly like for education when we look at the continents the certified visas correspond to salaries that on average are not larger than the denied. For Europe, North America, Oceania and South America they are lower.  For AfricaAsia the distributins look very similar. 

# %%
df = dfr[(dfr["no_of_employees"] > 0)]

# %% [markdown]
# Comment: Selectig rows with number o employees larger than zero (the company with zero or negative number of employees does not make sens)

# %%
sns.pairplot(df, "case_status")

# %%
sns.histplot(
    data=df, x="yr_of_estab", hue="case_status", stat="count",
)

# %%
df.describe()

# %% [markdown]
# Removing the negative number of employees rows did not change the statsitics.

# %%
df.info()

# %% [markdown]
# # EDA

# %% [markdown]
# **Questions**:
# 1. Those with higher education may want to travel abroad for a well-paid job. Does education play a role in Visa certification? 
# 
# 2. How does the visa status vary across different continents? 
#  
# 3. Experienced professionals might look abroad for opportunities to improve their lifestyles and career development. Does work experience influence visa status? 
#  
# 4. In the United States, employees are paid at different intervals. Which pay unit is most likely to be certified for a visa? 
#  
# 5. The US government has established a prevailing wage to protect local talent and foreign workers. How does the visa status change with the prevailing wage?

# %% [markdown]
# ## 1 Those with higher education may want to travel abroad for a well-paid job. Does education play a role in Visa certification? 

# %%
sns.histplot(
    data=df,
    x="education_of_employee",
    hue="case_status",
    palette="winter",
    stat="count",
)

# %% [markdown]
# Comment:Relatively more cases certified visa for Master BS and PhD education

# %%
deny = df[df["case_status"] == "Denied"]

# %%
certify = df[df["case_status"] == "Certified"]

# %% [markdown]
# I have split the data into 2 groups-one with denied visa one with certified visa

# %%
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
labeled_barplot(deny, "education_of_employee", perc=True, n=None)

# %%
labeled_barplot(certify, "education_of_employee", perc=True, n=None)

# %% [markdown]
# I separate the  original dataset into 4 different groups with different education level.

# %%
bach = df[df["education_of_employee"] == "Bachelor's"]

# %%
mast = df[df["education_of_employee"] == "Master's"]

# %%
dr = df[df["education_of_employee"] == "Doctorate"]

# %%
hs = df[df["education_of_employee"] == "High School"]

# %%
labeled_barplot(bach, "case_status", perc=True, n=None)

# %% [markdown]
# 62.3% applicats with BS get he visa certified.

# %%
labeled_barplot(hs, "case_status", perc=True, n=None)

# %% [markdown]
# 65.9% of applicants with HS education level have their visa denied.

# %%
labeled_barplot(dr, "case_status", perc=True, n=None)

# %% [markdown]
# 87.3% of applicants with PhD  get their visa certified. It is the highest percentage.

# %%
labeled_barplot(mast, "case_status", perc=True, n=None)

# %% [markdown]
# 78.6% of applicants with Master degree have their visa cetified. 

# %% [markdown]
# ## Answer:  Yes the education plays a role.The percentage of denied cases for HS, BS, MS and PhD are respectively 65.9%, 37.7%, 21.4%, 12.7%. So the number of denied cases is inversely proportional to the length of education

# %%
hs.describe()

# %%
bach.groupby("case_status")["wage_fin"].mean().plot(kind="bar")

# %%
mast.groupby("case_status")["wage_fin"].mean().plot(kind="bar")

# %%
dr.groupby("case_status")["wage_fin"].mean().plot(kind="bar")

# %%
hs.groupby("case_status")["wage_fin"].mean().plot(kind="bar")

# %% [markdown]
# Comment: For High School graduates there is no big difference in average salary between denied and dertified cases.For the BS, MS, PhD - the average salary for denied cases is significantly higher. 

# %% [markdown]
# ## 2. How does the visa status vary across different continents?

# %%
labeled_barplot(df, "continent", perc=True, n=None)

# %%
labeled_barplot(certify, "continent", perc=True, n=None)

# %% [markdown]
# ## Answer: Majority of cretified visas is for candidates from Asia 64.7% then Europe 17.4%. 

# %%
AF = df[df["continent"] == "Africa"]

# %%
AS = df[df["continent"] == "Asia"]

# %%
EU = df[df["continent"] == "Europe"]

# %%
NAM = df[df["continent"] == "North America"]

# %%
SAM = df[df["continent"] == "South America"]

# %%
OC = df[df["continent"] == "Oceania"]

# %%
labeled_barplot(AF, "case_status", perc=True, n=None)

# %%
labeled_barplot(AS, "case_status", perc=True, n=None)

# %%
labeled_barplot(EU, "case_status", perc=True, n=None)

# %%
labeled_barplot(NAM, "case_status", perc=True, n=None)

# %%
labeled_barplot(SAM, "case_status", perc=True, n=None)



# %%
labeled_barplot(OC, "case_status", perc=True, n=None)

# %% [markdown]
# ## Answer:Smallest denied percentage is for visa from Europe (1/5 th). Then Africa Asia and Oceania, North America ans South America.

# %% [markdown]
# ## 3. Experienced professionals might look abroad for opportunities to improve their lifestyles and career development. Does work experience influence visa status? 

# %%
stacked_barplot(df, "has_job_experience", "case_status")

# %% [markdown]
# ## Answer: Job experience has influence on the visas certifcation
# .

# %% [markdown]
# ## 4. In the United States, employees are paid at different intervals. Which pay unit is most likely to be certified for a visa? 

# %%
labeled_barplot(certify, "unit_of_wage", perc=True, n=None)

# %% [markdown]
# 94.3% of certified visa is for yearly wage applicants.

# %% [markdown]
# ## Answer: Yearly wage unit applications is most likely to be certified with visa.

# %% [markdown]
# ## 5.The US government has established a prevailing wage to protect local talent and foreign workers. How does the visa status change with the prevailing wage?

# %%
sns.boxplot(x="case_status", y="prevailing_wage", data=df)



# %%
g = sns.catplot(
    x="prevailing_wage",
    y="case_status",
    row="education_of_employee",
    kind="box",
    orient="h",
    height=1.5,
    aspect=4,
    data=dfr,
)
g.set(xscale="log")

# %%
sns.displot(
    df,
    x="prevailing_wage",
    hue="case_status",
    stat="probability",
    kde=True,
    common_norm=False,
    element="step",
)

# %%
sns.catplot(x="case_status", y="prevailing_wage", data=df, height=5, aspect=5.5)

# %%
sns.boxplot(x="unit_of_wage", y="prevailing_wage", data=deny)

# %%
sns.boxplot(x="unit_of_wage", y="prevailing_wage", data=certify)

# %% [markdown]
# ## Answer: Prevailing wage for both denied and certifiedvas applicants subsets has similar means and range, but for certified visa candidates the prevailing wage is higher. 

# %%
week = df[df["unit_of_wage"] == "Week"]

# %%
sns.catplot(x="continent", y="yr_of_estab", hue="case_status", kind="box", data=week)

# %%
year = df[df["unit_of_wage"] == "Year"]

# %%
sns.catplot(x="continent", y="yr_of_estab", hue="case_status", kind="box", data=year)

# %%
month = df[df["unit_of_wage"] == "Month"]

# %%
sns.catplot(x="continent", y="yr_of_estab", hue="case_status", kind="box", data=month)

# %%
hour = df[df["unit_of_wage"] == "Hour"]

# %%
sns.catplot(x="continent", y="yr_of_estab", hue="case_status", kind="box", data=hour)

# %%
sns.boxplot(x="unit_of_wage", y="wage_fin", data=certify)

# %%
sns.boxplot(x="region_of_employment", y="wage_fin", data=month)

# %%
sns.boxplot(x="region_of_employment", y="wage_fin", data=week)

# %%
sns.boxplot(x="region_of_employment", y="wage_fin", data=year)

# %%
sns.boxplot(x="region_of_employment", y="wage_fin", data=hour)

# %%
sns.boxplot(x="education_of_employee", y="prevailing_wage", data=month)

# %%
sns.boxplot(x="education_of_employee", y="prevailing_wage", data=week)

# %%
sns.boxplot(x="education_of_employee", y="prevailing_wage", data=hour)

# %%
sns.boxplot(x="education_of_employee", y="prevailing_wage", data=year)

# %% [markdown]
# Answer: There does not seem to be big differences for prevailing wage data for denied and certified visas.

# %% [markdown]
# 

# %%
df.info()

# %%
df.groupby("has_job_experience")["wage_fin"].mean().plot(kind="bar")

# %%
certify.groupby("has_job_experience")["wage_fin"].mean().plot(kind="bar")

# %%
deny.groupby("has_job_experience")["wage_fin"].mean().plot(kind="bar")

# %%
# function to plot stacked bar chart


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

# %%
stacked_barplot(df, "continent", "case_status")


# %%
stacked_barplot(df, "education_of_employee", "case_status")

# %%
stacked_barplot(df, "region_of_employment", "case_status")

# %%
stacked_barplot(df, "unit_of_wage", "case_status")

# %%
stacked_barplot(df, "full_time_position", "case_status")

# %% [markdown]
# Comments:
# 
#     * full timeposition does not change the odds of getting visa
#     
#     * 60+ % denied cases for hourly wage 
#     
#     * most certified (70%)visas have Midwest and Southwest regions
#     
#     * 60+% denied cases for education feature are for high school education level
#     
#     * highest chances of getting visa have candidateswith Masetr and Phd degree
#     
#     * Most certified visa for continent feature are from Europe least from South America
#     
#     * prior job experience increases chances of getting visa.
#     
#     

# %% [markdown]
# # Preparing the data set for model building

# %%
cols = df.select_dtypes(["object"])
cols.columns

# %%
for i in cols.columns:
    df[i] = df[i].astype("category")

# %%
df.info()

# %%
df.describe(include=["category"]).T

# %% [markdown]
# # Split the data

# %%
colscat = [
    "continent",
    "education_of_employee",
    "has_job_experience",
    "requires_job_training",
    "region_of_employment",
    "unit_of_wage",
    "full_time_position",
]

# %%
data = pd.get_dummies(data=df, columns=colscat, drop_first=True)

# %% [markdown]
# I have added an annual wage column.But since one of the questions mentons type ofwage I donot want to remove the 2 original "wage columns". I will prepare 2 sets one with new column ("wage_fin")and one where the new column replcase the orignal 2 columns.

# %%
data

# %%
X = data.drop(["case_status"], axis=1)
y = data["case_status"].apply(lambda x: 1 if x == "Certified" else 0)

# %%
# Splitting data into training and test set:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
print(X_train.shape, X_test.shape)

# %%
y.value_counts(1)

# %%
y_test.value_counts(1)

# %% [markdown]
# Goal- maximize the f1 score

# %%
# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance
    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score
    roc = roc_auc_score(target, pred)
    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "ROC-AUC": roc,
        },
        index=[0],
    )
    return df_perf

# %%
def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
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
# #  Decision Tree Model

# %% [markdown]
# * We will build our model using the DecisionTreeClassifier function. Using default 'gini' criteria to split. 
# * If the frequency of class A is 10% and the frequency of class B is 90%, then class B will become the dominant 
# class and the decision tree will become biased toward the dominant classes.
# 
# * In this case, we can pass a dictionary {0:0.67,1:0.33} to the model to specify the weight of each class and 
# the decision tree will give more weightage to class 1.
# 
# * class_weight is a hyperparameter for the decision tree classifier.
# * I will also try "balanced" option for class_weight

# %%
dtree = DecisionTreeClassifier(
    criterion="gini", class_weight={0: 0.67, 1: 0.33}, random_state=1
)

# %%
dtree.fit(X_train, y_train)

# %%
confusion_matrix_sklearn(dtree, X_test, y_test)

# %%
dtree_model_train_perf = model_performance_classification_sklearn(
    dtree, X_train, y_train
)
print("Training performance \n", dtree_model_train_perf)

# %%
dtree_model_test_perf = model_performance_classification_sklearn(dtree, X_test, y_test)
print("Testing performance \n", dtree_model_test_perf)

# %%
feature_names = list(X.columns)

# %%
plt.figure(figsize=(20, 30))
tree.plot_tree(
    dtree,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %%
print(
    pd.DataFrame(
        dtree.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
feature_names = X_train.columns
importances = dtree.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %% [markdown]
# ### Comment: Decision tree is working well on the training data but is not able to generalize well on the test data concerning the recall.

# %% [markdown]
# Now trying with balanced class_weights

# %%
dtree2 = DecisionTreeClassifier(
    criterion="gini", class_weight="balanced", random_state=1
)

# %%
dtree2.fit(X_train, y_train)

# %%
confusion_matrix_sklearn(dtree2, X_test, y_test)

# %%
dtree2_model_train_perf = model_performance_classification_sklearn(
    dtree2, X_train, y_train
)
print("Training performance \n", dtree2_model_train_perf)

# %%
dtree2_model_test_perf = model_performance_classification_sklearn(
    dtree2, X_test, y_test
)
print("Testing performance \n", dtree2_model_test_perf)

# %% [markdown]
# ### Comment: There is very little difference but the previous model worked a bit better.

# %% [markdown]
# ### Comment:(I checked for multicolinearity because if the hyperparameter tuning takes toolong I will remove the linearly dependent columns)

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
checking_vif(X_train)

# %% [markdown]
# #### Comment:yr_of_estab. is strongly multicolinear feature.

# %% [markdown]
# # Hyperparameter Tuning decision tree model

# %%
# Choose the type of classifier.
dtree_estimator = DecisionTreeClassifier(
    class_weight={0: 0.67, 1: 0.33}, random_state=1
)

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(2, 30),
    "min_samples_leaf": [1, 2, 5, 7, 10],
    "max_leaf_nodes": [2, 3, 5, 10, 15],
    "min_impurity_decrease": [0.0001, 0.001, 0.01, 0.1],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_estimator, parameters, scoring=scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
dtree_estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
dtree_estimator.fit(X_train, y_train)

# %% [markdown]
# #### Comment - Hyperparametertuned decision tree has very small max_depth. We have "very general" model but with limited applicability in real life. 

# %%
confusion_matrix_sklearn(dtree_estimator, X_test, y_test)

# %%
dtree_estimator_model_train_perf = model_performance_classification_sklearn(
    dtree_estimator, X_train, y_train
)
print("Training performance \n", dtree_estimator_model_train_perf)

# %%
dtree_estimator_model_test_perf = model_performance_classification_sklearn(
    dtree_estimator, X_test, y_test
)
print("Testing performance \n", dtree_estimator_model_test_perf)

# %%
print(
    pd.DataFrame(
        dtree_estimator.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# #### Comment: Very simple model that is already a good model with not much overfitting. Although it is extremely simple - the only feature that is important is High School education level.  (60+% percent denied visas are for this education level)

# %% [markdown]
# ### In order to compare with the other models it would be good to have more features,so I repeat gridsearch reoving the possibility to have too small tree. 

# %% [markdown]
# # Hyperparameter tuning Decision tree - 2nd model (to compare feature importance between different models)

# %%
# Choose the type of classifier.
dtree_estimatorb = DecisionTreeClassifier(
    class_weight={0: 0.67, 1: 0.33}, random_state=1
)

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(5, 7),
    "min_samples_leaf": [5, 7, 10],
    "max_leaf_nodes": [5, 10],
    "min_impurity_decrease": [0.0001, 0.001],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_estimatorb, parameters, scoring=scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
dtree_estimatorb = grid_obj.best_estimator_

# Fit the best algorithm to the data.
dtree_estimatorb.fit(X_train, y_train)

# %%
confusion_matrix_sklearn(dtree_estimatorb, X_test, y_test)

# %%
dtree_estimatorb_model_train_perf = model_performance_classification_sklearn(
    dtree_estimatorb, X_train, y_train
)
print("Training performance \n", dtree_estimatorb_model_train_perf)

# %%
dtree_estimatorb_model_test_perf = model_performance_classification_sklearn(
    dtree_estimatorb, X_test, y_test
)
print("Testing performance \n", dtree_estimatorb_model_test_perf)

# %%
print(
    pd.DataFrame(
        dtree_estimatorb.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
importances = dtree_estimatorb.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="blue", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
plt.figure(figsize=(20, 10))
tree.plot_tree(
    dtree_estimatorb,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()

# %% [markdown]
# #### Comment: It is very good model with no overfitting and high scores. The following features are important, in order ofdecreasing importance: education_of_employee_High School, has_job_experience_Y, education_of_employee_Master, continent_Europe, prevailing_wage, education_of_employee_Doctorate, region_of_employment_Midwest, unit_of_wage_Year.
# 
# These features were important alredy in EDA so the model isconsistent with the data vizualization.  
# 
# Surprisingly the previous model - default without hyperparameter tuning has better scores. 

# %% [markdown]
# # Prepare the randomized half sample set to get faster hyperparameter tuning 

# %%
df2 = data.sample(frac=0.5)

# %%
Xt = df2.drop(["case_status"], axis=1)
yt = df2["case_status"].apply(lambda x: 1 if x == "Certified" else 0)

# %%
len(yt)

# %%
# Splitting data into training and test set:
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    Xt, yt, test_size=0.3, random_state=1
)
print(X_train2.shape, X_test2.shape)

# %% [markdown]
# # Random Forest

# %%
# Fitting the model
rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train, y_train)

# Calculating different metrics
rf_estimator_model_train_perf = model_performance_classification_sklearn(
    rf_estimator, X_train, y_train
)
print("Training performance:\n", rf_estimator_model_train_perf)
rf_estimator_model_test_perf = model_performance_classification_sklearn(
    rf_estimator, X_test, y_test
)
print("Testing performance:\n", rf_estimator_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(rf_estimator, X_test, y_test)

# %% [markdown]
# Comment: Random forest model has some overfitting so it is not very general. 

# %%
feature_names = X_train.columns

# %%
importances = rf_estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="blue", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        rf_estimator.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# #### Comment: The Random forest model is overfittng. As the model is more complicated it includes other factors thn only High school education. In order of decreasing importance: prevailing_wage, no_of_employees, wage_fin                           yr_of_estab, education_of_employee_High School, has_job_experience_Y.
# 

# %% [markdown]
# ## For parameter tuning I will use now smaller set. 

# %% [markdown]
# #  Hyperparameter Tuning Random Forest
# 

# %%
# Choose the type of classifier.
rf_estimator2 = RandomForestClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [110, 200],
    "max_features": [0.7, 0.9, "log2", "auto"],
    "max_samples": [0.7, 0.9, None],
}

# Run the grid search
grid_obj = GridSearchCV(rf_estimator2, parameters, scoring="f1", cv=5)
grid_obj = grid_obj.fit(X_train2, y_train2)

# Set the clf to the best combination of parameters
rf_estimator2 = grid_obj.best_estimator_

# Fit the best algorithm to the data.
rf_estimator2.fit(X_train, y_train)

# %%
confusion_matrix_sklearn(rf_estimator2, X_test, y_test)

# %%
rf_estimator2_model_train_perf = model_performance_classification_sklearn(
    rf_estimator2, X_train, y_train
)
print("Training performance \n", rf_estimator_model_train_perf)

# %%
rf_estimator2_model_test_perf = model_performance_classification_sklearn(
    rf_estimator2, X_test, y_test
)
print("Testing performance \n", rf_estimator_model_test_perf)

# %%
# importance of features in the tree building ( The importance of a feature is computed as the
# (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print(
    pd.DataFrame(
        rf_estimator2.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
importances = rf_estimator2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %% [markdown]
# ### Comments:Strong overfitting. Model is not better thn "default" random forest. 
# Features in decreasing order of importance:
# prevailing_wage, no_of_employees,wage_fin, yr_of_estab, education_of_employee_High School, has_job_experience_Y,              education_of_employee_Master,education_of_employee_Doctorate, unit_of_wage_Year, continent_Europe                  

# %% [markdown]
# # Building bagging and boosting models

# %% [markdown]
# # Bagging classifier default model

# %%
# Fitting the model
bagging_classifier = BaggingClassifier(random_state=1)
bagging_classifier.fit(X_train, y_train)

# Calculating different metrics
bagging_classifier_model_train_perf = model_performance_classification_sklearn(
    bagging_classifier, X_train, y_train
)
print(bagging_classifier_model_train_perf)
bagging_classifier_model_test_perf = model_performance_classification_sklearn(
    bagging_classifier, X_test, y_test
)
print(bagging_classifier_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(bagging_classifier, X_test, y_test)

# %% [markdown]
# #### Comment: For the default bagging classifier the model performs very well for train data but worse for test. So it overfits to noise in the data set.

# %% [markdown]
# # Hyperparameter Tuning for bagging classifier with decision trees

# %%
# grid search for bagging classifier
cl1 = DecisionTreeClassifier(class_weight={0: 0.67, 1: 0.33}, random_state=1)
param_grid = {
    "base_estimator": [cl1],
    "n_estimators": [5, 7, 15, 51, 101],
    "max_features": [0.7, 0.8, 0.9, 1],
}

grid = GridSearchCV(
    BaggingClassifier(random_state=1, bootstrap=True),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
)
grid.fit(X_train, y_train)

# %%
## getting the best estimator
bagging_estimator = grid.best_estimator_
bagging_estimator.fit(X_train, y_train)

# %%
confusion_matrix_sklearn(bagging_estimator, X_test, y_test)

# %%
bagging_estimator_model_train_perf = model_performance_classification_sklearn(
    bagging_estimator, X_train, y_train
)
print("Training performance \n", bagging_estimator_model_train_perf)

# %%
bagging_estimator_model_test_perf = model_performance_classification_sklearn(
    bagging_estimator, X_test, y_test
)
print("Testing performance \n", bagging_estimator_model_test_perf)

# %% [markdown]
# #### Comment: Ensemble models are less interpretable than decision tree but bagging classifier is even less interpretable than random forest. It does not even have a feature importance attribute. And we have overfitting after hyperparameter tuning.

# %% [markdown]
# # Boosting models

# %% [markdown]
# # Adaboost classifier

# %%
abc = AdaBoostClassifier(random_state=1)
abc.fit(X_train, y_train)

# %%
abc_model_train_perf = model_performance_classification_sklearn(abc, X_train, y_train)
print("Training performance \n", abc_model_train_perf)

# %%
abc_model_test_perf = model_performance_classification_sklearn(abc, X_test, y_test)
print("Testing performance \n", abc_model_test_perf)

# %%
confusion_matrix_sklearn(abc, X_test, y_test)

# %%
importances = abc.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="green", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        abc.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# ### Comment: 
# #### Already a default Adaboost model works very well. The scores are high and there is no overfitting.The following features are important: no_of_employees, wage_fin, prevailing_wage, yr_of_estab, education_of_employee_Master,region_of_employment_Midwest, education_of_employee_High School,continent_Europe,unit_of_wage_Year

# %% [markdown]
# # Gradient Boosting Classifier

# %%
gbc = GradientBoostingClassifier(random_state=1)
gbc.fit(X_train, y_train)

# %%
gbc_model_train_perf = model_performance_classification_sklearn(gbc, X_train, y_train)
print("Training performance \n", gbc_model_train_perf)

# %%
gbc_model_test_perf = model_performance_classification_sklearn(gbc, X_test, y_test)
print("Training performance \n", gbc_model_test_perf)

# %%
confusion_matrix_sklearn(gbc, X_test, y_test)

# %%
importances = gbc.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="purple", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        gbc.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# ### Comment: Model performs similarly for test and rain sets - so no overfitting. F1 and recall scores are v.good. 

# %% [markdown]
# Important features:education_of_employee_High,School,has_job_experience_Y, prevailing_wage,education_of_employee_Master, education_of_employee_Doctorate,continent_Europe,unit_of_wage_Year,region_of_employment_Midwest

# %% [markdown]
# # XGboost Classifier

# %%
xgb = XGBClassifier(random_state=1, eval_metric="logloss")
xgb.fit(X_train, y_train)

# %%
xgb_classifier_model_train_perf = model_performance_classification_sklearn(
    xgb, X_train, y_train
)
print("Training performance:\n", xgb_classifier_model_train_perf)
xgb_classifier_model_test_perf = model_performance_classification_sklearn(
    xgb, X_test, y_test
)
print("Testing performance:\n", xgb_classifier_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(xgb, X_test, y_test)

# %%
importances = xgb.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        xgb.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# #### Comment: 
# Xgboost classifier workswell the only small issue is the small difference between test and train - for  f1 score. 
# But all of the scores are high (the highest). 
# 
# Important features: unit_of_wage_Year,education_of_employee_High School, has_job_experience_Y, education_of_employee_Doctorate, 
# continent_Europe, education_of_employee_Master, region_of_employment_Midwest, full_time_position_Y   

# %% [markdown]
# # Boosting models hyperparameter tuning

# %% [markdown]
# # Adaboost hyperparameter tuning 

# %%
# Choose the type of classifier.
abc_tuned = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    # Let's try different max_depth for base_estimator
    "base_estimator": [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=3),
    ],
    "n_estimators": np.arange(10, 90, 10),
    "learning_rate": np.arange(0.1, 2, 0.1),
}

# Type of scoring used to compare parameter  combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(abc_tuned, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(X_train2, y_train2)

# Set the clf to the best combination of parameters
abc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
abc_tuned.fit(X_train, y_train)

# %%
# Calculating different metrics
abc_tuned_model_train_perf = model_performance_classification_sklearn(
    abc_tuned, X_train, y_train
)
print(abc_tuned_model_train_perf)
abc_tuned_model_test_perf = model_performance_classification_sklearn(
    abc_tuned, X_test, y_test
)
print(abc_tuned_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(abc_tuned, X_test, y_test)

# %%
importances = abc_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        abc_tuned.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# #### Comment:
# Adaboost hyperparameter tuned classifier works very well. It has remarkably high recall and no overfitting. 
# 
# The most important features:education_of_employee_High School, has_job_experience_Y, education_of_employee_Master,
# continent_Europe, prevailing_wage, education_of_employee_Doctorate, region_of_employment_Midwest.     

# %% [markdown]
# # Gradient Boost hyperparameter tuning

# %%
# Choose the type of classifier.
gbc_tuned = GradientBoostingClassifier(
    init=AdaBoostClassifier(random_state=1), random_state=1
)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [100, 150, 200],
    "subsample": [0.8, 0.9, 1],
    "max_features": [0.7, 0.8, 0.9, 1],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(gbc_tuned, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(X_train2, y_train2)

# Set the clf to the best combination of parameters
gbc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
gbc_tuned.fit(X_train, y_train)

# %%
# Calculating different metrics
gbc_tuned_model_train_perf = model_performance_classification_sklearn(
    gbc_tuned, X_train, y_train
)
print("Training performance:\n", gbc_tuned_model_train_perf)
gbc_tuned_model_test_perf = model_performance_classification_sklearn(
    gbc_tuned, X_test, y_test
)
print("Testing performance:\n", gbc_tuned_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(gbc_tuned, X_test, y_test)

# %%
importances = gbc_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        gbc_tuned.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# #### Comment:
# Gradient Boost hyperparaeter tuned  is very good model. All scores are high, no overfitting.
# 
# The important features: education_of_employee_High School, has_job_experience_Y, prevailing_wage,
# education_of_employee_Master, continent_Europe, education_of_employee_Doctorate.

# %% [markdown]
# # XGBoost tuning

# %%
# Choose the type of classifier.
xgb_tuned = XGBClassifier(random_state=1, eval_metric="logloss")

# Grid of parameters to choose from
parameters = {
    "n_estimators": [10, 30, 50],
    "scale_pos_weight": [1, 2, 5],
    "subsample": [0.7, 0.9, 1],
    "learning_rate": [0.05, 0.1, 0.2],
    "colsample_bytree": [0.7, 0.9, 1],
    "colsample_bylevel": [0.5, 0.7, 1],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(xgb_tuned, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(X_train2, y_train2)

# Set the clf to the best combination of parameters
xgb_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
xgb_tuned.fit(X_train, y_train)

# %%
# Calculating different metrics
xgb_tuned_model_train_perf = model_performance_classification_sklearn(
    xgb_tuned, X_train, y_train
)
print("Training performance:\n", xgb_tuned_model_train_perf)
xgb_tuned_model_test_perf = model_performance_classification_sklearn(
    xgb_tuned, X_test, y_test
)
print("Testing performance:\n", xgb_tuned_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(xgb_tuned, X_test, y_test)

# %%
importances = xgb_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        xgb_tuned.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# #### Comment:
# XGBoost hyperparaeter tuned  is another very good model. All scores are high, no overfitting.
# 
# The important features: education_of_employee_High School, has_job_experience_Y, 
# education_of_employee_Master,  education_of_employee_Doctorate,unit_of_wage_Year,continent_Europe,region_of_employment_Midwest.

# %% [markdown]
# # Stacking classifier

# %%
estimators = [
    ("Random Forest", rf_estimator2),
    ("Gradient Boosting", gbc_tuned),
    ("Decision Tree", dtree_estimator),
]

final_estimator = xgb_tuned

stacking_classifier = StackingClassifier(
    estimators=estimators, final_estimator=final_estimator
)

stacking_classifier.fit(X_train, y_train)

# %%
# Calculating different metrics
stacking_classifier_model_train_perf = model_performance_classification_sklearn(
    stacking_classifier, X_train, y_train
)
print("Training performance:\n", stacking_classifier_model_train_perf)
stacking_classifier_model_test_perf = model_performance_classification_sklearn(
    stacking_classifier, X_test, y_test
)
print("Testing performance:\n", stacking_classifier_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(stacking_classifier, X_test, y_test)

# %% [markdown]
# #### Comment: Stacking classifier has the best scores. 

# %% [markdown]
# ##  Will tuning the hyperparameters improve the model performance?

# %% [markdown]
# It should but sometimes only one score get better bt the other scores get worse. I discuss it below.

# %% [markdown]
# ## Model Performance Comparison and Conclusions

# %%
# training performance comparison

models_train_comp_df = pd.concat(
    [
        dtree_model_train_perf.T,
        dtree_estimator_model_train_perf.T,
        dtree_estimatorb_model_train_perf.T,
        rf_estimator_model_train_perf.T,
        rf_estimator2_model_train_perf.T,
        bagging_classifier_model_train_perf.T,
        bagging_estimator_model_train_perf.T,
        abc_model_train_perf.T,
        abc_tuned_model_train_perf.T,
        gbc_model_train_perf.T,
        gbc_tuned_model_train_perf.T,
        xgb_classifier_model_train_perf.T,
        xgb_tuned_model_train_perf.T,
        stacking_classifier_model_train_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree",
    "Small Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Default Bagging Classifier",
    "Bagging Estimator Tuned",
    "Adaboost Classifier",
    "Adabosst Classifier Tuned",
    "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned",
    "XGBoost Classifier",
    "XGBoost Classifier Tuned",
    "Stacking Classifier",
]
print("Training performance comparison:")
models_train_comp_df

# %% [markdown]
# ### Comment 
# Bagging classifier got bettter score after tuning. 
# Adaboost tuned model has better Accuracy,Precision f1,ROC-AUC but slightly worse recall. Tuned gradient 
# boost classifier has slightly better Accuracy, precision, ROC-AUC but worse f1 and recall. XGBoost after tuning got 
# worse scores but there is less overfitting. Al of the yperparameter tuned models have no overfitting. 
# 

# %%
# training performance comparison

models_train_comp_df = pd.concat(
    [
        dtree_model_train_perf.T - dtree_model_test_perf.T,
        dtree_estimator_model_train_perf.T - dtree_estimator_model_test_perf.T,
        dtree_estimatorb_model_train_perf.T - dtree_estimatorb_model_test_perf.T,
        rf_estimator_model_train_perf.T - rf_estimator_model_test_perf.T,
        rf_estimator2_model_train_perf.T - rf_estimator2_model_test_perf.T,
        bagging_classifier_model_train_perf.T - bagging_classifier_model_test_perf.T,
        bagging_estimator_model_train_perf.T - bagging_estimator_model_test_perf.T,
        abc_model_train_perf.T - abc_model_test_perf.T,
        abc_tuned_model_train_perf.T - abc_tuned_model_test_perf.T,
        gbc_model_train_perf.T - gbc_model_test_perf.T,
        gbc_tuned_model_train_perf.T - gbc_tuned_model_test_perf.T,
        xgb_classifier_model_train_perf.T - xgb_classifier_model_test_perf.T,
        xgb_tuned_model_train_perf.T - xgb_tuned_model_test_perf.T,
        stacking_classifier_model_train_perf.T - stacking_classifier_model_test_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree",
    "Small Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Default Bagging Classifier",
    "Bagging Estimator Tuned",
    "Adaboost Classifier",
    "Adabosst Classifier Tuned",
    "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned",
    "XGBoost Classifier",
    "XGBoost Classifier Tuned",
    "Stacking Classifier",
]
print("Training performance comparison (difference in scores for train and test sets):")
models_train_comp_df

# %% [markdown]
# ### Conclusions: 
#     
#     1. Best model - XGbot hyperparameter tuned classifier is the best. As it was optional model - among the other models - stacking classifier is best (Highest f1score and smallest difference between train and test set scores).
#     
#     2. Hyperparameter tuning takes a long time. For XGBoost it improves to get less overfitting.
#     
#     3. All boosting methods have no overfitting and high scores. 
#     
#     4. Comparison and analysis of the importance of the features will be done below
#     

# %%


# %% [markdown]
# # NOTE:  Boosting models are sensitive to outliers (decision trees are robust to outliers but boosting methods not) so I repeat the calculations after treating the outliers.

# %%
# let's plot the boxplots of all columns to check for outliers
plt.figure(figsize=(7, 10))

# for i, variable in enumerate(numeric_columns):
plt.subplot(1, 3, 1)
plt.boxplot(data["no_of_employees"], whis=1.5)
plt.tight_layout()
plt.title("No of Emoplees")
plt.subplot(1, 3, 2)
plt.boxplot(data["prevailing_wage"], whis=1.5)
plt.tight_layout()
plt.title("Prev. wage")
plt.subplot(1, 3, 3)
plt.boxplot(data["wage_fin"], whis=1.5)
plt.tight_layout()
plt.title("Annual wage")

plt.show()



# %%
def treat_outliers(df, col):
    """
    treats outliers in a variable
    col: str, name of the numerical variable
    df: dataframe
    col: name of the column
    """
    Q1 = df[col].quantile(0.25)  # 25th quantile
    Q3 = df[col].quantile(0.75)  # 75th quantile
    IQR = Q3 - Q1
    Lower_Whisker = Q1 - 1.5 * IQR
    Upper_Whisker = Q3 + 1.5 * IQR

    # all the values smaller than Lower_Whisker will be assigned the value of Lower_Whisker
    # all the values greater than Upper_Whisker will be assigned the value of Upper_Whisker
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker)

    return df


def treat_outliers_all(df, col_list):
    """
    treat outlier in all numerical variables
    col_list: list of numerical variables
    df: data frame
    """
    for c in col_list:
        df = treat_outliers(df, c)

    return df

# %%
outlier_col = [
    "wage_fin",
    "no_of_employees",
    "prevailing_wage",
]

# %%
dataout = treat_outliers_all(data, outlier_col)

# %%
dataout.info()

# %% [markdown]
# Now will split data into regularand small set. 

# %%
dfout2 = dataout.sample(frac=0.5)

# %%
Xto = dfout2.drop(["case_status"], axis=1)
yto = dfout2["case_status"].apply(lambda x: 1 if x == "Certified" else 0)


# %%
# Splitting data into training and test set:
X_traino2, X_testo2, y_traino2, y_testo2 = train_test_split(
    Xto, yto, test_size=0.3, random_state=1
)
print(X_traino2.shape, X_testo2.shape)

# %%
Xo = dataout.drop(["case_status"], axis=1)
yo = dataout["case_status"].apply(lambda x: 1 if x == "Certified" else 0)


# %%
# Splitting data into training and test set:
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    Xo, yo, test_size=0.3, random_state=1, stratify=y
)
print(X_train3.shape, X_test3.shape)

# %%
print(y.value_counts(1))
print(y_test.value_counts(1))

# %% [markdown]
# ## I will now repeat calculations for bossting methods and random forest using the same methodology as before (hyperparameter tunig using smaller set of 50% ofrandomly selected rows.)  

# %% [markdown]
# # Random Forest trained on set without outliers

# %%

# Fitting the model
rfo_estimator = RandomForestClassifier(random_state=1)
rfo_estimator.fit(X_train3, y_train3)

# Calculating different metrics
rfo_estimator_model_train_perf = model_performance_classification_sklearn(
    rfo_estimator, X_train3, y_train3
)
print("Training performance:\n", rfo_estimator_model_train_perf)
rfo_estimator_model_test_perf = model_performance_classification_sklearn(
    rf_estimator, X_test3, y_test3
)
print("Testing performance:\n", rfo_estimator_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(rfo_estimator, X_test3, y_test3)


# %%
importances = rfo_estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="blue", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


print(
    pd.DataFrame(
        rfo_estimator.feature_importances_, columns=["Imp"], index=X_train3.columns
    ).sort_values(by="Imp", ascending=False)
)

# %% [markdown]
# Comment: Overfitted model.Among the most important features it has features related to employer.

# %% [markdown]
# # Random Forest hyperparameter tuning 

# %%
# Choose the type of classifier.
rfo_estimator2 = RandomForestClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [110, 200],
    "max_features": [0.7, 0.9, "log2", "auto"],
    "max_samples": [0.7, 0.9, None],
}

# Run the grid search
grid_obj = GridSearchCV(rfo_estimator2, parameters, scoring="f1", cv=5)
grid_obj = grid_obj.fit(X_traino2, y_traino2)

# Set the clf to the best combination of parameters
rfo_estimator2 = grid_obj.best_estimator_

# Fit the best algorithm to the data.
rfo_estimator2.fit(X_train3, y_train3)

# %%
importances = rfo_estimator2.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="blue", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


print(
    pd.DataFrame(
        rfo_estimator2.feature_importances_, columns=["Imp"], index=X_train3.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
confusion_matrix_sklearn(rfo_estimator2, X_test3, y_test3)
rfo_estimator2_model_train_perf = model_performance_classification_sklearn(
    rfo_estimator2, X_train3, y_train3
)
print("Training performance \n", rfo_estimator2_model_train_perf)

rfo_estimator2_model_test_perf = model_performance_classification_sklearn(
    rfo_estimator2, X_test3, y_test3
)
print("Testing performance \n", rfo_estimator2_model_test_perf)



# %% [markdown]
# Comment: Model still is overfitting. 

# %% [markdown]
# # Adaboost classifier

# %%
abco = AdaBoostClassifier(random_state=1)
abco.fit(X_train3, y_train3)
abco_model_train_perf = model_performance_classification_sklearn(
    abco, X_train3, y_train3
)
print("Training performance \n", abco_model_train_perf)
abco_model_test_perf = model_performance_classification_sklearn(abco, X_test3, y_test3)
print("Testing performance \n", abco_model_test_perf)
confusion_matrix_sklearn(abco, X_test3, y_test3)

# %%
importances = abco.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="blue", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


print(
    pd.DataFrame(
        abco.feature_importances_, columns=["Imp"], index=X_train3.columns
    ).sort_values(by="Imp", ascending=False)
)



# %% [markdown]
# 
# Comment: No overfitting and very good scores.

# %% [markdown]
# # Adaboost hyperparameter tuning

# %%
# Choose the type of classifier.
abco_tuned = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    # Let's try different max_depth for base_estimator
    "base_estimator": [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=3),
    ],
    "n_estimators": np.arange(10, 90, 10),
    "learning_rate": np.arange(0.1, 2, 0.1),
}

# Type of scoring used to compare parameter  combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(abc_tuned, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(X_traino2, y_traino2)

# Set the clf to the best combination of parameters
abco_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
abco_tuned.fit(X_train3, y_train3)

# %%
importances = abco_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="blue", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

print(
    pd.DataFrame(
        abco_tuned.feature_importances_, columns=["Imp"], index=X_train3.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
confusion_matrix_sklearn(abco_tuned, X_test3, y_test3)
abco_tuned_model_train_perf = model_performance_classification_sklearn(
    abco_tuned, X_train3, y_train3
)
print("Training performance \n", abco_tuned_model_train_perf)

abco_tuned_model_test_perf = model_performance_classification_sklearn(
    abco_tuned, X_test3, y_test3
)
print("Testing performance \n", abco_tuned_model_test_perf)



# %% [markdown]
# Comment: Hypertuning improved the model.

# %% [markdown]
# # Gradient Boost Classsfier

# %%
gbco = GradientBoostingClassifier(random_state=1)
gbco.fit(X_train3, y_train3)

gbco_model_train_perf = model_performance_classification_sklearn(gbco, X_train3, y_train3)
print("Training performance \n", gbco_model_train_perf)
gbco_model_test_perf = model_performance_classification_sklearn(gbco, X_test3, y_test3)
print("Training performance \n", gbco_model_test_perf)
confusion_matrix_sklearn(gbco, X_test3, y_test3)


# %%
importances = gbco.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="blue", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


print(
    pd.DataFrame(
        gbco.feature_importances_, columns=["Imp"], index=X_train3.columns
    ).sort_values(by="Imp", ascending=False)
)


# %% [markdown]
# Comment: Treatment of outliers did not change much. 

# %% [markdown]
# # Gradient Boost hyperparameter tuning

# %%
# Choose the type of classifier.
gbco_tuned = GradientBoostingClassifier(
    init=AdaBoostClassifier(random_state=1), random_state=1
)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [100, 150, 200],
    "subsample": [0.8, 0.9, 1],
    "max_features": [0.7, 0.8, 0.9, 1],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(gbc_tuned, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(X_traino2, y_traino2)

# Set the clf to the best combination of parameters
gbco_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
gbco_tuned.fit(X_train3, y_train3)


# %%
# Calculating different metrics
gbco_tuned_model_train_perf = model_performance_classification_sklearn(
    gbco_tuned, X_train3, y_train3
)
print("Training performance:\n", gbco_tuned_model_train_perf)
gbco_tuned_model_test_perf = model_performance_classification_sklearn(
    gbco_tuned, X_test3, y_test3
)
print("Testing performance:\n", gbco_tuned_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(gbco_tuned, X_test3, y_test3)

# %%
importances = gbco_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        gbco_tuned.feature_importances_, columns=["Imp"], index=X_train3.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
Comment: very good model,litle overfitting,high scores. 

# %% [markdown]
# # XGBoost hyperparameter tuning 

# %%
# Choose the type of classifier.
xgbo_tuned = XGBClassifier(random_state=1, eval_metric="logloss")

# Grid of parameters to choose from
parameters = {
    "n_estimators": [10, 30, 50],
    "scale_pos_weight": [1, 2, 5],
    "subsample": [0.7, 0.9, 1],
    "learning_rate": [0.05, 0.1, 0.2],
    "colsample_bytree": [0.7, 0.9, 1],
    "colsample_bylevel": [0.5, 0.7, 1],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(xgbo_tuned, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(X_traino2, y_traino2)

# Set the clf to the best combination of parameters
xgbo_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
xgbo_tuned.fit(X_train3, y_train3)

# %%
# Calculating different metrics
xgbo_tuned_model_train_perf = model_performance_classification_sklearn(
    xgb_tuned, X_train3, y_train3
)
print("Training performance:\n", xgbo_tuned_model_train_perf)
xgbo_tuned_model_test_perf = model_performance_classification_sklearn(
    xgbo_tuned, X_test3, y_test3
)
print("Testing performance:\n", xgbo_tuned_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(xgbo_tuned, X_test3, y_test3)

# %%
importances = xgbo_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# %%
print(
    pd.DataFrame(
        xgbo_tuned.feature_importances_, columns=["Imp"], index=X_train3.columns
    ).sort_values(by="Imp", ascending=False)
)

# %%
Comment:

# %% [markdown]
# # Stacking model

# %%
estimators = [
    ("Random Forest", rfo_estimator2),
    ("Gradient Boosting", gbco_tuned),
    ("Decision Tree", dtree_estimator),
]

final_estimator = xgbo_tuned

stacking_classifiero = StackingClassifier(
    estimators=estimators, final_estimator=final_estimator
)

stacking_classifiero.fit(X_train3, y_train3)

# %%
# Calculating different metrics
stacking_classifiero_model_train_perf = model_performance_classification_sklearn(
    stacking_classifiero, X_train3, y_train3
)
print("Training performance:\n", stacking_classifiero_model_train_perf)
stacking_classifiero_model_test_perf = model_performance_classification_sklearn(
    stacking_classifiero, X_test3, y_test3
)
print("Testing performance:\n", stacking_classifiero_model_test_perf)

# Creating confusion matrix
confusion_matrix_sklearn(stacking_classifiero, X_test3, y_test3)

# %%
# training performance comparison

models_train_comp_df = pd.concat(
    [
        rfo_estimator_model_train_perf.T,
        rfo_estimator2_model_train_perf.T,
        abco_model_train_perf.T,
        abco_tuned_model_train_perf.T,
        gbco_model_train_perf.T,
        gbco_tuned_model_train_perf.T,
        xgbo_tuned_model_train_perf.T,
        stacking_classifiero_model_train_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Adaboost Classifier",
    "Adabosst Classifier Tuned",
    "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned",
    "XGBoost Classifier Tuned",
    "Stacking Classifier",
]
print("Training performance comparisonfor set with treated outliers:")
models_train_comp_df

# %% [markdown]
# The best is stacking classifier,after that XGBoost tuned model.
# 
# Effect of hyperparameter tuning:
# 
#     1) Random forest has lower scores and only minimally less overfitted
#     
#     2) Adaboost Classifier after hyperparameter tuning has allscores better except recall (minimally worse)
#     
#     4)Gradient Boost after hyperparameter tuning has better recall but other scores are worse

# %%
# training performance comparison

models_train_comp_df = pd.concat(
    [
        rfo_estimator_model_train_perf.T - rfo_estimator_model_test_perf.T,
        rfo_estimator2_model_train_perf.T - rfo_estimator2_model_test_perf.T,
        abco_model_train_perf.T - abco_model_test_perf.T,
        abco_tuned_model_train_perf.T - abco_tuned_model_test_perf.T,
        gbco_model_train_perf.T - gbco_model_test_perf.T,
        gbco_tuned_model_train_perf.T - gbco_tuned_model_test_perf.T,
        xgbo_tuned_model_train_perf.T - xgbo_tuned_model_test_perf.T,
        stacking_classifiero_model_train_perf.T
        - stacking_classifiero_model_test_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Adaboost Classifier",
    "Adabosst Classifier Tuned",
    "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned",
    "XGBoost Classifier Tuned",
    "Stacking Classifier",
]
print(
    "Training performance comparison for treated outliers (difference in scores for train and test sets):"
)
models_train_comp_df

# %% [markdown]
# ### Comment: Again stacking classifier performed best

# %% [markdown]
# ## Analysis of  features importance (in order of decreasing importance listed for each model)

# %% [markdown]
# Random forest:       prevailing_wage, no_of_employees,wage_final,yr_of_estab,education_of_employee_High School
#     
# Random forest tuned: prevailing_wage,no_of_employees,wage_final,yr_of_estab,education_of_employee_High School
#     
# Adaboost Classifier:      wage_final,prevailing_wage,no_of_employees,yr_of_estab,education_of_employee_Master
#     
# Adabosst Classifier Tuned: education_of_employee_High School,prevailing_wage, education_of_employee_Master, 
#     has_job_experience_Y,education_of_employee_Doctorate   
# 
# Gradient Boost Classifier:
#     education_of_employee_High School,has_job_experience_Y,prevailing_wage,education_of_employee_Master,
#     education_of_employee_Doctorate,continent_Europe 
#     
# Gradient Boost Classifier Tuned:
#     education_of_employee_High School,has_job_experience_Y,education_of_employee_Master,prevailing_wage,
#     education_of_employee_Doctorate,unit_of_wage_Year 
# 
# XGBoost Classifier Tuned: education_of_employee_High School,unit_of_wage_Year,continent_Europe,has_job_experience_Y,
#     education_of_employee_Master,education_of_employee_Doctorate   

# %% [markdown]
# ### Conclusions:
# 
# 1)Random forest models have the same order ofimprtance for features, with prevailing wage as most impatful.
# 
# 2) Adaboost - tuning has effect on the relative importance of the features. After hyperparameter tuning the most important is education at High School level.
# 
# 3)All the other methods has also asmost important education at High School level.  
# 
# 4) Other important fetures is job experience,Master or PhD,annual wage,application from Europe,prevailing wage, no ofemployees and year ofbusines established.
# 
# 5) Treatment of outliers has increased the score for stacking classifier f1-score from 0.83 to 0.85. For other methods the change was minimal.

# %% [markdown]
# # Actionable Insights and Recommendations

# %% [markdown]
# 1) year-of-estab is feature highly multicoliear so shouldbe considered with caution. 
# The ensamble methods are not sensitive to multicolinearity but if we want to discuss feature importance one has to be careful.
# 
# 2) Becaue many High school graduate's applications for visa are denied, it is one of the most important feature for classification. It should be undertood why that is the case. For example prevailing wage and annual wage are 2 other features that are important if visa is certified or denied. It might be that the jobs for High School graduates are paid weekly or hourly.
# 
# 3)Master and PhD degrees has lower percentage of denied visas so the clasification model has this type of educaiotn as 
# important.
# 
# 4) Visa applicants from Europe are more likely to have visa certified
# 
# 5) another factor that in some models was more important for certified visa was job at Midwest region
# 
# 6) It is important what company the applicant will work - both the number of employees and the year of estabcome out as imprtant features forseveral models

# %%


