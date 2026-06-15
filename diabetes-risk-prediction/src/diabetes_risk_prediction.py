# %% [markdown]
# # Problem Statement
# Diabetes is one of the most frequent diseases worldwide and the number of diabetic patients is growing over the years. The main cause of diabetes remains unknown, yet scientists believe that both genetic factors and environmental lifestyle play a major role in diabetes. 
# 
# Individuals with diabetes face a risk of developing some secondary health issues such as heart diseases and nerve damage. Thus, early detection and treatment of diabetes can prevent complications and assist in reducing the risk of severe health problems. 
# Even though it's incurable, it can be managed by treatment and medication.
# 
# 

# %% [markdown]
# # Objective:
# 
# To build a model to predict whether an individual is at risk of diabetes or not.

# %% [markdown]
# # Data Description:
# 
# * Pregnancies: Number of times pregnant
# * Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
# * BloodPressure: Diastolic blood pressure (mm Hg)
# * SkinThickness: Triceps skinfold thickness (mm)
# * Insulin: 2-Hour serum insulin (mu U/ml)
# * BMI: Body mass index (weight in kg/(height in m)^2)
# * Pedigree: Diabetes pedigree function - A function that scores likelihood of diabetes based on family history.
# * Age: Age in years
# * Class: Class variable (0: the person is not diabetic or 1: the person is diabetic)

# %%
# Library to suppress warnings or deprecation notes 
import warnings
warnings.filterwarnings('ignore')

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# Library to split data 
from sklearn.model_selection import train_test_split

# Libraries to import decision tree classifier and different ensemble classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Libtune to tune model, get different metric scores
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# ### Read the dataset

# %%
pima=pd.read_csv("pima-diabetes.csv")

# %%
# copying data to another varaible to avoid any changes to original data
data=pima.copy()

# %% [markdown]
# ### View the first and last 5 rows of the dataset.

# %%
data.head()

# %%
data.tail()

# %% [markdown]
# ### Understand the shape of the dataset.

# %%
data.shape

# %% [markdown]
# * There are 768 observations and 9 columns in the dataset

# %% [markdown]
# ### Check the data types of the columns for the dataset.

# %%
data.info()

# %% [markdown]
# **Observations -**
# * All variables are integer or float types
# * There are no null values in the dataset

# %% [markdown]
# ### Summary of the dataset.

# %%
data.describe().T

# %% [markdown]
# **Observations -**
# * We have data of women with an average of 4 pregnancies.
# * Variables like Glucose, BloodPressure, SkinThickness, and Insulin have minimum values of 0 which might be data input errors and we should explore it further.
# * There is a large difference between the 3rd quartile and maximum value for variables like SkinThickness, Insulin, and Age which suggest that there might be outliers present in the data.
# * The average age of women in the data is 33 years.

# %% [markdown]
# ### EDA

# %% [markdown]
# ### Univariate analysis

# %%
# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# %% [markdown]
# ### Observations on Pregnancies

# %%
histogram_boxplot(data, "Pregnancies")

# %% [markdown]
# * The distribution of the number of pregnancies is right-skewed.
# * The boxplot shows that there are few outliers to the right for this variable.
# * From the boxplot, we can see that the third quartile (Q3) is approximately equal to 6 which means 75% of women have less than 6 pregnancies and an average of 4 pregnancies.

# %% [markdown]
# ### Observations on Glucose

# %%
histogram_boxplot(data,"Glucose")

# %% [markdown]
# * The distribution of plasma glucose concentration looks like a bells-shaped curve i.e. fairly normal.
# * The boxplot shows that 0 value is an outlier for this variable - but a 0 value of Glucose concentration is not possible we should treat the 0 values as missing data.
# * From the boxplot, we can see that the third quartile (Q3) is equal to 140 which means 75% of women have less than 140 units of plasma glucose concentration.

# %% [markdown]
# ### Observations on BloodPressure

# %%
histogram_boxplot(data,"BloodPressure")

# %% [markdown]
# * The distribution for blood pressure looks fairly normal except few outliers evident from the boxplot.
# * We can see that there are some observations with 0 blood pressure - but a 0 value of blood pressure is not possible and we should treat the 0 value as missing data.
# * From the boxplot, we can see that the third quartile (Q3) is equal to 80 mmHg which means 75% of women have less than 80 mmHg of blood pressure and average blood pressure of 69 mmHg. We can say that most women have normal blood pressure.

# %% [markdown]
# ### Observations on SkinThickness

# %%
histogram_boxplot(data,"SkinThickness")

# %%
data[data['SkinThickness']>80]

# %% [markdown]
# * There is one extreme value of 99 in this variable. 
# * There are much values with 0 value of skin thickness but a 0 value of skin thickness is not possible and we should treat the 0 values as missing data.
# * From the boxplot, we can see that the third quartile (Q3) is equal to 32 mm, which means 75% of women have less than 32 mm of skin thickness and an average skin thickness of 21 mm.

# %% [markdown]
# ### Observations on Insulin

# %%
histogram_boxplot(data,"Insulin")

# %% [markdown]
# * The distribution of insulin is right-skewed.
# * There are some outliers to the right in this variable.
# * A 0 value in insulin is not possible. We should treat the 0 values as missing data.
# * From the boxplot, we can see that the third quartile (Q3) is equal to 127 mu U/ml, which means 75% of women have less than 127 mu U/ml of insulin concentration and an average of 80 mu U/ml.

# %% [markdown]
# ### Observations on BMI

# %%
histogram_boxplot(data,"BMI")

# %% [markdown]
# * The distribution of mass looks normally distributed with the mean and median of approximately 32.
# * There are some outliers in this variable.
# * A 0 value in mass is not possible we should treat the 0 values as missing data.

# %% [markdown]
# ### Observations on Pedigree

# %%
histogram_boxplot(data,"Pedigree")

# %% [markdown]
# * The distribution is skewed to the right and there are some outliers in this variable.
# * From the boxplot, we can see that the third quartile (Q3) is equal to 0.62 which means 75% of women have less than 0.62 diabetes pedigree function value and an average of 0.47.

# %% [markdown]
# ### Observations on Age

# %%
histogram_boxplot(data,"Age")

# %% [markdown]
# * The distribution of age is right-skewed.
# * There are outliers in this variable.
# * From the boxplot, we can see that the third quartile (Q3) is equal to 41 which means 75% of women have less than 41 age in our data and the average age is 33 years.

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

# %% [markdown]
# ### Observations on Class

# %%
labeled_barplot(data,"Class",perc=True)

# %% [markdown]
# * The data is slightly imbalanced as there are only ~35% of the women in data who are diabetic and ~65% of women who are not diabetic.

# %% [markdown]
# ### Observations on Preg

# %%
labeled_barplot(data,"Pregnancies",perc=True)

# %% [markdown]
# * The most common number of pregnancies amongst women is 1.
# * Surprisingly, there are many observations with more than 10 pregnancies.

# %% [markdown]
# ### Bivariate Analysis

# %%
plt.figure(figsize=(15,7))
sns.heatmap(data.corr(),annot=True,vmin=-1,vmax=1,cmap="Spectral")
plt.show()

# %% [markdown]
# **Observations-**
# * Dependent variable class shows a moderate correlation with 'Glucose'.
# * There is a positive correlation between age and the number of pregnancies which makes sense.
# * Insulin and skin thickness also shows a moderate positive correlation.

# %%
sns.pairplot(data=data,hue="Class")
plt.show()

# %% [markdown]
# * We can see that most non-diabetic persons have glucose concentration<=100 and BMI<30 
# * However, there are overlapping distributions for diabetic and non-diabetic persons. We should investigate it further.

# %%
### Function to plot boxplot
def boxplot(x):
    plt.figure(figsize=(10,7))
    sns.boxplot(data=data, x="Class",y=data[x],palette="PuBu")
    plt.show()

# %% [markdown]
# ### Class vs Pregnancies

# %%
data.columns

# %%
boxplot('Pregnancies')

# %% [markdown]
# * Diabetes is more prominent in women with more pregnancies.

# %% [markdown]
# ### Class vs Glucose

# %%
boxplot('Glucose')

# %% [markdown]
# * Women with diabetes have higher plasma glucose concentrations.

# %% [markdown]
# ### Class vs BloodPressure

# %%
boxplot('BloodPressure')

# %% [markdown]
# * There is not much difference between the blood pressure levels of a diabetic and a non-diabetic person.

# %% [markdown]
# ### Class vs SkinThickness

# %%
boxplot('SkinThickness')

# %% [markdown]
# * There's not much difference between skin thickness of diabetic and non-diabetic person.
# * There is one outlier with very high skin thickness in diabetic patients

# %% [markdown]
# ### Class vs Insulin

# %%
boxplot('Insulin')

# %% [markdown]
# * Higher levels of insulin are found in women having diabetes.

# %% [markdown]
# ### Class vs BMI

# %%
boxplot('BMI')

# %% [markdown]
# * Diabetic women are the ones with higher BMI.

# %% [markdown]
# ### Class vs Pedigree

# %%
boxplot('Pedigree')

# %% [markdown]
# * Diabetic women have a higher diabetes pedigree function values.

# %% [markdown]
# ### Class vs Age

# %%
boxplot('Age')

# %% [markdown]
# * Diabetes is more prominent in middle-aged to older aged women. However, there are some outliers in non-diabetic patients

# %% [markdown]
# ### Missing value treatment

# %%
data.loc[data.Glucose == 0, 'Glucose'] = data.Glucose.median()
data.loc[data.BloodPressure == 0, 'BloodPressure'] = data.BloodPressure.median()
data.loc[data.SkinThickness == 0, 'SkinThickness'] = data.SkinThickness.median()
data.loc[data.Insulin == 0, 'Insulin'] = data.Insulin.median()
data.loc[data.BMI == 0, 'BMI'] = data.BMI.median()

# %% [markdown]
# * 0 values replaced by the median of the respective variable

# %% [markdown]
# ## Split Data

# %%
X = data.drop('Class',axis=1)
y = data['Class'] 

# %%
# Splitting data into training and test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)
print(X_train.shape, X_test.shape)

# %% [markdown]
# **The Stratify arguments maintain the original distribution of classes in the target variable while splitting the data into train and test sets.**

# %%
y.value_counts(1)

# %%
y_test.value_counts(1)

# %% [markdown]
# ## Model evaluation 

# %% [markdown]
# **Let's define a function to provide recall scores on the train and test set and a function to show confusion matrix so that we do not have to use the same code repetitively while evaluating models.**

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

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
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
# ### Decision Tree

# %%
#Fitting the model
d_tree = DecisionTreeClassifier(random_state=1)
d_tree.fit(X_train,y_train)

#Calculating different metrics
dtree_model_train_perf=model_performance_classification_sklearn(d_tree,X_train,y_train)
print("Training performance:\n",dtree_model_train_perf)
dtree_model_test_perf=model_performance_classification_sklearn(d_tree,X_test,y_test)
print("Testing performance:\n",dtree_model_test_perf)
#Creating confusion matrix
confusion_matrix_sklearn(d_tree, X_test, y_test)

# %% [markdown]
# * The decision tree is overfitting the training data as there is a huge difference between training and test scores for all the metrics.
# * The test recall is very low i.e. only 58%.

# %% [markdown]
# ### Random Forest

# %%
#Fitting the model
rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train,y_train)

#Calculating different metrics
rf_estimator_model_train_perf=model_performance_classification_sklearn(rf_estimator,X_train,y_train)
print("Training performance:\n",rf_estimator_model_train_perf)
rf_estimator_model_test_perf=model_performance_classification_sklearn(rf_estimator,X_test,y_test)
print("Testing performance:\n",rf_estimator_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(rf_estimator, X_test, y_test)

# %% [markdown]
# * Random forest is overfitting the training data as there is a huge difference between training and test scores for all the metrics.
# * The test recall is even lower than the decision tree but has a higher test precision.

# %% [markdown]
# ### Bagging Classifier

# %%
#Fitting the model
bagging_classifier = BaggingClassifier(random_state=1)
bagging_classifier.fit(X_train,y_train)

#Calculating different metrics
bagging_classifier_model_train_perf=model_performance_classification_sklearn(bagging_classifier,X_train,y_train)
print("Training performance:\n",bagging_classifier_model_train_perf)
bagging_classifier_model_test_perf=model_performance_classification_sklearn(bagging_classifier,X_test,y_test)
print("Testing performance:\n",bagging_classifier_model_test_perf)
#Creating confusion matrix
confusion_matrix_sklearn(bagging_classifier, X_test, y_test)

# %% [markdown]
# * Bagging classifier giving a similar performance as random forest.
# * It is also overfitting the training data and lower test recall than decision trees.

# %% [markdown]
# ### Tuning Decision Tree

# %%
#Choose the type of classifier. 
dtree_estimator = DecisionTreeClassifier(class_weight={0:0.35,1:0.65},random_state=1)

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2,10), 
              'min_samples_leaf': [5, 7, 10, 15],
              'max_leaf_nodes' : [2, 3, 5, 10,15],
              'min_impurity_decrease': [0.0001,0.001,0.01,0.1]
             }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_estimator, parameters, scoring=scorer,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
dtree_estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
dtree_estimator.fit(X_train, y_train)

# %%
#Calculating different metrics
dtree_estimator_model_train_perf=model_performance_classification_sklearn(dtree_estimator,X_train,y_train)
print("Training performance:\n",dtree_estimator_model_train_perf)
dtree_estimator_model_test_perf=model_performance_classification_sklearn(dtree_estimator,X_test,y_test)
print("Testing performance:\n",dtree_estimator_model_test_perf)
#Creating confusion matrix
confusion_matrix_sklearn(dtree_estimator, X_test, y_test)

# %% [markdown]
# * The test recall has increased significantly after hyperparameter tuning and the decision tree is giving a generalized performance.
# * The confusion matrix shows that the model can identify the majority of patients who are at risk of diabetes.

# %% [markdown]
# ### Tuning Random Forest

# %%
# Choose the type of classifier. 
rf_tuned = RandomForestClassifier(class_weight={0:0.35,1:0.65},random_state=1)

parameters = {  
                'max_depth': list(np.arange(3,10,1)),
                'max_features': np.arange(0.6,1.1,0.1),
                'max_samples': np.arange(0.7,1.1,0.1),
                'min_samples_split': np.arange(2, 20, 5),
                'n_estimators': np.arange(30,160,20),
                'min_impurity_decrease': [0.0001,0.001,0.01,0.1]
}


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(rf_tuned, parameters, scoring=scorer,cv=5,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rf_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rf_tuned.fit(X_train, y_train)

# %%
#Calculating different metrics
rf_tuned_model_train_perf=model_performance_classification_sklearn(rf_tuned,X_train,y_train)
print("Training performance:\n",rf_tuned_model_train_perf)
rf_tuned_model_test_perf=model_performance_classification_sklearn(rf_tuned,X_test,y_test)
print("Testing performance:\n",rf_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(rf_tuned, X_test, y_test)

# %% [markdown]
# * The test recall has increased significantly after hyperparameter tuning but the  model is still overfitting the training data.
# * The confusion matrix shows that the model can identify the majority of patients who are at risk of diabetes.

# %% [markdown]
# ### Tuning Bagging Classifier

# %%
# Choose the type of classifier. 
bagging_estimator_tuned = BaggingClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {'max_samples': [0.7,0.8,0.9,1], 
              'max_features': [0.7,0.8,0.9,1],
              'n_estimators' : [10,20,30,40,50],
             }

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(bagging_estimator_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
bagging_estimator_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
bagging_estimator_tuned.fit(X_train, y_train)

# %%
#Calculating different metrics
bagging_estimator_tuned_model_train_perf=model_performance_classification_sklearn(bagging_estimator_tuned,X_train,y_train)
print("Training performance:\n",bagging_estimator_tuned_model_train_perf)
bagging_estimator_tuned_model_test_perf=model_performance_classification_sklearn(bagging_estimator_tuned,X_test,y_test)
print("Testing performance:\n",bagging_estimator_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(bagging_estimator_tuned, X_test, y_test)

# %% [markdown]
# * Surprisingly, the test recall has decreased after hyperparameter tuning and the  model is still overfitting the training data.
# * The confusion matrix shows that the model is not good at identifying patients who are at risk of diabetes.

# %% [markdown]
# ## Comparing all the models

# %%
# training performance comparison

models_train_comp_df = pd.concat(
    [dtree_model_train_perf.T,dtree_estimator_model_train_perf.T,rf_estimator_model_train_perf.T,rf_tuned_model_train_perf.T,
     bagging_classifier_model_train_perf.T,bagging_estimator_tuned_model_train_perf.T],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Bagging Classifier",
    "Bagging Estimator Tuned"]
print("Training performance comparison:")
models_train_comp_df

# %%
# testing performance comparison

models_test_comp_df = pd.concat(
    [dtree_model_test_perf.T,dtree_estimator_model_test_perf.T,rf_estimator_model_test_perf.T,rf_tuned_model_test_perf.T,
     bagging_classifier_model_test_perf.T, bagging_estimator_tuned_model_test_perf.T],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Bagging Classifier",
    "Bagging Estimator Tuned"]
print("Testing performance comparison:")
models_test_comp_df

# %% [markdown]
# * A tuned decision tree is the best model for our data as it has the highest test recall and giving a generalized performance as compared to other models.

# %% [markdown]
# ### Feature importance of tuned decision tree

# %%
# Text report showing the rules of a decision tree -
feature_names = list(X_train.columns)
print(tree.export_text(dtree_estimator,feature_names=feature_names,show_weights=True))

# %%
feature_names = X_train.columns
importances = dtree_estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# %% [markdown]
# * We can see that Glucose concentration is the most important feature followed by Age and BMI.
# * The tuned decision tree is using only three variables to separate the two classes. 

# %% [markdown]
# ## Conclusion
# * We can see that three variables - Glucose, Age, and BMI are the most important factors in identifying persons who are at risk of diabetes. Other variables' importance is not significant. 
# * Once the desired performance is achieved from the model, the company can use it to predict the risk factor of diabetes in new patients. This would help to reduce the cost and increase the efficiency of the process.
# * Identifying the risk of diabetes at early stages, especially among pregnant women, can help to control the disease and prevent the second health problem.
# * As per the decision tree business rules:
#     - Women's glucose level <=127 and age <=28 have a lower risk of diabetes.
#     - Women's glucose level >100 and age >28 have a higher risk of diabetes.
#     - Women's glucose level >127 and BMI <=28 have a lower risk of diabetes.
# * Based on the above analysis, we can say that:
#     - Middle-aged to older women has a higher risk of diabetes. They should keep the glucose level in check and take proper precautions.
#     - Overweight women have a higher risk of diabetes. They should keep the glucose level in check and exercise regularly. 

