# Dataset: Pima Indians Diabetes

## Source
- **File**: `pima-diabetes.csv`
- **Origin**: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- Also available on [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## Overview
| Property | Value |
|---|---|
| Rows | 768 |
| Columns | 9 (8 features + 1 target) |
| Task | Binary classification (diabetic risk) |
| Class balance | ~65% non-diabetic (0), ~35% diabetic (1) |

## Features
| Column | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hr oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| Pedigree | Diabetes pedigree function (family history score) |
| Age | Age in years |
| **Class** | **Target: 0 = not diabetic, 1 = diabetic** |

## Preprocessing Applied
- Zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing → replaced with column median
- 70/30 train-test split, stratified by Class

## Models Trained

### 1. Decision Tree (`build_decision_tree`)
- Default `DecisionTreeClassifier(random_state=1)`
- Overfits training data; low test recall (~58%)

### 2. Decision Tree Tuned (`build_decision_tree_tuned`) — **Best Model**
- `class_weight={0:0.35, 1:0.65}` to penalize missing diabetics
- GridSearchCV over max_depth, min_samples_leaf, max_leaf_nodes, min_impurity_decrease
- Scoring metric: **Recall** (minimize false negatives)
- Top features: Glucose, Age, BMI
- Gives generalized performance with significantly improved test recall

### 3. Random Forest (`build_random_forest`)
- Default `RandomForestClassifier(random_state=1)`
- Overfits; higher precision but lower recall than tuned decision tree

### 4. Random Forest Tuned (`build_random_forest_tuned`)
- `class_weight={0:0.35, 1:0.65}`, GridSearchCV over n_estimators, max_depth, max_features, max_samples
- Recall improves significantly but model still overfits

### 5. Bagging Classifier (`build_bagging`)
- Default `BaggingClassifier(random_state=1)`
- Similar performance to Random Forest

### 6. Bagging Classifier Tuned (`build_bagging_tuned`)
- GridSearchCV over n_estimators, max_samples, max_features
- Test recall decreases after tuning (counterintuitive result)

## Conclusions
- **Tuned Decision Tree** is the best model: highest test recall, generalized, interpretable
- Key risk factors: **Glucose concentration** (most important), followed by **Age** and **BMI**
- Business rule from tree: glucose ≤127 and age ≤28 → lower risk; glucose >127 and BMI ≤28 → lower risk
