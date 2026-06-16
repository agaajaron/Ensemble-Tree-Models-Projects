# Dataset: EasyVisa — US Visa Application Prediction

## Source
- **File**: `EasyVisa.csv`
- **Origin**: Office of Foreign Labor Certification (OFLC), U.S. Department of Labor
- Also available on [Kaggle](https://www.kaggle.com/datasets/moro23/easyvisa-dataset)

## Overview
| Property | Value |
|---|---|
| Raw rows | ~25,000 |
| After filtering (no_of_employees > 0) | ~25,000 |
| Columns | 12 (after dropping case_id) |
| Task | Binary classification (visa Certified?) |
| Target balance | Slightly imbalanced (~60% Certified) |

## Features
| Column | Description |
|---|---|
| continent | Applicant continent of origin |
| education_of_employee | Highest education level |
| has_job_experience | Y/N |
| requires_job_training | Y/N |
| no_of_employees | Employer company size |
| yr_of_estab | Year employer was established |
| region_of_employment | US region where job is located |
| prevailing_wage | Prevailing wage for the role |
| unit_of_wage | Hour / Week / Month / Year |
| full_time_position | Y/N |
| **case_status** | **Target: Certified=1, Denied=0** |

## Engineered Features
| Column | Description |
|---|---|
| wage_fin | Annualized wage: prevailing_wage × unit multiplier (Hour×2080, Week×52, Month×12, Year×1) |

## Preprocessing Applied
- Removed rows where `no_of_employees ≤ 0`
- `wage_fin` created to normalize wages to annual basis
- Target: `case_status` Certified=1, Denied=0
- `pd.get_dummies` on 7 categorical columns
- 70/30 stratified train-test split
- **Outlier treatment**: IQR clipping on no_of_employees, prevailing_wage, wage_fin → separate dataset for comparison
- **50% subsample** of training data used for faster GridSearchCV

## Models Trained

### Decision Trees
| Model | Notes |
|---|---|
| `build_dt_weighted` | class_weight={0:0.67, 1:0.33} |
| `build_dt_balanced` | class_weight="balanced" |
| `build_dt_tuned_wide` | GridSearchCV max_depth 2–30 |
| `build_dt_tuned_constrained` | GridSearchCV max_depth 5–7, stricter params |

### Random Forest
| Model | Notes |
|---|---|
| `build_rf` | Default; overfits |
| `build_rf_tuned` | GridSearchCV over n_estimators, max_depth, max_features, max_samples |
| `build_rf` (outlier) | Same on outlier-treated data |
| `build_rf_tuned` (outlier) | Tuned on outlier-treated data |

### Bagging
| Model | Notes |
|---|---|
| `build_bagging` | Default BaggingClassifier |
| `build_bagging_tuned` | GridSearchCV with DTree base_estimator |

### Boosting
| Model | Notes |
|---|---|
| `build_adaboost` | **No overfitting out of the box — high test scores** |
| `build_adaboost_tuned` | GridSearchCV, DTree base max_depth 1–3 |
| `build_gbm` | GradientBoostingClassifier default |
| `build_gbm_tuned` | GridSearchCV over n_estimators, learning_rate, max_depth, subsample |
| `build_xgb` | XGBClassifier(eval_metric='logloss') |
| `build_xgb_tuned` | GridSearchCV — **best individual model** |

All boosting models also trained on outlier-treated dataset.

### Stacking (Best Overall)
```python
estimators = [
    ("Random Forest", rfo_tuned),
    ("Gradient Boosting", gbco_tuned),
    ("Decision Tree", dt_tuned_constrained),
]
final_estimator = XGBClassifier(eval_metric='logloss')
StackingClassifier(estimators=estimators, final_estimator=final_estimator)
```
- Trained on both regular and outlier-treated datasets

## Conclusions
- **Stacking Classifier** (RF + GBM + DTree → XGB) achieves highest F1 and best generalization
- **XGBoost Tuned** is the best individual model
- **AdaBoost** is notable: no overfitting even without tuning
- Outlier treatment shows modest improvement for wage-based features

## Top Predictive Features
1. `education_of_employee_High School` (higher edu → higher certification)
2. `has_job_experience_Y` (experience significantly boosts approval)
3. `education_of_employee_Master`
4. `continent_Europe`
5. `prevailing_wage` (higher wage → higher certification probability)

## Business Insight
- Applicants with Master's/PhD degree and prior job experience from Europe have the highest approval rates
- Full-time positions in the Northeast and South regions have higher certification rates
