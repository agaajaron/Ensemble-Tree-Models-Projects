# Dataset: Hotel Booking Cancellation (Star Hotels Group)

## Source
- **File**: `StarHotelsGroup.csv`
- **Context**: Booking records from a group of star-rated hotels

## Overview
| Property | Value |
|---|---|
| Raw rows | 56,926 |
| After deduplication | 42,576 |
| Columns | 18 |
| Task | Binary classification (booking canceled?) |
| Target balance | ~Balanced (see EDA) |

## Features
| Column | Description |
|---|---|
| lead_time | Days between booking and arrival |
| no_of_weekend_nights | Weekend nights booked |
| no_of_week_nights | Weekday nights booked |
| type_of_meal_plan | Meal plan selected (categorical) |
| required_car_parking_space | 0/1 |
| room_type_reserved | Reserved room type (categorical) |
| arrival_year | Year of arrival |
| arrival_month | Month of arrival (1–12) |
| arrival_date | Day of month |
| market_segment_type | Booking channel (categorical) |
| repeated_guest | 0 = first time, 1 = repeat |
| no_of_previous_cancellations | Prior cancellations |
| no_of_previous_bookings_not_canceled | Prior completed bookings |
| avg_price_per_room | Average nightly rate (EUR) |
| no_of_special_requests | Number of special requests |
| no_of_adults | Adults in booking |
| no_of_children | Children in booking |
| **booking_status** | **Target: Canceled / Not_Canceled** |

## Preprocessing Applied
- Duplicates removed (56,926 → 42,576 rows)
- Target re-encoded: `Not_Canceled=1`, `Canceled=0`
- `pd.get_dummies` on: type_of_meal_plan, room_type_reserved, market_segment_type
- VIF analysis to detect multicollinear features (market_segment columns had high VIF)
- 70/30 train-test split (no stratify)
- statsmodels models require `sm.add_constant(X)` for intercept

## Key EDA Findings
- **lead_time > 150 days → ~75% cancellation rate** (root node of best decision tree)
- **Special requests**: not-canceled bookings have median=1, canceled have median=0
- Online segment has highest cancellation rate; offline and corporate lower

## Models Trained

### 1. Logistic Regression — statsmodels Full (`build_logit_full`)
- `sm.Logit(y_train, X_train_with_const).fit()`
- Baseline; some features with high p-values

### 2. Logistic Regression — VIF Pruned (`build_logit_vif_pruned`)
- Removed features with VIF > 5 (market_segment multicollinearity)
- Slight improvement in coefficient interpretability

### 3. Logistic Regression — P-value Pruned (`build_logit_pvalue_pruned`)
- Iteratively removes features with p-value > 0.05
- Final model keeps only statistically significant predictors

### 4. Logistic Regression — Regularized (`build_logit_regularized`)
- `fit_regularized(alpha=1, L1_wt=1)` (Lasso)
- Sparse coefficients

### 5. Logistic Regression — sklearn Full (`build_logreg_sklearn`)
- `LogisticRegression(max_iter=1000)`
- Similar accuracy to statsmodels version

### 6. Logistic Regression — sklearn VIF Pruned (`build_logreg_sklearn_vif_pruned`)
- Same VIF-pruned feature set as statsmodels version

### 7. Decision Tree Default (`build_decision_tree_default`)
- Fully grown tree, overfits heavily
- 100% train accuracy, much lower test

### 8. Decision Tree Depth=7 (`build_decision_tree_depth7`) — **Best Model**
- `max_depth=7` — best balance of F1 and generalization
- Root node: lead_time; second level: market_segment_type, no_of_special_requests
- No significant overfitting

### 9. Decision Tree Depth=6 (`build_decision_tree_depth6`)
- Slightly less expressive than depth=7

### 10. Decision Tree Balanced (`build_decision_tree_balanced`)
- `class_weight="balanced"` — improves recall but hurts precision

### 11. Decision Tree Tuned — GridSearchCV (`build_decision_tree_tuned`)
- Scorer: F1
- Searches max_depth (2–9), min_samples_leaf, max_leaf_nodes, min_impurity_decrease

### 12. Decision Tree Cost Complexity Pruned (`build_decision_tree_pruned`)
- Explores full `ccp_alpha` path; selects alpha maximizing test F1

## Conclusions
- **Decision Tree (max_depth=7)** is the best model — interpretable, no overfitting, highest F1
- **Key predictor**: `lead_time` — the longer in advance, the higher the cancellation risk
- Logistic regression achieves reasonable AUC but lower F1 vs Decision Tree
- Actionable insight: hotels should flag bookings >150 days out for proactive outreach
