# Ensemble-Tree-Models-Projects

A collection of machine learning projects using decision trees and ensemble methods (Random Forest, Bagging, Boosting) for classification problems.

## Projects

### [diabetes-risk-prediction](diabetes-risk-prediction)
Predicts whether an individual is at risk of diabetes using the Pima Indians Diabetes dataset.
- Features include pregnancies, glucose level, blood pressure, BMI, age, and more
- Builds and compares decision tree and ensemble classifiers

### [visa-application-prediction](visa-application-prediction)
EasyVisa project — predicts the outcome (certified/denied) of US employer labor certification applications.
- Exploratory data analysis on visa application data
- Decision tree and ensemble-based classification models

### [hotel-booking-cancellation](hotel-booking-cancellation)
Predicts hotel booking cancellations using the StarHotelsGroup dataset (~56,926 bookings, 18 features).
- Data cleaning and exploratory analysis
- Tree-based and ensemble models to flag likely cancellations

## Project Structure

Each project follows a standard ML project layout:

```
<project-name>/
├── data/              # Raw/processed datasets (not tracked in git)
├── notebooks/         # Original exploratory Jupyter notebook
├── src/               # Python script version of the analysis/pipeline
├── models/            # Saved/trained model artifacts (not tracked in git)
└── requirements.txt   # Python dependencies
```

The `src/*.py` files are converted from the original notebooks (in `# %%` cell-marker format, compatible with VS Code / Jupyter interactive mode).

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Decision Trees, Random Forest, Boosting/Bagging)
