# Student Performance Analysis

**Author:** Miguel Estrada

## Project Overview

This project explores a student performance dataset using Python and machine learning. The notebook performs exploratory data analysis (EDA), visualizes relationships between student-related factors and exam scores, trains an XGBoost regression model, and evaluates model performance.

The prediction target is **`Exam_Score`**, a numerical exam score. The model treats this as a regression problem rather than a classification problem.

## Dataset

The project uses `StudentPerformanceFactors.csv`, obtained from Kaggle.

The dataset contains 6,607 rows and 20 columns, including the target `Exam_Score` and 19 potential input variables.

The features include:

- `Hours_Studied`
- `Attendance`
- `Parental_Involvement`
- `Access_to_Resources`
- `Extracurricular_Activities`
- `Sleep_Hours`
- `Previous_Scores`
- `Motivation_Level`
- `Internet_Access`
- `Tutoring_Sessions`
- `Family_Income`
- `Teacher_Quality`
- `School_Type`
- `Peer_Influence`
- `Physical_Activity`
- `Learning_Disabilities`
- `Parental_Education_Level`
- `Distance_from_Home`
- `Gender`

The target variable is:

- `Exam_Score` — the student's exam score.

## Objectives

- Inspect and clean the dataset.
- Explore the distribution of exam scores.
- Examine correlations among numerical variables.
- Visualize relationships between selected student factors and exam scores.
- Train an XGBoost regression model.
- Evaluate predictions on a held-out test set.
- Estimate feature importance using multiple approaches.
- Compare feature-importance rankings and discuss their limitations.

## Exploratory Data Analysis

The notebook includes visualizations such as:

- Exam score distribution with mean and median markers
- Correlation heatmap for numerical columns
- Attendance versus exam score, grouped by internet access
- Exam scores by parental involvement and motivation level
- Study-hours distributions by school type and gender
- Mean exam scores across tutoring-session counts and family-income groups
- Student counts by parental education and peer influence

These visualizations support descriptive analysis. They can reveal patterns in this dataset, but they do not demonstrate that a factor causes exam scores to change.

## Data Preparation

The notebook:

1. Loads the CSV file using Pandas.
2. Checks missing values and removes rows containing missing data.
3. Separates `Exam_Score` from the predictor variables.
4. Identifies categorical and numerical columns.
5. Splits the data into training and testing sets using an 80/20 split and `random_state=42`.
6. Applies One-Hot Encoding to categorical variables.

The One-Hot Encoder uses `handle_unknown="ignore"`, so categories encountered in the test data that were not present during training can be handled without raising an error.

The preprocessing and model are combined in a Scikit-learn Pipeline.

## Machine Learning Model

The notebook uses **XGBoost Regression** (`XGBRegressor`) to predict exam scores.

The model configuration includes:

- Objective: `reg:squarederror`
- Learning rate: `0.05`
- Maximum tree depth: `4`
- Number of estimators: `500`
- Subsample: `0.8`
- Column sampling: `0.8`
- Random state: `42`

XGBoost is a gradient-boosted tree model that learns patterns from the training data and produces numerical predictions.

## Model Evaluation

The notebook evaluates the model on the test set using:

| Metric | Meaning |
|---|---|
| RMSE | Typical prediction error magnitude, with larger errors penalized more strongly |
| MSE | Average squared prediction error |
| MAE | Average absolute prediction error |
| R² | Proportion of target variance explained by the model relative to a mean-baseline model |

Lower RMSE, MSE, and MAE indicate better prediction accuracy. R² closer to 1 indicates stronger performance against the baseline; it can be negative if the model performs worse than predicting the test-set mean.

The actual scores are generated when the notebook is executed and depend on the dataset and environment.

## Feature Importance

The notebook uses two feature-importance approaches.

### 1. Permutation Importance

Permutation importance measures how much the model's test-set performance deteriorates when the values in one original feature column are shuffled.

The notebook uses negative RMSE as the scoring function and repeats the shuffle 20 times.

A larger positive mean importance indicates that shuffling that feature increased the model's prediction error more substantially. The standard deviation across repeats provides an indication of variability.

### 2. XGBoost Feature Importance

XGBoost provides an internal importance score for the encoded features used by the model.

Because One-Hot Encoding expands categorical variables into multiple columns, the notebook aggregates encoded importance values back to their original feature columns.

### Comparing the Methods

The notebook compares permutation importance and aggregated XGBoost importance.

- Permutation importance reflects the effect of a feature on held-out model performance.
- XGBoost importance reflects how the fitted trees used features internally.
- Different rankings are possible and should not be treated as a contradiction; the methods measure different things.

Feature importance is model- and dataset-specific. Correlated variables can share or mask importance, and importance does not imply causation.

## Project Structure

```text
StudentPerformance/
├── main.ipynb
├── StudentPerformanceFactors.csv
├── README.md
└── requirements.txt
```

## Requirements

The notebook uses:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

Install the dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## How to Run

1. Clone or download the repository.
2. Ensure `StudentPerformanceFactors.csv` is in the same directory as the notebook.
3. Install the dependencies.
4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open `main.ipynb` and run the cells from top to bottom.

## Notes and Potential Improvements

- The notebook currently removes rows with missing values. Consider reporting how many rows are removed and comparing this approach with imputation.
- The train/test split is reproducible but does not explicitly set a fixed proportion for any particular student subgroup.
- Consider cross-validation and hyperparameter tuning to estimate generalization more robustly.
- Compare XGBoost with a simple baseline such as predicting the training-set mean.
- Consider reporting a residual plot and checking whether errors vary across the range of exam scores.
- Ensure notebook descriptions consistently refer to student exam scores rather than unrelated targets such as insurance charges.

## Conclusion

This project demonstrates a regression workflow for analyzing student performance: data inspection, visualization, preprocessing, model training, evaluation, and feature-importance analysis.

The results can help identify which variables are useful for predicting exam scores in this dataset. They should be interpreted as predictive associations rather than causal explanations or definitive statements about individual students.

---

**Author:** Miguel Estrada  
GitHub: [miguelestradam3](https://github.com/miguelestradam3)
