# Titanic Survival Prediction

## Project Overview

This project is a machine learning solution for the Kaggle **Titanic: Machine Learning from Disaster** competition.

The objective is to predict whether a passenger survived the Titanic disaster using passenger information such as class, age, sex, family relationships, fare, and port of embarkation.

The final submission achieved a **Kaggle prediction score of 0.7878**.

## Kaggle Result

**Public prediction score: 0.7878**

The notebook also achieved approximately:

```text
Holdout Accuracy: 0.7933
```

on the internal stratified test split.

## Dataset

The project uses the standard Titanic competition files:

```text
train.csv
test.csv
gender_submission.csv
```

Dataset sizes:

```text
train.csv: 891 passengers
test.csv: 418 passengers
```

The training dataset contains the following columns:

```text
PassengerId
Survived
Pclass
Name
Sex
Age
SibSp
Parch
Ticket
Fare
Cabin
Embarked
```

`Survived` is the target variable:

```text
0 = Did not survive
1 = Survived
```

## Exploratory Data Analysis

The notebook begins with basic statistical exploration of the training dataset using pandas.

```python
titanic_data.describe()
```

A correlation heatmap is also created using seaborn:

```python
sns.heatmap(
    titanic_data.select_dtypes(include=[np.number]).corr(),
    cmap="YlGnBu"
)
```

This provides an initial overview of relationships between the numerical variables and survival.

## Train and Validation Split

Instead of using a completely random split, the project uses `StratifiedShuffleSplit`.

The split preserves the distribution of:

```text
Survived
Pclass
Sex
```

The dataset is divided into:

```text
Training set: 712 passengers
Validation set: 179 passengers
```

with a validation size of 20%.

## Data Preprocessing

A custom scikit-learn preprocessing pipeline was created using custom transformers.

### Age Imputation

The `Age` column contains missing values.

Missing ages are replaced with the mean age using:

```python
SimpleImputer(strategy="mean")
```

This logic is implemented inside a custom `AgeImputer` transformer.

### Categorical Encoding

Categorical variables are converted into numerical features using `OneHotEncoder`.

The project encodes:

```text
Embarked
Sex
```

This produces indicator columns for embarkation categories and passenger sex.

### Feature Removal

Columns that are not used directly by the model are removed:

```text
Embarked
Name
Ticket
Cabin
Sex
```

After preprocessing, the model works with numerical features including:

```text
PassengerId
Pclass
Age
SibSp
Parch
Fare
Embarked indicators
Sex indicators
```

### Feature Scaling

The numerical feature matrix is standardized using:

```python
StandardScaler()
```

## Preprocessing Pipeline

The preprocessing stages are combined into a scikit-learn `Pipeline`:

```python
pipeline = Pipeline([
    ("age_imputer", AgeImputer()),
    ("feature_encoder", FeatureEncoder()),
    ("feature_dropper", FeatureDropper())
])
```

This keeps the preprocessing workflow organized and reusable for the training and competition test datasets.

## Model

The project uses a:

```text
Random Forest Classifier
```

from scikit-learn.

Random Forest was chosen as the main classification algorithm for predicting passenger survival.

## Hyperparameter Tuning

The model is tuned using `GridSearchCV`.

The following hyperparameters are tested:

```python
param_grid = [
    {
        "n_estimators": [10, 100, 200, 500],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 3, 4]
    }
]
```

The search uses:

```text
3-fold cross-validation
Accuracy scoring
```

An initial tuning run selected:

```python
RandomForestClassifier(
    max_depth=10,
    n_estimators=500
)
```

## Validation Performance

After preprocessing and training, the selected model was evaluated on the held-out stratified validation set.

The notebook produced:

```text
Accuracy: 0.7932960893854749
```

or approximately:

```text
79.33%
```

## Final Training

After validation, the complete training dataset is processed again and used to train the production model.

The workflow is:

```text
Full Training Dataset
        ↓
Preprocessing Pipeline
        ↓
Feature Scaling
        ↓
GridSearchCV
        ↓
Best Random Forest Model
        ↓
Titanic Test Dataset
        ↓
Predictions
```

## Kaggle Submission

Predictions are generated for all 418 passengers in `test.csv`.

The final submission contains only:

```text
PassengerId
Survived
```

Example:

```text
PassengerId,Survived
892,0
893,0
894,0
895,0
896,1
```

The predictions are exported with:

```python
final_df.to_csv(
    "predictions.csv",
    index=False
)
```

The generated submission contains:

```text
418 predictions
```

with:

```text
271 predicted non-survivors
147 predicted survivors
```

## Project Structure

```text
titanic-survival-prediction/
│
├── titan.ipynb
├── train.csv
├── test.csv
├── gender_submission.csv
├── predictions.csv
└── README.md
```

## Technologies Used

```text
Python
NumPy
pandas
Matplotlib
Seaborn
scikit-learn
Jupyter Notebook
```

## Machine Learning Concepts Demonstrated

This project demonstrates practical experience with:

1. Binary classification
2. Exploratory data analysis
3. Missing-value handling
4. Categorical feature encoding
5. Custom scikit-learn transformers
6. Machine learning pipelines
7. Stratified train/test splitting
8. Feature scaling
9. Random Forest classification
10. Hyperparameter tuning
11. Grid search
12. Cross-validation
13. Model evaluation
14. Training on the complete dataset
15. Generating Kaggle competition submissions

## How to Run the Project

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Place the following files in the same directory:

```text
titan.ipynb
train.csv
test.csv
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
titan.ipynb
```

and run the cells in order.

The final cells create:

```text
predictions.csv
```

which can be uploaded directly to Kaggle.

## Possible Improvements

There are several ways the project could be improved further:

1. Fit preprocessing steps only on the training data and reuse them when transforming validation and test data.
2. Fit `StandardScaler` on the training set and use the same scaler for validation and Kaggle test data.
3. Remove `PassengerId` from the predictive features.
4. Engineer passenger titles from the `Name` column.
5. Extract family size from `SibSp` and `Parch`.
6. Create an `IsAlone` feature.
7. Extract deck information from `Cabin`.
8. Improve missing-value handling for `Fare` and `Embarked`.
9. Compare Random Forest with Logistic Regression, Gradient Boosting, XGBoost, or other classifiers.
10. Add confusion matrix, precision, recall, and F1-score evaluation.
11. Use reproducible `random_state` values.
12. Use a single fitted preprocessing pipeline for the complete modeling workflow.

## Result

The project successfully produced a valid Titanic Kaggle submission and achieved:

```text
Kaggle Score: 0.7878
```

The project demonstrates an end-to-end machine learning workflow from raw passenger data through preprocessing, model selection, hyperparameter tuning, validation, prediction, and competition submission.
