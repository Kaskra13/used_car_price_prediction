# Used Car Price Prediction

A machine learning regression model that predicts used car prices using Random Forest with advanced feature engineering and hyperparameter optimization. The model achieves 77.1% R² score and reduces prediction error by 59.6% compared to baseline.

## Overview
This project analyzes a dataset of 4009 used cars to build a predictive model for vehicle pricing. Through systematic feature engineering, outlier removal, and hyperparameter tuning, the final model achieves a Mean Absolute Error (MAE) of $6,823 on previously unseen data.

### Dataset
The dataset contains 12 initial features including:

- Vehicle specifications: Brand, model, model year, mileage, engine, transmission

- Appearance: Exterior color, interior color

- Fuel type: Gasoline, Hybrid, E85 Flex Fuel, etc.

- History: Accident records, clean title status

- Target variable: Price (ranging from $2,000 to $99,000 after outlier removal)


### Key features

#### Feature engineering
- Car age calculation: Derived from model year

- Horsepower extraction: Parsed from engine descriptions using regex

- Mileage per year: Normalized usage metric

- Luxury brand indicator: Binary flag for 27 premium brands (Lexus, BMW, Tesla, etc.)

- Target encoding: K-Fold cross-validated brand encoding to prevent overfitting

- One-hot encoding: For fuel type categories

#### Data preprocessing
- Missing value imputation using training set medians

- IQR-based outlier removal (removed 6.1% of extreme values)

- Log transformation of target variable to handle skewness

- Accident history feature engineering from text descriptions


### Technologies used
```text
- Python 3.x
- pandas & numpy - Data manipulation
- scikit-learn - Machine learning models
- matplotlib & seaborn - Visualization
- joblib - Model serialization
```

### Model performance
| Model                 | MAE     | RMSE    | R²     | Improvement |
| --------------------- | ------- | ------- | ------ | ----------- |
| Baseline (Mean)       | $16,897 | $20,936 | -0.000 | -           |
| Random Forest (Tuned) | $6,823  | $10,019 | 0.771  | +59.6%      |


#### Top 3 most important features
1. Mileage (38.2%) - Total distance driven

2. Car age (27.6%) - Vehicle age in years

3. Horsepower (22.4%) - Engine power rating

These three features account for 88.1% of the model's decision-making process.


### Methodology
1. Data cleaning: Handled missing values, standardized formats, extracted numeric values from strings

2. Feature engineering: Created 10+ derived features including target-encoded brand values

3. Outlier removal: Applied IQR method to remove price outliers beyond 1.5*IQR

4. Train/test split: 80/20 split with stratified sampling (3012 training / 753 test samples)

5. Hyperparameter tuning: RandomizedSearchCV with 20 iterations, 3-fold CV

6. Model evaluation: Compared against baseline using MAE, RMSE, and R² metrics

### Installation
```bash
# Clone the repository
git clone https://github.com/Kaskra13/used_car_price_prediction
cd used-car-price-prediction

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Usage
Training the model

```python
# The notebook contains the complete pipeline
# Key steps:
# 1. Load and clean data
# 2. Engineer features
# 3. Train Random Forest with hyperparameter tuning
# 4. Evaluate and save model
```

Making predictions
```python
import joblib
import pandas as pd

# Load saved model and artifacts
model = joblib.load('best_car_price_model.pkl')
features = joblib.load('model_features.pkl')
brand_means = joblib.load('brand_means.pkl')
train_medians = joblib.load('train_medians.pkl')

# Prepare new data with same features
# Make prediction (returns log-transformed value)
prediction_log = model.predict(new_data)
predicted_price = np.expm1(prediction_log)
```

### Model details

Algorithm: Random Forest Regressor

Optimal hyperparameters:

- n_estimators: 500

- max_depth: None (fully grown trees)

- min_samples_split: 5

- min_samples_leaf: 1

- max_features: None

Cross-validation: 3-fold CV with MAE scoring

Target transformation: log1p (log transformation) to normalize price distribution


### Project structure
```text
├── used_cars.csv                            # Raw dataset
└── used_car_price_prediction.ipynb          # Complete analysis notebook
```
