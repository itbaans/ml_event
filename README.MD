# Machine Learning Module Guide - Probattle
## Binary Classification Workflow

### Introduction

AssalamaAalaikum and welcome to the Machine Learning Module of Probattle. This guide provides a structured approach to creating efficient ML workflows for binary classification tasks. While these guidelines offer best practices and suggestions, teams are encouraged to adapt or develop their own approaches based on their expertise and requirements.

### Environment Setup

We recommend using Kaggle notebooks for training due to their advantages:
- Free GPU/TPU access
- Ability to run up to 5 notebooks simultaneously
- Pre-installed ML libraries
- Collaborative features

### Detailed Workflow

#### 1. Library Management

Let's start with a clean, organized way to import all necessary libraries:

```python
# Data manipulation and analysis
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score
from sklearn.impute import SimpleImputer


# Specific models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
```

#### 2. Data Loading and Initial Processing

Here's a structured approach to load and prepare your data:

```python
def load_and_prepare_data(filepath, sample_size=None):
    """
    Load data with optional sampling for quick iterations
    
    Parameters:
    filepath: str - Path to the dataset
    sample_size: int or None - Number of samples to load (None for full dataset)
    
    Returns:
    X, y - Features and target variables
    """
    # Read the data
    df = pd.read_csv(filepath)
    
    # Optional sampling for quick iterations
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y

# Usage example
X, y = load_and_prepare_data('your_train_set_data_path')
X = X.drop('row_id', axis=1)

# For quick iterations with sample
sX, sy = load_and_prepare_data('your_train_set_data_path', sample_size=1000)
sX = sX.drop('row_id', axis=1)
```

#### 3. Model Configuration

Create a centralized model configuration setup:

```python
def initialize_models():
    """
    Initialize all models with their hyperparameters
    Returns dictionary of model instances
    """
    models = {
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'logistic': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    return models
```

#### 4. Feature Processing Pipeline

Here's how to create a modular feature processing pipeline:

```python
# Custom transformer for feature selection based on percentage NOTE: THIS IS JUST A EXAMPLE CLASS, YOU CAN MAKE YOUR OWN TAYLOARED TO YOUR OWN NEEDS. WE RECOMMEND TO USE AI ASSISTANT TO CREATE THIS
class PercentageFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer that selects a percentage of features based on importance
    """
    def __init__(self, feature_percentage=0.8):
        self.feature_percentage = feature_percentage
        self.selected_features = None
    
    def fit(self, X, y=None):
        # Calculate feature importance using correlation with target
        if isinstance(X, pd.DataFrame):
            correlations = abs(X.corrwith(pd.Series(y)))
            n_features = int(len(correlations) * self.feature_percentage)
            self.selected_features = correlations.nlargest(n_features).index
        return self
    
    def transform(self, X):
        return X[self.selected_features] if isinstance(X, pd.DataFrame) else X

# Create the main feature processing pipeline
def create_feature_pipeline():
    """
    Creates a complete feature processing pipeline
    """
    feature_pipeline = Pipeline([
        ('feature_selector', PercentageFeatureSelector(feature_percentage=0.8)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
        # Add more transformers as needed
    ])
    return feature_pipeline
```

#### 5. Training and Evaluation Framework

Here's a complete framework to train and evaluate models:

```python
#Training, this is using k-fold cross validation as it is best for providing the model performance. If the data is large you can use simple training too, change it as you need.
def train_and_evaluate(X, y, models, k=5, model_name='xgboost'):
    """
    Train and evaluate multiple models using K-Fold cross-validation and the feature pipeline.
    
    Parameters:
    X: Features DataFrame
    y: Target Series
    feature_pipeline: Sklearn Pipeline object
    models: Dictionary of model instances
    k: Number of folds for cross-validation (default=5)
    
    Returns:
    Dictionary of results
    """
    # Converting to DataFrame to use iloc
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    results = {name: {'accuracy': [], 'auc_roc': []} for name in models.keys()}
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_df.iloc[train_idx], y_df.iloc[test_idx]
        
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Store metrics
            results[name]['accuracy'].append(accuracy_score(y_test, y_pred))
            results[name]['auc_roc'].append(roc_auc_score(y_test, y_proba))
    
    # Aggregate results
    final_results = {
        name: {
            'accuracy_mean': np.mean(metrics['accuracy']),
            'accuracy_std': np.std(metrics['accuracy']),
            'auc_roc_mean': np.mean(metrics['auc_roc']),
            'auc_roc_std': np.std(metrics['auc_roc']),
        }
        for name, metrics in results.items()
    }
    
    return final_results, results[model_name]['auc_roc']
```

### Best Practices and Tips

1. Data Validation: Always validate your data before processing:
   - Check for missing values
   - Verify data types
   - Look for outliers
   - Examine class distribution

2. Feature Engineering:
   - Start with simple features
   - Document all transformations
   - Test feature importance
   - Use cross-validation for validation

3. Model Development:
   - Begin with simple models
   - Use cross-validation for model selection
   - Monitor for overfitting
   - Keep track of experiments

4. Code Organization:
   - Maintain modular code
   - Use consistent naming conventions
   - Document functions and classes
   - Create reusable components

This guide provides a foundation for approaching binary classification tasks. Teams can modify and extend these components based on their specific needs and the competition requirements.

We have provided a sample notebook to get you started. Use this notebook in a dummy-competition on this link to create submissions and play around with it.

We also adviced you to create visualizing workflows for accuracies, feature_importances etc, this will help you create better reports at the end for your overall submission.

Thanks!

and GOOD LUCKS!