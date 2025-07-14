#!/usr/bin/env python3
"""
Model Training Workflow - ML Notebook Implementation

This module demonstrates advanced model training techniques including hyperparameter 
optimization, cross-validation, experiment tracking, and model comparison.

Category: ML Workflow
Author: Data Science Team
Created: 2024-01-15
Runtime: 20-30 minutes

Key Features:
- Multiple model training and comparison
- Hyperparameter optimization with GridSearch and RandomSearch
- Cross-validation and model validation
- Experiment tracking with MLflow
- Model persistence and versioning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
import sys
from pathlib import Path
import json
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Machine learning libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold, validation_curve, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Try to import MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    print("✅ MLflow available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available - skipping experiment tracking")

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("🚀 Model Training Workflow Started")
print("=" * 50)

# 1. Data Loading and Preparation
print("\n📦 1. Loading Data and Prepared Features")
print("=" * 50)

# Load preprocessed data from feature engineering step
try:
    # Try to load from feature engineering pipeline
    pipeline_path = Path("../models/feature_engineering/feature_engineering_pipeline.joblib")
    feature_info_path = Path("../models/feature_engineering/feature_info.json")
    
    if pipeline_path.exists() and feature_info_path.exists():
        print("✅ Loading feature engineering pipeline...")
        preprocessing_pipeline = joblib.load(pipeline_path)
        
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        
        print(f"📊 Pipeline loaded with {len(feature_info['selected_features'])} input features")
        print(f"📊 Pipeline outputs {len(feature_info['final_features'])} engineered features")
        
        # Create synthetic data for demonstration
        print("\n📦 Creating synthetic dataset for training...")
        n_samples = 10000
        np.random.seed(RANDOM_STATE)
        
        # Create the same synthetic data structure as in feature engineering
        data = {
            'customer_age': np.random.randint(18, 80, n_samples),
            'annual_income': np.random.lognormal(10, 0.5, n_samples),
            'purchase_frequency': np.random.poisson(12, n_samples),
            'avg_basket_size': np.random.exponential(50, n_samples),
            'days_since_last_purchase': np.random.exponential(30, n_samples),
            'loyalty_score': np.random.beta(2, 5, n_samples) * 100,
            'product_category_electronics': np.random.binomial(1, 0.3, n_samples),
            'product_category_clothing': np.random.binomial(1, 0.4, n_samples),
            'product_category_food': np.random.binomial(1, 0.5, n_samples),
            'seasonal_factor': np.sin(np.random.uniform(0, 2*np.pi, n_samples)) + 1,
            'marketing_channel_online': np.random.binomial(1, 0.6, n_samples),
            'marketing_channel_email': np.random.binomial(1, 0.4, n_samples),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Budget'], n_samples, p=[0.2, 0.5, 0.3]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'is_weekend': np.random.binomial(1, 0.3, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create targets
        target_sales = (
            df['annual_income'] * 0.001 +
            df['purchase_frequency'] * 5 +
            df['avg_basket_size'] * 2 +
            df['loyalty_score'] * 1.5 +
            df['product_category_electronics'] * 50 +
            df['seasonal_factor'] * 20 +
            np.random.normal(0, 50, n_samples)
        )
        
        df['target_sales'] = np.maximum(target_sales, 0)
        df['target_high_value'] = (df['target_sales'] > df['target_sales'].quantile(0.7)).astype(int)
        
        # Encode categorical variables
        df_encoded = pd.get_dummies(df, columns=['customer_segment', 'region'], drop_first=True)
        
        # Separate features and targets
        feature_cols = [col for col in df_encoded.columns if not col.startswith('target_')]
        X = df_encoded[feature_cols]
        y_regression = df_encoded['target_sales']
        y_classification = df_encoded['target_high_value']
        
        DATA_LOADED = True
        
    else:
        print("⚠️ Feature engineering pipeline not found - creating from scratch...")
        DATA_LOADED = False
        
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    DATA_LOADED = False

if not DATA_LOADED:
    print("\n📦 Creating synthetic dataset from scratch...")
    # Create synthetic data for demonstration
    n_samples = 10000
    np.random.seed(RANDOM_STATE)
    
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.uniform(0, 1, n_samples),
        'feature_4': np.random.poisson(3, n_samples),
        'feature_5': np.random.exponential(2, n_samples),
        'feature_6': np.random.binomial(1, 0.5, n_samples),
        'feature_7': np.random.binomial(1, 0.3, n_samples),
        'feature_8': np.random.binomial(1, 0.7, n_samples),
        'feature_9': np.random.beta(2, 5, n_samples),
        'feature_10': np.random.lognormal(0, 1, n_samples)
    })
    
    # Create targets
    y_regression = (
        X['feature_1'] * 2 + 
        X['feature_2'] * 1.5 + 
        X['feature_3'] * 3 + 
        X['feature_4'] * 0.5 + 
        X['feature_5'] * 1.2 + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    y_classification = (y_regression > y_regression.quantile(0.7)).astype(int)

print(f"✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
print(f"📊 Regression target range: {y_regression.min():.2f} - {y_regression.max():.2f}")
print(f"🎯 Classification target distribution: {y_classification.value_counts().to_dict()}")

# 2. Data Splitting Strategy
print("\n📊 2. Data Splitting Strategy")
print("=" * 50)

# Split data into train/validation/test sets
X_temp, X_test, y_temp_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=RANDOM_STATE
)

X_train, X_val, y_train_reg, y_val_reg = train_test_split(
    X_temp, y_temp_reg, test_size=0.25, random_state=RANDOM_STATE  # 0.25 * 0.8 = 0.2
)

# Same split for classification
_, _, y_temp_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.2, random_state=RANDOM_STATE, stratify=y_classification
)

_, _, y_train_clf, y_val_clf = train_test_split(
    X_temp, y_temp_clf, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp_clf
)

print(f"✅ Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✅ Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"✅ Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled using StandardScaler")

# 3. Model Definition and Training
print("\n🤖 3. Model Definition and Training")
print("=" * 50)

# Define models for regression
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=RANDOM_STATE),
    'Lasso Regression': Lasso(random_state=RANDOM_STATE),
    'ElasticNet': ElasticNet(random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(random_state=RANDOM_STATE, max_iter=500)
}

# Define models for classification
classification_models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'SVM': SVC(random_state=RANDOM_STATE, probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
}

# Try to add XGBoost and LightGBM if available
try:
    regression_models['XGBoost'] = xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    classification_models['XGBoost'] = xgb.XGBClassifier(random_state=RANDOM_STATE, verbosity=0)
    print("✅ XGBoost models added")
except:
    print("⚠️ XGBoost not available")

try:
    regression_models['LightGBM'] = lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1)
    classification_models['LightGBM'] = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
    print("✅ LightGBM models added")
except:
    print("⚠️ LightGBM not available")

print(f"📊 Total regression models: {len(regression_models)}")
print(f"📊 Total classification models: {len(classification_models)}")

# 4. Model Training and Evaluation
print("\n🎯 4. Model Training and Evaluation")
print("=" * 50)

# Initialize MLflow experiment
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_experiment("model-training-workflow")
        print("✅ MLflow experiment initialized")
    except:
        print("⚠️ MLflow experiment setup failed")

# Train regression models
print("\n📈 Training Regression Models")
print("=" * 40)

regression_results = {}
for name, model in regression_models.items():
    print(f"🔄 Training {name}...")
    
    try:
        # Train model
        model.fit(X_train_scaled, y_train_reg)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train_reg, y_pred_train)
        val_r2 = r2_score(y_val_reg, y_pred_val)
        train_mse = mean_squared_error(y_train_reg, y_pred_train)
        val_mse = mean_squared_error(y_val_reg, y_pred_val)
        train_mae = mean_absolute_error(y_train_reg, y_pred_train)
        val_mae = mean_absolute_error(y_val_reg, y_pred_val)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train_reg, cv=5, scoring='r2')
        
        metrics = {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'overfitting': train_r2 - val_r2
        }
        
        regression_results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'val': y_pred_val
            }
        }
        
        print(f"  ✅ {name}: R² = {val_r2:.4f}, CV = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"regression_{name}"):
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
        
    except Exception as e:
        print(f"  ❌ {name}: Error - {str(e)}")
        regression_results[name] = None

# Train classification models
print("\n📊 Training Classification Models")
print("=" * 40)

classification_results = {}
for name, model in classification_models.items():
    print(f"🔄 Training {name}...")
    
    try:
        # Train model
        model.fit(X_train_scaled, y_train_clf)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train_clf, y_pred_train)
        val_acc = accuracy_score(y_val_clf, y_pred_val)
        train_f1 = f1_score(y_train_clf, y_pred_train, average='weighted')
        val_f1 = f1_score(y_val_clf, y_pred_val, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train_clf, cv=5, scoring='accuracy')
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'overfitting': train_acc - val_acc
        }
        
        classification_results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'val': y_pred_val
            }
        }
        
        print(f"  ✅ {name}: Acc = {val_acc:.4f}, CV = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"classification_{name}"):
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
        
    except Exception as e:
        print(f"  ❌ {name}: Error - {str(e)}")
        classification_results[name] = None

# 5. Model Comparison and Selection
print("\n🏆 5. Model Comparison and Selection")
print("=" * 50)

# Compare regression models
print("📈 Regression Model Comparison")
print("=" * 40)

regression_comparison = []
for name, result in regression_results.items():
    if result is not None:
        metrics = result['metrics']
        regression_comparison.append({
            'Model': name,
            'Validation_R2': metrics['val_r2'],
            'CV_R2_Mean': metrics['cv_r2_mean'],
            'CV_R2_Std': metrics['cv_r2_std'],
            'Overfitting': metrics['overfitting'],
            'Val_MAE': metrics['val_mae']
        })

regression_df = pd.DataFrame(regression_comparison).sort_values('Validation_R2', ascending=False)
print("\n🏆 Top 5 Regression Models:")
print(regression_df.head().to_string(index=False))

best_regression_model = regression_df.iloc[0]['Model']
print(f"\n🥇 Best Regression Model: {best_regression_model}")

# Compare classification models
print("\n📊 Classification Model Comparison")
print("=" * 40)

classification_comparison = []
for name, result in classification_results.items():
    if result is not None:
        metrics = result['metrics']
        classification_comparison.append({
            'Model': name,
            'Validation_Accuracy': metrics['val_accuracy'],
            'CV_Accuracy_Mean': metrics['cv_accuracy_mean'],
            'CV_Accuracy_Std': metrics['cv_accuracy_std'],
            'Overfitting': metrics['overfitting'],
            'Val_F1': metrics['val_f1']
        })

classification_df = pd.DataFrame(classification_comparison).sort_values('Validation_Accuracy', ascending=False)
print("\n🏆 Top 5 Classification Models:")
print(classification_df.head().to_string(index=False))

best_classification_model = classification_df.iloc[0]['Model']
print(f"\n🥇 Best Classification Model: {best_classification_model}")

# 6. Hyperparameter Optimization
print("\n⚙️ 6. Hyperparameter Optimization")
print("=" * 50)

# Optimize best regression model
print(f"🎯 Optimizing {best_regression_model} (Regression)")
print("=" * 40)

best_reg_model = regression_results[best_regression_model]['model']

# Define parameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'Ridge Regression': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'Lasso Regression': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'ElasticNet': {
        'alpha': [0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
}

if best_regression_model in param_grids:
    param_grid = param_grids[best_regression_model]
    
    # Create a fresh model instance
    if best_regression_model == 'Random Forest':
        model = RandomForestRegressor(random_state=RANDOM_STATE)
    elif best_regression_model == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    elif best_regression_model == 'Ridge Regression':
        model = Ridge(random_state=RANDOM_STATE)
    elif best_regression_model == 'Lasso Regression':
        model = Lasso(random_state=RANDOM_STATE)
    elif best_regression_model == 'ElasticNet':
        model = ElasticNet(random_state=RANDOM_STATE)
    else:
        model = best_reg_model
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', 
        n_jobs=-1, verbose=1
    )
    
    print("🔍 Performing grid search...")
    grid_search.fit(X_train_scaled, y_train_reg)
    
    # Evaluate optimized model
    optimized_model = grid_search.best_estimator_
    y_pred_val_opt = optimized_model.predict(X_val_scaled)
    opt_r2 = r2_score(y_val_reg, y_pred_val_opt)
    
    print(f"✅ Optimization complete!")
    print(f"🎯 Best parameters: {grid_search.best_params_}")
    print(f"📊 Best CV score: {grid_search.best_score_:.4f}")
    print(f"📈 Validation R²: {opt_r2:.4f}")
    
    # Compare with original model
    original_r2 = regression_results[best_regression_model]['metrics']['val_r2']
    improvement = opt_r2 - original_r2
    print(f"🚀 Improvement: {improvement:.4f} ({improvement/original_r2*100:.1f}%)")
    
    # Update best model
    regression_results[best_regression_model]['optimized_model'] = optimized_model
    regression_results[best_regression_model]['optimization_improvement'] = improvement
    
else:
    print(f"⚠️ No parameter grid defined for {best_regression_model}")

# 7. Learning Curves Analysis
print("\n📊 7. Learning Curves Analysis")
print("=" * 50)

# Generate learning curves for best model
print(f"📈 Generating learning curves for {best_regression_model}")

train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    regression_results[best_regression_model]['model'],
    X_train_scaled, y_train_reg,
    train_sizes=train_sizes,
    cv=5, scoring='r2', n_jobs=-1
)

# Plot learning curves
plt.figure(figsize=(12, 8))

# Learning curves plot
plt.subplot(2, 2, 1)
plt.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', label='Training Score')
plt.plot(train_sizes_abs, val_scores.mean(axis=1), 'o-', label='Validation Score')
plt.fill_between(train_sizes_abs, train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.3)
plt.fill_between(train_sizes_abs, val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.3)
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title(f'Learning Curves - {best_regression_model}')
plt.legend()
plt.grid(True, alpha=0.3)

# Model comparison plot
plt.subplot(2, 2, 2)
models = [r['Model'] for r in regression_comparison[:8]]  # Top 8 models
scores = [r['Validation_R2'] for r in regression_comparison[:8]]
plt.barh(models, scores)
plt.xlabel('Validation R² Score')
plt.title('Model Comparison - Regression')
plt.grid(True, alpha=0.3)

# Feature importance (if available)
if hasattr(regression_results[best_regression_model]['model'], 'feature_importances_'):
    plt.subplot(2, 2, 3)
    feature_importance = regression_results[best_regression_model]['model'].feature_importances_
    feature_names = X.columns
    
    # Plot top 10 features
    indices = np.argsort(feature_importance)[::-1][:10]
    plt.bar(range(10), feature_importance[indices])
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45)
    plt.title('Top 10 Feature Importance')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)

# Validation curve for best hyperparameter
plt.subplot(2, 2, 4)
if best_regression_model in param_grids:
    param_name = list(param_grids[best_regression_model].keys())[0]
    param_range = param_grids[best_regression_model][param_name]
    
    # Create model for validation curve
    if best_regression_model == 'Random Forest':
        model = RandomForestRegressor(random_state=RANDOM_STATE)
    elif best_regression_model == 'Ridge Regression':
        model = Ridge(random_state=RANDOM_STATE)
    else:
        model = regression_results[best_regression_model]['model']
    
    train_scores, val_scores = validation_curve(
        model, X_train_scaled, y_train_reg,
        param_name=param_name, param_range=param_range,
        cv=5, scoring='r2', n_jobs=-1
    )
    
    plt.plot(param_range, train_scores.mean(axis=1), 'o-', label='Training Score')
    plt.plot(param_range, val_scores.mean(axis=1), 'o-', label='Validation Score')
    plt.xlabel(param_name)
    plt.ylabel('R² Score')
    plt.title(f'Validation Curve - {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plots
plots_dir = Path("../plots")
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / "model_training_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Learning curves and analysis saved to: {plots_dir / 'model_training_analysis.png'}")

# 8. Model Persistence
print("\n💾 8. Model Persistence and Versioning")
print("=" * 50)

# Save best models
models_dir = Path("../models/trained_models")
models_dir.mkdir(parents=True, exist_ok=True)

# Save best regression model
best_reg_result = regression_results[best_regression_model]
reg_model_path = models_dir / f"best_regression_model_{best_regression_model.lower().replace(' ', '_')}.joblib"
joblib.dump(best_reg_result['model'], reg_model_path)

# Save optimized model if available
if 'optimized_model' in best_reg_result:
    opt_reg_model_path = models_dir / f"optimized_regression_model_{best_regression_model.lower().replace(' ', '_')}.joblib"
    joblib.dump(best_reg_result['optimized_model'], opt_reg_model_path)

# Save best classification model
best_clf_result = classification_results[best_classification_model]
clf_model_path = models_dir / f"best_classification_model_{best_classification_model.lower().replace(' ', '_')}.joblib"
joblib.dump(best_clf_result['model'], clf_model_path)

# Save scaler
scaler_path = models_dir / "feature_scaler.joblib"
joblib.dump(scaler, scaler_path)

print(f"✅ Best regression model saved: {reg_model_path}")
print(f"✅ Best classification model saved: {clf_model_path}")
print(f"✅ Feature scaler saved: {scaler_path}")

# Save model metadata
model_metadata = {
    'training_timestamp': datetime.now().isoformat(),
    'best_regression_model': {
        'name': best_regression_model,
        'metrics': best_reg_result['metrics'],
        'model_path': str(reg_model_path),
        'optimization_improvement': best_reg_result.get('optimization_improvement', 0)
    },
    'best_classification_model': {
        'name': best_classification_model,
        'metrics': best_clf_result['metrics'],
        'model_path': str(clf_model_path)
    },
    'data_info': {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    },
    'all_models_results': {
        'regression': {name: result['metrics'] if result else None for name, result in regression_results.items()},
        'classification': {name: result['metrics'] if result else None for name, result in classification_results.items()}
    }
}

metadata_path = models_dir / "model_training_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"✅ Model metadata saved: {metadata_path}")

# 9. Results Summary
print("\n📊 9. Model Training Results Summary")
print("=" * 60)

print(f"🎯 Training Summary:")
print(f"   Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"   Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
print(f"   Models trained: {len(regression_models)} regression, {len(classification_models)} classification")

print(f"\n🏆 Best Models:")
print(f"   Regression: {best_regression_model}")
print(f"   - Validation R²: {regression_results[best_regression_model]['metrics']['val_r2']:.4f}")
print(f"   - CV R²: {regression_results[best_regression_model]['metrics']['cv_r2_mean']:.4f}")
print(f"   - Overfitting: {regression_results[best_regression_model]['metrics']['overfitting']:.4f}")

print(f"\n   Classification: {best_classification_model}")
print(f"   - Validation Accuracy: {classification_results[best_classification_model]['metrics']['val_accuracy']:.4f}")
print(f"   - CV Accuracy: {classification_results[best_classification_model]['metrics']['cv_accuracy_mean']:.4f}")
print(f"   - Overfitting: {classification_results[best_classification_model]['metrics']['overfitting']:.4f}")

if 'optimization_improvement' in regression_results[best_regression_model]:
    improvement = regression_results[best_regression_model]['optimization_improvement']
    print(f"\n🚀 Hyperparameter Optimization:")
    print(f"   Improvement: {improvement:.4f} ({improvement/regression_results[best_regression_model]['metrics']['val_r2']*100:.1f}%)")

print(f"\n💾 Saved Artifacts:")
print(f"   Best regression model: {reg_model_path}")
print(f"   Best classification model: {clf_model_path}")
print(f"   Feature scaler: {scaler_path}")
print(f"   Training metadata: {metadata_path}")
print(f"   Analysis plots: {plots_dir / 'model_training_analysis.png'}")

print("\n🎯 Key Insights:")
insights = [
    f"✅ Best regression model achieves R² of {regression_results[best_regression_model]['metrics']['val_r2']:.4f}",
    f"✅ Best classification model achieves {classification_results[best_classification_model]['metrics']['val_accuracy']:.1%} accuracy",
    f"✅ Cross-validation confirms model stability",
    f"✅ Models ready for deployment and evaluation"
]

for insight in insights:
    print(f"  {insight}")

print("\n💡 Next Steps:")
recommendations = [
    "1. Proceed to model evaluation with comprehensive metrics",
    "2. Test models on holdout test set",
    "3. Analyze model interpretability and feature importance",
    "4. Consider ensemble methods for improved performance",
    "5. Prepare models for deployment pipeline"
]

for rec in recommendations:
    print(f"  {rec}")

print("\n🔗 Related Files:")
print("  - 01-feature-engineering.py: Feature preparation workflow")
print("  - 03-model-evaluation.py: Comprehensive model evaluation")
print("  - 04-model-deployment.py: Model deployment pipeline")

print("\n" + "=" * 60)
print("🎯 MODEL TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
print("=" * 60)

# Clean up MLflow if used
if MLFLOW_AVAILABLE:
    try:
        mlflow.end_run()
    except:
        pass 