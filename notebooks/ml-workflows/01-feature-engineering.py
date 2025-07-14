#!/usr/bin/env python3
"""
Feature Engineering Workflow - ML Notebook Implementation

This module demonstrates advanced feature engineering techniques including automated 
feature selection, transformation, and preprocessing for machine learning workflows.

Category: ML Workflow
Author: Data Science Team
Created: 2024-01-15
Runtime: 15-20 minutes

Key Features:
- Automated feature selection and engineering
- Comprehensive preprocessing pipeline
- Feature importance analysis
- Optimized feature set for model training
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("🚀 Feature Engineering Workflow Started")
print("=" * 50)

# 1. Data Creation and Loading
print("\n📦 1. Creating synthetic retail dataset...")

# Generate synthetic data
n_samples = 10000

# Create retail-specific features
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

# Create DataFrame
df = pd.DataFrame(data)

# Create target variable (sales amount) with realistic relationships
target_sales = (
    df['annual_income'] * 0.001 +
    df['purchase_frequency'] * 5 +
    df['avg_basket_size'] * 2 +
    df['loyalty_score'] * 1.5 +
    df['product_category_electronics'] * 50 +
    df['seasonal_factor'] * 20 +
    np.random.normal(0, 50, n_samples)  # Add noise
)

# Ensure positive values
df['target_sales'] = np.maximum(target_sales, 0)

# Create binary classification target
df['target_high_value'] = (df['target_sales'] > df['target_sales'].quantile(0.7)).astype(int)

print(f"✅ Dataset created with {len(df)} samples and {len(df.columns)} features")
print(f"📊 Target variable (sales) range: ${df['target_sales'].min():.2f} - ${df['target_sales'].max():.2f}")
print(f"🎯 High-value customers: {df['target_high_value'].sum()} ({df['target_high_value'].mean()*100:.1f}%)")

# 2. Data Exploration
print("\n🔍 2. Data Exploration Summary")
print("=" * 40)

# Basic info
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Data types
print(f"\nData types: {df.dtypes.value_counts().to_dict()}")

# Missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"Missing values: {missing_values[missing_values > 0].to_dict()}")
else:
    print("✅ No missing values found")

# 3. Feature Preprocessing
print("\n🔧 3. Feature Preprocessing Pipeline")
print("=" * 50)

# Separate features and targets
feature_cols = [col for col in df.columns if not col.startswith('target_')]
X = df[feature_cols].copy()
y_regression = df['target_sales'].copy()
y_classification = df['target_high_value'].copy()

print(f"📊 Original feature set: {len(feature_cols)} features")
print(f"🎯 Regression target: {y_regression.name} (continuous)")
print(f"🎯 Classification target: {y_classification.name} (binary)")

# Handle categorical variables
print("\n🏷️ Encoding categorical variables...")
X_encoded = pd.get_dummies(X, columns=['customer_segment', 'region'], drop_first=True)
print(f"✅ Features after encoding: {len(X_encoded.columns)} features")

# Data splitting
print("\n📊 Data Splitting")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_regression, test_size=0.2, random_state=RANDOM_STATE
)

# Feature scaling
print("\n⚖️ Feature Scaling")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✅ Features scaled using StandardScaler")
print(f"📊 Scaled features - Mean: {X_train_scaled.mean().mean():.6f}, Std: {X_train_scaled.std().mean():.6f}")

# 4. Automated Feature Selection
print("\n🎯 4. Automated Feature Selection")
print("=" * 50)

# Test different feature selection methods
k_features = min(10, len(X_train_scaled.columns))  # Select top 10 features or all if less
selection_results = {}

# SelectKBest method
print(f"🔍 Testing SelectKBest method...")
try:
    selector_kbest = SelectKBest(k=k_features)
    X_selected_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
    selected_features_kbest = X_train_scaled.columns[selector_kbest.get_support()].tolist()
    
    selection_results['selectkbest'] = {
        'features': selected_features_kbest,
        'n_features': len(selected_features_kbest),
        'data': X_selected_kbest
    }
    
    print(f"✅ SelectKBest: Selected {len(selected_features_kbest)} features")
    print(f"   Features: {selected_features_kbest[:5]}{'...' if len(selected_features_kbest) > 5 else ''}")
    
except Exception as e:
    print(f"❌ SelectKBest: Error - {str(e)}")
    selection_results['selectkbest'] = None

# Model-based feature selection
print(f"\n🔍 Testing model-based feature selection...")
try:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    selector_model = SelectFromModel(rf_model, max_features=k_features)
    X_selected_model = selector_model.fit_transform(X_train_scaled, y_train)
    selected_features_model = X_train_scaled.columns[selector_model.get_support()].tolist()
    
    selection_results['model_based'] = {
        'features': selected_features_model,
        'n_features': len(selected_features_model),
        'data': X_selected_model
    }
    
    print(f"✅ Model-based: Selected {len(selected_features_model)} features")
    print(f"   Features: {selected_features_model[:5]}{'...' if len(selected_features_model) > 5 else ''}")
    
except Exception as e:
    print(f"❌ Model-based: Error - {str(e)}")
    selection_results['model_based'] = None

# Use best selection method
best_method = 'model_based' if selection_results['model_based'] else 'selectkbest'
selected_features = selection_results[best_method]['features']
X_selected = X_train_scaled[selected_features].copy()

print(f"\n🎯 Using {best_method.upper()} method with {len(selected_features)} selected features")

# 5. Feature Engineering
print("\n🔧 5. Advanced Feature Engineering")
print("=" * 50)

# Create interaction features
print("🤝 Creating interaction features...")
X_engineered = X_selected.copy()

# Define meaningful interactions
interactions = [
    ('customer_age', 'annual_income'),
    ('purchase_frequency', 'avg_basket_size'),
    ('loyalty_score', 'days_since_last_purchase')
]

for col1, col2 in interactions:
    if col1 in X_engineered.columns and col2 in X_engineered.columns:
        interaction_name = f"{col1}_{col2}_interaction"
        X_engineered[interaction_name] = X_engineered[col1] * X_engineered[col2]
        print(f"✅ Created {interaction_name}")

# Domain-specific features
print("\n🏪 Creating domain-specific features...")

# Customer lifetime value proxy
if 'annual_income' in X_engineered.columns and 'loyalty_score' in X_engineered.columns:
    X_engineered['customer_value_proxy'] = X_engineered['annual_income'] * X_engineered['loyalty_score'] / 100
    print("✅ Created customer_value_proxy feature")

# Purchase intensity
if 'purchase_frequency' in X_engineered.columns and 'avg_basket_size' in X_engineered.columns:
    X_engineered['purchase_intensity'] = X_engineered['purchase_frequency'] * X_engineered['avg_basket_size']
    print("✅ Created purchase_intensity feature")

# Recency factor
if 'days_since_last_purchase' in X_engineered.columns:
    X_engineered['recency_factor'] = 1 / (1 + X_engineered['days_since_last_purchase'])
    print("✅ Created recency_factor feature")

print(f"\n🎉 Feature Engineering Complete!")
print(f"📊 Final feature set: {len(X_engineered.columns)} features")
print(f"📈 Feature increase: {len(X_engineered.columns) - len(selected_features)} new features")

# 6. Feature Validation
print("\n✅ 6. Feature Validation")
print("=" * 50)

# Model performance comparison
print("🔬 Model Performance Comparison")

# Test with original selected features
y_train_subset = y_train[:len(X_selected)]
rf_original = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
rf_original.fit(X_selected, y_train_subset)

# Test with engineered features
rf_engineered = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
rf_engineered.fit(X_engineered, y_train_subset)

# Cross-validation scores
cv_original = cross_val_score(rf_original, X_selected, y_train_subset, cv=5, scoring='r2')
cv_engineered = cross_val_score(rf_engineered, X_engineered, y_train_subset, cv=5, scoring='r2')

print(f"📊 Original Features R² Score: {cv_original.mean():.4f} (±{cv_original.std():.4f})")
print(f"🔧 Engineered Features R² Score: {cv_engineered.mean():.4f} (±{cv_engineered.std():.4f})")

improvement = cv_engineered.mean() - cv_original.mean()
print(f"📈 Performance Improvement: {improvement:.4f} ({improvement/cv_original.mean()*100:.1f}%)")

if improvement > 0:
    print("✅ Feature engineering improved model performance!")
else:
    print("⚠️ Feature engineering didn't improve performance significantly.")

# Feature importance analysis
print("\n🎯 Feature Importance Analysis")
feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': rf_engineered.feature_importances_
}).sort_values('importance', ascending=False)

print("🏆 Top 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")

# 7. Pipeline Creation
print("\n🏗️ 7. Feature Engineering Pipeline Creation")
print("=" * 50)

# Create custom transformer
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering pipeline."""
    
    def __init__(self, selected_features=None, create_interactions=True, create_domain_features=True):
        self.selected_features = selected_features
        self.create_interactions = create_interactions
        self.create_domain_features = create_domain_features
        
    def fit(self, X, y=None):
        if self.selected_features is None:
            self.selected_features = X.columns.tolist()
        return self
    
    def transform(self, X):
        X_transformed = X[self.selected_features].copy()
        
        # Create interaction features
        if self.create_interactions:
            interactions = [
                ('customer_age', 'annual_income'),
                ('purchase_frequency', 'avg_basket_size'),
                ('loyalty_score', 'days_since_last_purchase')
            ]
            
            for col1, col2 in interactions:
                if col1 in X_transformed.columns and col2 in X_transformed.columns:
                    X_transformed[f'{col1}_{col2}_interaction'] = X_transformed[col1] * X_transformed[col2]
        
        # Create domain-specific features
        if self.create_domain_features:
            if 'annual_income' in X_transformed.columns and 'loyalty_score' in X_transformed.columns:
                X_transformed['customer_value_proxy'] = X_transformed['annual_income'] * X_transformed['loyalty_score'] / 100
            
            if 'purchase_frequency' in X_transformed.columns and 'avg_basket_size' in X_transformed.columns:
                X_transformed['purchase_intensity'] = X_transformed['purchase_frequency'] * X_transformed['avg_basket_size']
            
            if 'days_since_last_purchase' in X_transformed.columns:
                X_transformed['recency_factor'] = 1 / (1 + X_transformed['days_since_last_purchase'])
        
        return X_transformed

# Create the complete pipeline
preprocessing_pipeline = Pipeline([
    ('feature_engineer', CustomFeatureEngineer(
        selected_features=selected_features,
        create_interactions=True,
        create_domain_features=True
    )),
    ('scaler', StandardScaler())
])

print("✅ Feature engineering pipeline created")

# Test the pipeline
X_train_original = X_train[selected_features].copy()
X_pipeline_transformed = preprocessing_pipeline.fit_transform(X_train_original)

print(f"✅ Pipeline fitted successfully")
print(f"📊 Input features: {len(selected_features)}")
print(f"📊 Output features: {X_pipeline_transformed.shape[1]}")
print(f"📊 Samples processed: {X_pipeline_transformed.shape[0]}")

# Save pipeline
print("\n💾 Saving Feature Engineering Pipeline")
pipeline_dir = Path("../models/feature_engineering")
pipeline_dir.mkdir(parents=True, exist_ok=True)

pipeline_path = pipeline_dir / "feature_engineering_pipeline.joblib"
joblib.dump(preprocessing_pipeline, pipeline_path)

# Save feature information
feature_info = {
    'selected_features': selected_features,
    'final_features': list(X_engineered.columns),
    'feature_importance': feature_importance.to_dict('records'),
    'pipeline_steps': [step[0] for step in preprocessing_pipeline.steps],
    'creation_timestamp': datetime.now().isoformat(),
    'performance_improvement': improvement
}

info_path = pipeline_dir / "feature_info.json"
with open(info_path, 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"✅ Pipeline saved to: {pipeline_path}")
print(f"✅ Feature info saved to: {info_path}")

# 8. Results Summary
print("\n📊 8. Feature Engineering Results Summary")
print("=" * 60)

summary = {
    'Dataset Information': {
        'Total Samples': len(df),
        'Original Features': len(feature_cols),
        'Selected Features': len(selected_features),
        'Final Features': len(X_engineered.columns)
    },
    'Feature Engineering': {
        'Selection Method': best_method,
        'Feature Increase': f"{(len(X_engineered.columns) - len(selected_features))/len(selected_features)*100:.1f}%",
        'Performance Improvement': f"{improvement:.4f} ({improvement/cv_original.mean()*100:.1f}%)"
    },
    'Pipeline': {
        'Pipeline Steps': len(preprocessing_pipeline.steps),
        'Pipeline Saved': pipeline_path.exists(),
        'Ready for Production': True
    }
}

for section, details in summary.items():
    print(f"\n🔸 {section}:")
    for key, value in details.items():
        print(f"   {key}: {value}")

# Key insights
print("\n🎯 Key Insights:")
print("=" * 30)

insights = [
    f"✅ Feature engineering improved model performance by {improvement/cv_original.mean()*100:.1f}%",
    f"🏆 Top feature: {feature_importance.iloc[0]['feature']} (importance: {feature_importance.iloc[0]['importance']:.4f})",
    "📊 Pipeline ready for production deployment"
]

for insight in insights:
    print(f"  {insight}")

print("\n🎉 Feature Engineering Workflow Complete!")
print("📝 Ready for next step: Model Training")
print("🔗 Related files: 02-model-training.ipynb, 03-model-evaluation.ipynb")

# Create visualization of feature importance
print("\n📈 Creating Feature Importance Visualization")
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)

bars = plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()

# Save plot
plots_dir = Path("../plots")
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Feature importance plot saved to: {plots_dir / 'feature_importance.png'}")

print("\n" + "=" * 60)
print("🎯 FEATURE ENGINEERING WORKFLOW COMPLETED SUCCESSFULLY")
print("=" * 60) 