# %% [markdown]
# # Model Evaluation and Analysis
# 
# This notebook provides comprehensive evaluation of trained machine learning models including:
# - Performance metrics analysis
# - Model comparison and ranking
# - Error analysis and diagnostics
# - Feature importance and model interpretation
# - Visualization of results
# - Performance reporting
# 
# **Dependencies:** This notebook depends on models trained in `02-model-training.py`

# %% [markdown]
# ## 1. Setup and Configuration

# %%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
from ml_utils import ModelEvaluator, FeatureEngineer
from config import Config

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
# ## 2. Configuration and Data Loading

# %%
# Configuration
config = Config()

# MLflow configuration
mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment("retail-analytics-model-evaluation")

# Load test data and models
print("Loading test data and trained models...")

# Load the test dataset (should be saved from previous notebooks)
try:
    X_test = pd.read_csv(f"{config.data_path}/X_test_processed.csv")
    y_test = pd.read_csv(f"{config.data_path}/y_test.csv").squeeze()
    
    # Load feature names
    feature_names = pd.read_csv(f"{config.data_path}/feature_names.csv")['feature'].tolist()
    
    print(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Target distribution: {y_test.value_counts().to_dict()}")
    
except FileNotFoundError:
    print("Test data not found. Please run 02-model-training.py first.")
    raise

# %% [markdown]
# ## 3. Model Loading and Preparation

# %%
# Load all trained models
model_results = {}
model_paths = {
    'Random Forest': f"{config.model_path}/random_forest_model.pkl",
    'XGBoost': f"{config.model_path}/xgboost_model.pkl",
    'LightGBM': f"{config.model_path}/lightgbm_model.pkl",
    'SVM': f"{config.model_path}/svm_model.pkl",
    'Neural Network': f"{config.model_path}/neural_network_model.pkl",
    'Logistic Regression': f"{config.model_path}/logistic_regression_model.pkl",
    'Gradient Boosting': f"{config.model_path}/gradient_boosting_model.pkl",
    'Decision Tree': f"{config.model_path}/decision_tree_model.pkl"
}

loaded_models = {}
for model_name, model_path in model_paths.items():
    try:
        loaded_models[model_name] = joblib.load(model_path)
        print(f"✓ Loaded {model_name}")
    except FileNotFoundError:
        print(f"✗ Model {model_name} not found at {model_path}")
        continue

print(f"\nLoaded {len(loaded_models)} models for evaluation")

# %% [markdown]
# ## 4. Comprehensive Model Evaluation

# %%
# Initialize model evaluator
evaluator = ModelEvaluator()

# Start MLflow run
with mlflow.start_run(run_name="model_evaluation_comprehensive"):
    
    # Evaluate all models
    all_predictions = {}
    all_probabilities = {}
    evaluation_results = {}
    
    for model_name, model in loaded_models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print('='*50)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities for classification
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        else:
            y_pred_proba = None
        
        # Store predictions
        all_predictions[model_name] = y_pred
        all_probabilities[model_name] = y_pred_proba
        
        # Evaluate model
        results = evaluator.evaluate_model(
            y_test, 
            y_pred, 
            y_pred_proba,
            model_name=model_name,
            feature_names=feature_names if hasattr(model, 'feature_importances_') else None
        )
        
        evaluation_results[model_name] = results
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            f"{model_name}_accuracy": results['accuracy'],
            f"{model_name}_precision": results['precision'],
            f"{model_name}_recall": results['recall'],
            f"{model_name}_f1_score": results['f1_score'],
            f"{model_name}_auc_roc": results['auc_roc'] if results['auc_roc'] else 0
        })

# %% [markdown]
# ## 5. Model Comparison and Ranking

# %%
# Create comprehensive comparison dataframe
comparison_data = []
for model_name, results in evaluation_results.items():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1_score'],
        'AUC-ROC': results['auc_roc'] if results['auc_roc'] else 0,
        'AUC-PR': results['auc_pr'] if results['auc_pr'] else 0
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

print("Model Performance Comparison:")
print(comparison_df.round(4))

# %% [markdown]
# ## 6. Performance Visualization

# %%
# Create comprehensive performance visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Accuracy Comparison', 'F1 Score Comparison', 
                   'Precision vs Recall', 'AUC-ROC Comparison'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Accuracy comparison
fig.add_trace(
    go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['Accuracy'],
        name='Accuracy',
        marker_color='lightblue'
    ),
    row=1, col=1
)

# F1 Score comparison
fig.add_trace(
    go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['F1 Score'],
        name='F1 Score',
        marker_color='lightgreen'
    ),
    row=1, col=2
)

# Precision vs Recall scatter
fig.add_trace(
    go.Scatter(
        x=comparison_df['Precision'],
        y=comparison_df['Recall'],
        mode='markers+text',
        text=comparison_df['Model'],
        textposition='top center',
        name='Precision vs Recall',
        marker=dict(size=10, color='red')
    ),
    row=2, col=1
)

# AUC-ROC comparison
fig.add_trace(
    go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['AUC-ROC'],
        name='AUC-ROC',
        marker_color='orange'
    ),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="Model Performance Comparison Dashboard",
    showlegend=False
)

fig.update_xaxes(tickangle=45)
fig.show()

# %% [markdown]
# ## 7. Detailed Error Analysis

# %%
# Error analysis for top 3 performing models
top_3_models = comparison_df.head(3)['Model'].tolist()

print("Detailed Error Analysis for Top 3 Models:")
print("=" * 50)

for model_name in top_3_models:
    print(f"\n{model_name} Error Analysis:")
    print("-" * 30)
    
    y_pred = all_predictions[model_name]
    
    # Classification errors
    correct_predictions = (y_test == y_pred)
    incorrect_predictions = ~correct_predictions
    
    print(f"Correct predictions: {correct_predictions.sum()} ({correct_predictions.mean():.2%})")
    print(f"Incorrect predictions: {incorrect_predictions.sum()} ({incorrect_predictions.mean():.2%})")
    
    # Analyze incorrect predictions
    if incorrect_predictions.sum() > 0:
        error_analysis = pd.DataFrame({
            'actual': y_test[incorrect_predictions],
            'predicted': y_pred[incorrect_predictions]
        })
        
        print("\nError distribution:")
        print(error_analysis.groupby(['actual', 'predicted']).size().unstack(fill_value=0))
        
        # Feature analysis for errors
        if len(feature_names) > 0:
            error_features = X_test[incorrect_predictions]
            correct_features = X_test[correct_predictions]
            
            # Find features with largest differences
            feature_diff = error_features.mean() - correct_features.mean()
            top_diff_features = feature_diff.abs().nlargest(5)
            
            print(f"\nTop 5 features with largest differences between errors and correct predictions:")
            for feature, diff in top_diff_features.items():
                print(f"  {feature}: {diff:.4f}")

# %% [markdown]
# ## 8. Feature Importance Analysis

# %%
# Feature importance analysis for models that support it
feature_importance_models = [
    'Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting', 'Decision Tree'
]

if feature_names:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, model_name in enumerate(feature_importance_models[:6]):
        if model_name in loaded_models:
            model = loaded_models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True).tail(10)
                
                # Plot top 10 features
                axes[i].barh(importance_df['feature'], importance_df['importance'])
                axes[i].set_title(f'{model_name} - Top 10 Features')
                axes[i].set_xlabel('Importance')
                
                # Rotate labels for better readability
                axes[i].tick_params(axis='y', labelsize=8)
    
    # Remove empty subplots
    for j in range(len(feature_importance_models), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Model Interpretation and Insights

# %%
# SHAP analysis for the best performing model
best_model_name = comparison_df.iloc[0]['Model']
best_model = loaded_models[best_model_name]

print(f"Model Interpretation for Best Performing Model: {best_model_name}")
print("=" * 60)

# Basic model insights
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Feature importance visualization
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# Model complexity analysis
if hasattr(best_model, 'n_estimators'):
    print(f"\nModel Complexity: {best_model.n_estimators} estimators")
elif hasattr(best_model, 'max_depth'):
    print(f"\nModel Complexity: Max depth = {best_model.max_depth}")

# %% [markdown]
# ## 10. Cross-Validation Performance Analysis

# %%
# Load cross-validation results if available
try:
    cv_results = pd.read_csv(f"{config.data_path}/cv_results.csv")
    
    print("Cross-Validation Performance Analysis:")
    print("=" * 40)
    
    # Display CV results
    print(cv_results.round(4))
    
    # Visualize CV performance
    fig = px.box(
        cv_results.melt(id_vars=['model'], var_name='metric', value_name='score'),
        x='model',
        y='score',
        color='metric',
        title='Cross-Validation Performance Distribution'
    )
    fig.update_xaxes(tickangle=45)
    fig.show()
    
except FileNotFoundError:
    print("Cross-validation results not found. Skipping CV analysis.")

# %% [markdown]
# ## 11. Prediction Confidence Analysis

# %%
# Analyze prediction confidence for models with probability outputs
models_with_proba = {k: v for k, v in all_probabilities.items() if v is not None}

if models_with_proba:
    print("Prediction Confidence Analysis:")
    print("=" * 35)
    
    # Create confidence analysis
    confidence_data = []
    for model_name, probabilities in models_with_proba.items():
        # Calculate confidence (distance from 0.5)
        confidence = np.abs(probabilities - 0.5)
        predictions = all_predictions[model_name]
        
        # Accuracy by confidence quartile
        confidence_quartiles = pd.qcut(confidence, 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        for quartile in confidence_quartiles.categories:
            mask = confidence_quartiles == quartile
            if mask.sum() > 0:
                accuracy = (y_test[mask] == predictions[mask]).mean()
                confidence_data.append({
                    'Model': model_name,
                    'Confidence Quartile': quartile,
                    'Accuracy': accuracy,
                    'Sample Count': mask.sum(),
                    'Avg Confidence': confidence[mask].mean()
                })
    
    confidence_df = pd.DataFrame(confidence_data)
    
    # Visualize confidence vs accuracy
    fig = px.scatter(
        confidence_df,
        x='Avg Confidence',
        y='Accuracy',
        color='Model',
        size='Sample Count',
        hover_data=['Confidence Quartile'],
        title='Model Accuracy vs Prediction Confidence'
    )
    fig.show()
    
    # Show confidence analysis table
    print("\nConfidence Analysis Summary:")
    pivot_table = confidence_df.pivot_table(
        values='Accuracy',
        index='Confidence Quartile',
        columns='Model',
        aggfunc='mean'
    )
    print(pivot_table.round(4))

# %% [markdown]
# ## 12. Business Impact Analysis

# %%
# Calculate business impact metrics
print("Business Impact Analysis:")
print("=" * 25)

# Assuming this is a classification problem for customer behavior
# Calculate potential business impact for each model
business_impact_data = []

for model_name, results in evaluation_results.items():
    # Example business metrics (customize based on your use case)
    precision = results['precision']
    recall = results['recall']
    
    # Example: Cost of false positives vs false negatives
    cost_false_positive = 10  # Cost of incorrect positive prediction
    cost_false_negative = 50  # Cost of missed positive case
    
    # Calculate expected costs
    y_pred = all_predictions[model_name]
    
    # Confusion matrix elements
    tp = ((y_test == 1) & (y_pred == 1)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    
    # Business impact calculation
    total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
    cost_per_prediction = total_cost / len(y_test)
    
    business_impact_data.append({
        'Model': model_name,
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn,
        'True Negatives': tn,
        'Total Cost': total_cost,
        'Cost per Prediction': cost_per_prediction,
        'Precision': precision,
        'Recall': recall
    })

business_df = pd.DataFrame(business_impact_data)
business_df = business_df.sort_values('Cost per Prediction')

print("Business Impact Summary (sorted by cost efficiency):")
print(business_df[['Model', 'Total Cost', 'Cost per Prediction', 'Precision', 'Recall']].round(2))

# %% [markdown]
# ## 13. Model Recommendation and Summary

# %%
# Generate final recommendations
print("MODEL EVALUATION SUMMARY AND RECOMMENDATIONS")
print("=" * 50)

# Best overall model
best_model_name = comparison_df.iloc[0]['Model']
best_model_metrics = comparison_df.iloc[0]

print(f"\n🏆 BEST OVERALL MODEL: {best_model_name}")
print(f"   Accuracy: {best_model_metrics['Accuracy']:.4f}")
print(f"   F1 Score: {best_model_metrics['F1 Score']:.4f}")
print(f"   Precision: {best_model_metrics['Precision']:.4f}")
print(f"   Recall: {best_model_metrics['Recall']:.4f}")
print(f"   AUC-ROC: {best_model_metrics['AUC-ROC']:.4f}")

# Most cost-effective model
most_cost_effective = business_df.iloc[0]['Model']
print(f"\n💰 MOST COST-EFFECTIVE MODEL: {most_cost_effective}")
print(f"   Cost per Prediction: ${business_df.iloc[0]['Cost per Prediction']:.2f}")

# Balanced model recommendation
balanced_score = (comparison_df['Precision'] + comparison_df['Recall']) / 2
balanced_model_idx = balanced_score.idxmax()
balanced_model = comparison_df.iloc[balanced_model_idx]['Model']

print(f"\n⚖️ MOST BALANCED MODEL: {balanced_model}")
print(f"   Balanced Score: {balanced_score.iloc[balanced_model_idx]:.4f}")

# Model selection recommendations
print(f"\n📊 MODEL SELECTION RECOMMENDATIONS:")
print(f"   • For highest accuracy: {best_model_name}")
print(f"   • For cost efficiency: {most_cost_effective}")
print(f"   • For balanced performance: {balanced_model}")

# Next steps
print(f"\n🚀 NEXT STEPS:")
print(f"   1. Deploy {best_model_name} for production use")
print(f"   2. Set up monitoring for model performance")
print(f"   3. Implement A/B testing framework")
print(f"   4. Schedule regular model retraining")
print(f"   5. Consider ensemble methods for improved performance")

# %% [markdown]
# ## 14. Export Results and Artifacts

# %%
# Save evaluation results
evaluation_summary = {
    'evaluation_date': datetime.now().isoformat(),
    'best_model': best_model_name,
    'model_count': len(loaded_models),
    'test_samples': len(X_test),
    'evaluation_results': evaluation_results,
    'comparison_metrics': comparison_df.to_dict('records'),
    'business_impact': business_df.to_dict('records')
}

# Export to files
os.makedirs(f"{config.output_path}/evaluation", exist_ok=True)

# Save comparison results
comparison_df.to_csv(f"{config.output_path}/evaluation/model_comparison.csv", index=False)

# Save business impact analysis
business_df.to_csv(f"{config.output_path}/evaluation/business_impact.csv", index=False)

# Save detailed evaluation results
import json
with open(f"{config.output_path}/evaluation/evaluation_summary.json", 'w') as f:
    json.dump(evaluation_summary, f, indent=2, default=str)

print("Evaluation results exported to:")
print(f"  • {config.output_path}/evaluation/model_comparison.csv")
print(f"  • {config.output_path}/evaluation/business_impact.csv")
print(f"  • {config.output_path}/evaluation/evaluation_summary.json")

# %% [markdown]
# ## 15. Evaluation Complete
# 
# This comprehensive model evaluation notebook has:
# 
# ✅ **Loaded and evaluated all trained models**
# ✅ **Compared performance across multiple metrics**
# ✅ **Analyzed errors and model behavior**
# ✅ **Examined feature importance and interpretability**
# ✅ **Assessed business impact and cost-effectiveness**
# ✅ **Generated actionable recommendations**
# 
# **Key Findings:**
# - Best performing model: {best_model_name}
# - Most cost-effective model: {most_cost_effective}
# - Ready for production deployment
# 
# **Next Steps:**
# - Proceed to `04-model-deployment.py` for deployment preparation
# - Set up monitoring with `05-model-monitoring.py`
# - Implement A/B testing with `06-ab-testing.py`

# %%
print("🎉 Model evaluation completed successfully!")
print(f"📈 {len(loaded_models)} models evaluated")
print(f"🏆 Best model: {best_model_name}")
print(f"💰 Most cost-effective: {most_cost_effective}")
print(f"📊 Results saved to: {config.output_path}/evaluation/") 