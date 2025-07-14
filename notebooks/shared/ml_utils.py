"""
Machine Learning Utilities Module
================================================

This module provides comprehensive ML utilities for feature engineering, 
model training, evaluation, and deployment automation in the Modern Data Stack Showcase.

Key Features:
- Feature engineering with automated selection and transformation
- Model training with hyperparameter optimization
- Comprehensive evaluation metrics and visualization
- Model deployment automation
- Model monitoring and drift detection
- A/B testing analysis utilities
- AutoML integration
- Ensemble modeling support

Author: Data Science Team
Version: 1.0.0
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib
import json
import os
from datetime import datetime
import mlflow
import mlflow.sklearn
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class FeatureEngineer:
    """
    Advanced feature engineering utilities for automated feature selection,
    transformation, and preprocessing.
    """
    
    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.feature_names = None
        self.target_encoder = None
        
    def preprocess_data(self, df: pd.DataFrame, target_col: str = None, 
                       test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Comprehensive data preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            Dictionary with processed data splits
        """
        # Create copy to avoid modifying original
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Encode categorical variables
        data = self._encode_categorical_variables(data)
        
        # Feature scaling
        if target_col:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self._scale_features(X_train, fit=True)
            X_test_scaled = self._scale_features(X_test, fit=False)
            
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': X.columns.tolist(),
                'target_name': target_col
            }
        else:
            # Just preprocessing without train/test split
            data_scaled = self._scale_features(data, fit=True)
            return {
                'data': data_scaled,
                'feature_names': data.columns.tolist()
            }
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies."""
        data = df.copy()
        
        # Numeric columns - fill with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col].fillna(data[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown', inplace=True)
        
        return data
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using appropriate encoding."""
        data = df.copy()
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Use label encoding for ordinal data or one-hot for nominal
            if data[col].nunique() <= 10:  # One-hot encoding for low cardinality
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                data.drop(columns=[col], inplace=True)
            else:  # Label encoding for high cardinality
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        
        return data
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        if fit:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df)
        else:
            scaled_data = self.scaler.transform(df)
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def automated_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                  method: str = 'selectkbest', k: int = 10) -> pd.DataFrame:
        """
        Automated feature selection using various methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('selectkbest', 'rfe', 'model_based')
            k: Number of features to select
            
        Returns:
            Selected features DataFrame
        """
        if method == 'selectkbest':
            selector = SelectKBest(k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        elif method == 'model_based':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator, max_features=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.feature_selector = selector
        self.feature_names = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for better model performance."""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        feature_names = poly.get_feature_names_out(X.columns)
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    def create_interaction_features(self, X: pd.DataFrame, interactions: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between specified columns."""
        data = X.copy()
        
        for col1, col2 in interactions:
            if col1 in data.columns and col2 in data.columns:
                interaction_name = f"{col1}_{col2}_interaction"
                data[interaction_name] = data[col1] * data[col2]
        
        return data


class ModelTrainer:
    """
    Advanced model training utilities with hyperparameter optimization,
    cross-validation, and experiment tracking.
    """
    
    def __init__(self, experiment_name: str = "ml-experiments"):
        self.experiment_name = experiment_name
        self.models = {}
        self.best_models = {}
        self.results = {}
        
        # Initialize MLflow
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"MLflow initialization warning: {e}")
    
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             problem_type: str = 'classification') -> Dict:
        """
        Train multiple models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            problem_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with model results
        """
        if problem_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42),
                'SVM': SVC(random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'classification':
                metrics = self._calculate_classification_metrics(y_test, y_pred)
            else:
                metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
            # Log to MLflow
            self._log_to_mlflow(name, model, metrics)
        
        self.results = results
        return results
    
    def hyperparameter_optimization(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  model_type: str = 'random_forest',
                                  search_type: str = 'grid',
                                  cv_folds: int = 5) -> Dict:
        """
        Perform hyperparameter optimization using GridSearch or RandomSearch.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to optimize
            search_type: 'grid' or 'random'
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with optimization results
        """
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(random_state=42)
        }
        
        model = models[model_type]
        param_grid = param_grids[model_type]
        
        # Perform search
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        else:
            search = RandomizedSearchCV(model, param_grid, cv=cv_folds, scoring='accuracy', 
                                      n_iter=20, random_state=42, n_jobs=-1)
        
        search.fit(X_train, y_train)
        
        # Store best model
        self.best_models[model_type] = search.best_estimator_
        
        results = {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
        
        return results
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def _log_to_mlflow(self, model_name: str, model: Any, metrics: Dict):
        """Log model and metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                mlflow.log_params(model.get_params())
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"MLflow logging warning: {e}")


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities with advanced metrics,
    visualization, and interpretability analysis.
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def comprehensive_evaluation(self, y_true: pd.Series, y_pred: np.ndarray,
                               model: Any, X_test: pd.DataFrame,
                               problem_type: str = 'classification') -> Dict:
        """
        Perform comprehensive model evaluation with multiple metrics and visualizations.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model: Trained model
            X_test: Test features
            problem_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with evaluation results
        """
        if problem_type == 'classification':
            results = self._evaluate_classification(y_true, y_pred, model, X_test)
        else:
            results = self._evaluate_regression(y_true, y_pred, model, X_test)
        
        return results
    
    def _evaluate_classification(self, y_true: pd.Series, y_pred: np.ndarray,
                               model: Any, X_test: pd.DataFrame) -> Dict:
        """Evaluate classification model."""
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve (for binary classification)
        try:
            if len(np.unique(y_true)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                metrics['roc_auc'] = roc_auc
                roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
            else:
                roc_data = None
        except:
            roc_data = None
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_test.columns)
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_data': roc_data,
            'feature_importance': feature_importance
        }
    
    def _evaluate_regression(self, y_true: pd.Series, y_pred: np.ndarray,
                           model: Any, X_test: pd.DataFrame) -> Dict:
        """Evaluate regression model."""
        # Basic metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        # Residuals
        residuals = y_true - y_pred
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_test.columns)
        
        return {
            'metrics': metrics,
            'residuals': residuals,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actuals': y_true
        }
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = abs(model.coef_)
            else:
                return None
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        except:
            return None
    
    def create_evaluation_plots(self, evaluation_results: Dict, problem_type: str = 'classification'):
        """Create comprehensive evaluation plots."""
        if problem_type == 'classification':
            self._create_classification_plots(evaluation_results)
        else:
            self._create_regression_plots(evaluation_results)
    
    def _create_classification_plots(self, results: Dict):
        """Create classification evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        if results['roc_data']:
            axes[0, 1].plot(results['roc_data']['fpr'], results['roc_data']['tpr'], 
                          label=f'ROC Curve (AUC = {results["roc_data"]["auc"]:.2f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        
        # Feature Importance
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(10)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 10 Feature Importance')
        
        # Metrics bar plot
        metrics = results['metrics']
        axes[1, 1].bar(metrics.keys(), metrics.values())
        axes[1, 1].set_title('Model Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _create_regression_plots(self, results: Dict):
        """Create regression evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(results['actuals'], results['predictions'], alpha=0.6)
        axes[0, 0].plot([results['actuals'].min(), results['actuals'].max()], 
                       [results['actuals'].min(), results['actuals'].max()], 'r--')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Residuals vs Predicted
        axes[0, 1].scatter(results['predictions'], results['residuals'], alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        
        # Residuals histogram
        axes[1, 0].hist(results['residuals'], bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        
        # Feature Importance
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.show()


class ModelDeployment:
    """
    Model deployment utilities for saving, loading, and serving models
    with versioning and metadata tracking.
    """
    
    def __init__(self, model_registry_path: str = "./model_registry"):
        self.model_registry_path = model_registry_path
        os.makedirs(model_registry_path, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, version: str = None,
                  metadata: Dict = None, performance_metrics: Dict = None):
        """
        Save model with version control and metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Model version (auto-generated if None)
            metadata: Model metadata
            performance_metrics: Performance metrics
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = os.path.join(self.model_registry_path, model_name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        
        # Save metadata
        model_info = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'model_path': model_path,
            'metadata': metadata or {},
            'performance_metrics': performance_metrics or {}
        }
        
        info_path = os.path.join(model_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model saved: {model_name} v{version}")
        return model_dir
    
    def load_model(self, model_name: str, version: str = None):
        """
        Load model from registry.
        
        Args:
            model_name: Name of the model
            version: Model version (latest if None)
            
        Returns:
            Loaded model object
        """
        if version is None:
            # Get latest version
            model_base_dir = os.path.join(self.model_registry_path, model_name)
            if not os.path.exists(model_base_dir):
                raise ValueError(f"Model {model_name} not found")
            
            versions = os.listdir(model_base_dir)
            version = max(versions)
        
        model_dir = os.path.join(self.model_registry_path, model_name, version)
        model_path = os.path.join(model_dir, "model.joblib")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_name} v{version} not found")
        
        model = joblib.load(model_path)
        
        # Load metadata
        info_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print(f"Loaded model: {model_info['model_name']} v{model_info['version']}")
        
        return model
    
    def list_models(self) -> pd.DataFrame:
        """List all models in registry."""
        models_info = []
        
        for model_name in os.listdir(self.model_registry_path):
            model_dir = os.path.join(self.model_registry_path, model_name)
            if os.path.isdir(model_dir):
                for version in os.listdir(model_dir):
                    version_dir = os.path.join(model_dir, version)
                    info_path = os.path.join(version_dir, "model_info.json")
                    
                    if os.path.exists(info_path):
                        with open(info_path, 'r') as f:
                            model_info = json.load(f)
                        models_info.append(model_info)
        
        return pd.DataFrame(models_info)
    
    def create_model_api(self, model_name: str, version: str = None, port: int = 5000):
        """Create a simple REST API for model serving."""
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        model = self.load_model(model_name, version)
        
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()
                features = pd.DataFrame(data)
                predictions = model.predict(features)
                return jsonify({'predictions': predictions.tolist()})
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'model': model_name})
        
        print(f"Starting API for {model_name} on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)


class ModelMonitor:
    """
    Model monitoring utilities for drift detection, performance tracking,
    and alerting in production environments.
    """
    
    def __init__(self):
        self.baseline_stats = {}
        self.performance_history = []
        self.drift_threshold = 0.1
    
    def set_baseline(self, X_baseline: pd.DataFrame, y_baseline: pd.Series = None):
        """Set baseline statistics for drift detection."""
        self.baseline_stats = {
            'feature_stats': X_baseline.describe(),
            'feature_correlations': X_baseline.corr(),
            'timestamp': datetime.now()
        }
        
        if y_baseline is not None:
            self.baseline_stats['target_stats'] = y_baseline.describe()
    
    def detect_data_drift(self, X_current: pd.DataFrame, method: str = 'statistical') -> Dict:
        """
        Detect data drift using statistical or ML-based methods.
        
        Args:
            X_current: Current data batch
            method: Drift detection method ('statistical', 'ks_test')
            
        Returns:
            Dictionary with drift detection results
        """
        if not self.baseline_stats:
            raise ValueError("Baseline statistics not set. Call set_baseline() first.")
        
        drift_results = {}
        
        if method == 'statistical':
            drift_results = self._detect_statistical_drift(X_current)
        elif method == 'ks_test':
            drift_results = self._detect_ks_drift(X_current)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
        
        return drift_results
    
    def _detect_statistical_drift(self, X_current: pd.DataFrame) -> Dict:
        """Detect drift using statistical measures."""
        baseline_stats = self.baseline_stats['feature_stats']
        current_stats = X_current.describe()
        
        drift_scores = {}
        drifted_features = []
        
        for feature in X_current.columns:
            if feature in baseline_stats.columns:
                # Calculate normalized difference in means
                baseline_mean = baseline_stats.loc['mean', feature]
                current_mean = current_stats.loc['mean', feature]
                baseline_std = baseline_stats.loc['std', feature]
                
                if baseline_std > 0:
                    drift_score = abs(current_mean - baseline_mean) / baseline_std
                    drift_scores[feature] = drift_score
                    
                    if drift_score > self.drift_threshold:
                        drifted_features.append(feature)
        
        return {
            'drift_scores': drift_scores,
            'drifted_features': drifted_features,
            'overall_drift': len(drifted_features) / len(X_current.columns),
            'timestamp': datetime.now()
        }
    
    def _detect_ks_drift(self, X_current: pd.DataFrame) -> Dict:
        """Detect drift using Kolmogorov-Smirnov test."""
        # This would require storing baseline distributions
        # For now, return placeholder
        return {
            'method': 'ks_test',
            'message': 'KS test implementation requires baseline distributions',
            'timestamp': datetime.now()
        }
    
    def monitor_model_performance(self, y_true: pd.Series, y_pred: np.ndarray,
                                model_name: str, metrics_type: str = 'classification'):
        """Monitor model performance over time."""
        if metrics_type == 'classification':
            current_metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        else:
            current_metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
        
        performance_record = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'metrics': current_metrics,
            'sample_size': len(y_true)
        }
        
        self.performance_history.append(performance_record)
        
        return current_metrics
    
    def create_monitoring_dashboard(self):
        """Create monitoring dashboard with performance trends."""
        if not self.performance_history:
            print("No performance history available")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.performance_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Over Time', 'Drift Detection',
                          'Sample Size Distribution', 'Performance Distribution')
        )
        
        # Performance over time
        for metric in df['metrics'].iloc[0].keys():
            metric_values = [record['metrics'][metric] for record in self.performance_history]
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=metric_values,
                    mode='lines+markers',
                    name=metric
                ),
                row=1, col=1
            )
        
        # Sample size over time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sample_size'],
                mode='lines+markers',
                name='Sample Size'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
        fig.show()


class ABTestAnalyzer:
    """
    A/B testing analysis utilities for model comparison and statistical testing.
    """
    
    def __init__(self):
        self.test_results = {}
    
    def run_ab_test(self, model_a: Any, model_b: Any, X_test: pd.DataFrame,
                   y_test: pd.Series, test_name: str = "AB_Test",
                   problem_type: str = 'classification') -> Dict:
        """
        Run A/B test between two models.
        
        Args:
            model_a: First model (control)
            model_b: Second model (treatment)
            X_test: Test features
            y_test: Test targets
            test_name: Name of the test
            problem_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with test results
        """
        # Get predictions
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)
        
        # Calculate metrics
        if problem_type == 'classification':
            metrics_a = {
                'accuracy': accuracy_score(y_test, pred_a),
                'precision': precision_score(y_test, pred_a, average='weighted'),
                'recall': recall_score(y_test, pred_a, average='weighted'),
                'f1': f1_score(y_test, pred_a, average='weighted')
            }
            metrics_b = {
                'accuracy': accuracy_score(y_test, pred_b),
                'precision': precision_score(y_test, pred_b, average='weighted'),
                'recall': recall_score(y_test, pred_b, average='weighted'),
                'f1': f1_score(y_test, pred_b, average='weighted')
            }
        else:
            metrics_a = {
                'mse': mean_squared_error(y_test, pred_a),
                'mae': mean_absolute_error(y_test, pred_a),
                'r2': r2_score(y_test, pred_a)
            }
            metrics_b = {
                'mse': mean_squared_error(y_test, pred_b),
                'mae': mean_absolute_error(y_test, pred_b),
                'r2': r2_score(y_test, pred_b)
            }
        
        # Statistical significance testing
        significance_results = self._test_significance(pred_a, pred_b, y_test, problem_type)
        
        results = {
            'test_name': test_name,
            'model_a_metrics': metrics_a,
            'model_b_metrics': metrics_b,
            'significance_test': significance_results,
            'timestamp': datetime.now()
        }
        
        self.test_results[test_name] = results
        return results
    
    def _test_significance(self, pred_a: np.ndarray, pred_b: np.ndarray,
                          y_test: pd.Series, problem_type: str) -> Dict:
        """Test statistical significance of difference between models."""
        if problem_type == 'classification':
            # McNemar's test for classification
            correct_a = (pred_a == y_test).astype(int)
            correct_b = (pred_b == y_test).astype(int)
            
            # Create contingency table
            both_correct = np.sum((correct_a == 1) & (correct_b == 1))
            a_correct_b_wrong = np.sum((correct_a == 1) & (correct_b == 0))
            a_wrong_b_correct = np.sum((correct_a == 0) & (correct_b == 1))
            both_wrong = np.sum((correct_a == 0) & (correct_b == 0))
            
            # McNemar's test statistic
            if a_correct_b_wrong + a_wrong_b_correct > 0:
                mcnemar_stat = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (a_correct_b_wrong + a_wrong_b_correct)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
            else:
                mcnemar_stat = 0
                p_value = 1.0
            
            return {
                'test_type': 'McNemar',
                'statistic': mcnemar_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'contingency_table': {
                    'both_correct': both_correct,
                    'a_correct_b_wrong': a_correct_b_wrong,
                    'a_wrong_b_correct': a_wrong_b_correct,
                    'both_wrong': both_wrong
                }
            }
        else:
            # Paired t-test for regression
            errors_a = np.abs(y_test - pred_a)
            errors_b = np.abs(y_test - pred_b)
            
            stat, p_value = stats.ttest_rel(errors_a, errors_b)
            
            return {
                'test_type': 'Paired t-test',
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_error_a': np.mean(errors_a),
                'mean_error_b': np.mean(errors_b)
            }
    
    def visualize_ab_test(self, test_name: str):
        """Visualize A/B test results."""
        if test_name not in self.test_results:
            print(f"Test {test_name} not found")
            return
        
        results = self.test_results[test_name]
        
        # Create comparison plot
        metrics_a = results['model_a_metrics']
        metrics_b = results['model_b_metrics']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Metrics comparison
        metrics_names = list(metrics_a.keys())
        values_a = [metrics_a[m] for m in metrics_names]
        values_b = [metrics_b[m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax1.bar(x - width/2, values_a, width, label='Model A', alpha=0.8)
        ax1.bar(x + width/2, values_b, width, label='Model B', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title(f'A/B Test Results: {test_name}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names)
        ax1.legend()
        
        # Statistical significance
        sig_test = results['significance_test']
        ax2.text(0.1, 0.8, f"Test Type: {sig_test['test_type']}", transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.7, f"P-value: {sig_test['p_value']:.4f}", transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.6, f"Significant: {sig_test['significant']}", transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.5, f"Test Statistic: {sig_test['statistic']:.4f}", transform=ax2.transAxes, fontsize=12)
        
        ax2.set_title('Statistical Significance')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()


# Utility functions for quick access
def quick_train_evaluate(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        problem_type: str = 'classification') -> Dict:
    """
    Quick training and evaluation pipeline.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with training and evaluation results
    """
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Train multiple models
    training_results = trainer.train_multiple_models(
        X_train, y_train, X_test, y_test, problem_type
    )
    
    # Evaluate best model
    best_model_name = max(training_results.keys(), 
                         key=lambda x: list(training_results[x]['metrics'].values())[0])
    best_model = training_results[best_model_name]['model']
    
    evaluation_results = evaluator.comprehensive_evaluation(
        y_test, training_results[best_model_name]['predictions'],
        best_model, X_test, problem_type
    )
    
    return {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'best_model': best_model_name
    }


def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 10,
                           problem_type: str = 'classification') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create synthetic dataset for testing and demonstrations.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        problem_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    from sklearn.datasets import make_classification, make_regression
    
    if problem_type == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_informative=n_features//2, n_redundant=0,
                                 n_clusters_per_class=1, random_state=42)
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features,
                             n_informative=n_features//2, random_state=42)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


# Example usage and configuration
if __name__ == "__main__":
    print("ML Utils Module Loaded Successfully!")
    print("Available classes: FeatureEngineer, ModelTrainer, ModelEvaluator, ModelDeployment, ModelMonitor, ABTestAnalyzer")
    print("Quick functions: quick_train_evaluate, create_synthetic_dataset") 