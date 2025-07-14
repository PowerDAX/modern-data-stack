# %% [markdown]
# # A/B Testing Framework for ML Models
# 
# This notebook provides a comprehensive A/B testing framework for ML models including:
# - Experimental design and setup
# - Traffic splitting and routing
# - Statistical significance testing
# - Performance comparison analysis
# - Automated decision-making
# - Continuous monitoring and reporting
# 
# **Dependencies:** This notebook builds on models from previous notebooks

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
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import ttest_power
import math

# ML libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Other utilities
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import uuid

# Import our utilities
from ml_utils import ABTestAnalyzer, ModelEvaluator
from config import Config

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
# ## 2. Configuration and Model Loading

# %%
# Configuration
config = Config()

# MLflow configuration
mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment("retail-analytics-ab-testing")

# Load multiple models for A/B testing
try:
    # Load evaluation results to get top models
    evaluation_results = pd.read_csv(f"{config.output_path}/evaluation/model_comparison.csv")
    
    # Select top 3 models for A/B testing
    top_models = evaluation_results.head(3)
    
    models = {}
    for idx, row in top_models.iterrows():
        model_name = row['Model']
        model_path = f"{config.model_path}/{model_name.lower().replace(' ', '_')}_model.pkl"
        try:
            models[model_name] = joblib.load(model_path)
            print(f"✓ Loaded {model_name}")
        except FileNotFoundError:
            print(f"✗ Model {model_name} not found")
    
    # Load feature names and data
    feature_names = pd.read_csv(f"{config.data_path}/feature_names.csv")['feature'].tolist()
    X_test = pd.read_csv(f"{config.data_path}/X_test_processed.csv")
    y_test = pd.read_csv(f"{config.data_path}/y_test.csv").squeeze()
    
    print(f"✓ Loaded {len(models)} models for A/B testing")
    print(f"✓ Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run previous notebooks first")
    raise

# %% [markdown]
# ## 3. A/B Testing Framework Classes

# %%
class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ABTestConfiguration:
    """A/B test configuration"""
    experiment_name: str
    description: str
    control_model: str
    treatment_models: List[str]
    traffic_split: Dict[str, float]
    success_metrics: List[str]
    minimum_sample_size: int
    significance_level: float = 0.05
    statistical_power: float = 0.8
    minimum_effect_size: float = 0.02
    max_duration_days: int = 30
    early_stopping_enabled: bool = True
    early_stopping_criteria: Dict[str, float] = field(default_factory=dict)

@dataclass
class ABTestResult:
    """A/B test result"""
    timestamp: datetime
    variant: str
    user_id: str
    features: Dict[str, Any]
    prediction: Any
    actual_outcome: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TrafficSplitter:
    """Traffic splitting for A/B tests"""
    
    def __init__(self, config: ABTestConfiguration):
        self.config = config
        self.traffic_split = config.traffic_split
        
        # Validate traffic split
        total_traffic = sum(self.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.001:
            raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")
    
    def assign_variant(self, user_id: str) -> str:
        """Assign user to variant based on consistent hashing"""
        # Use consistent hashing for stable assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        hash_ratio = (hash_value % 10000) / 10000
        
        # Assign based on cumulative traffic split
        cumulative = 0
        for variant, traffic in self.traffic_split.items():
            cumulative += traffic
            if hash_ratio <= cumulative:
                return variant
        
        # Fallback to control
        return self.config.control_model
    
    def get_variant_assignment_stats(self, user_ids: List[str]) -> Dict[str, int]:
        """Get variant assignment statistics"""
        assignments = {}
        for user_id in user_ids:
            variant = self.assign_variant(user_id)
            assignments[variant] = assignments.get(variant, 0) + 1
        
        return assignments

class ABTestEngine:
    """A/B testing engine for ML models"""
    
    def __init__(self, config: ABTestConfiguration, models: Dict[str, Any]):
        self.config = config
        self.models = models
        self.traffic_splitter = TrafficSplitter(config)
        self.results = []
        self.status = ExperimentStatus.SETUP
        self.start_time = None
        self.end_time = None
        
        # Initialize analyzer
        self.analyzer = ABTestAnalyzer()
        
        # Validate models
        for model_name in [config.control_model] + config.treatment_models:
            if model_name not in models:
                raise ValueError(f"Model {model_name} not found in provided models")
    
    def start_experiment(self):
        """Start the A/B test experiment"""
        self.status = ExperimentStatus.RUNNING
        self.start_time = datetime.now()
        logger.info(f"Started A/B test: {self.config.experiment_name}")
    
    def pause_experiment(self):
        """Pause the experiment"""
        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused A/B test: {self.config.experiment_name}")
    
    def resume_experiment(self):
        """Resume the experiment"""
        self.status = ExperimentStatus.RUNNING
        logger.info(f"Resumed A/B test: {self.config.experiment_name}")
    
    def stop_experiment(self):
        """Stop the experiment"""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.now()
        logger.info(f"Stopped A/B test: {self.config.experiment_name}")
    
    def get_prediction(self, user_id: str, features: pd.DataFrame) -> ABTestResult:
        """Get prediction for user (with variant assignment)"""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError("Experiment is not running")
        
        # Assign variant
        variant = self.traffic_splitter.assign_variant(user_id)
        
        # Get model prediction
        model = self.models[variant]
        prediction = model.predict(features)[0]
        
        # Get prediction probability if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features)[0]
        
        # Create result
        result = ABTestResult(
            timestamp=datetime.now(),
            variant=variant,
            user_id=user_id,
            features=features.iloc[0].to_dict(),
            prediction=prediction,
            metadata={
                'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
                'model_name': variant
            }
        )
        
        # Store result
        self.results.append(result)
        
        return result
    
    def update_outcome(self, user_id: str, timestamp: datetime, actual_outcome: Any):
        """Update actual outcome for a prediction"""
        # Find matching result
        for result in self.results:
            if (result.user_id == user_id and 
                abs((result.timestamp - timestamp).total_seconds()) < 60):  # Within 1 minute
                result.actual_outcome = actual_outcome
                break
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get current experiment statistics"""
        if not self.results:
            return {'total_predictions': 0, 'variant_distribution': {}}
        
        # Basic stats
        total_predictions = len(self.results)
        
        # Variant distribution
        variant_counts = {}
        for result in self.results:
            variant_counts[result.variant] = variant_counts.get(result.variant, 0) + 1
        
        # Conversion rates (for results with outcomes)
        results_with_outcomes = [r for r in self.results if r.actual_outcome is not None]
        conversion_rates = {}
        
        for variant in variant_counts.keys():
            variant_results = [r for r in results_with_outcomes if r.variant == variant]
            if variant_results:
                conversions = sum(1 for r in variant_results if r.actual_outcome == 1)
                conversion_rates[variant] = conversions / len(variant_results)
        
        return {
            'total_predictions': total_predictions,
            'results_with_outcomes': len(results_with_outcomes),
            'variant_distribution': variant_counts,
            'conversion_rates': conversion_rates,
            'experiment_duration': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
        }
    
    def check_early_stopping(self) -> Dict[str, Any]:
        """Check if early stopping criteria are met"""
        if not self.config.early_stopping_enabled:
            return {'should_stop': False, 'reason': 'Early stopping disabled'}
        
        stats = self.get_experiment_stats()
        
        # Check minimum sample size
        if stats['results_with_outcomes'] < self.config.minimum_sample_size:
            return {'should_stop': False, 'reason': 'Minimum sample size not reached'}
        
        # Check statistical significance
        significance_results = self.analyze_statistical_significance()
        
        if significance_results['overall_significant']:
            return {
                'should_stop': True, 
                'reason': 'Statistical significance achieved',
                'significance_results': significance_results
            }
        
        # Check maximum duration
        if stats['experiment_duration'] > self.config.max_duration_days * 24:
            return {
                'should_stop': True, 
                'reason': 'Maximum duration reached',
                'duration_hours': stats['experiment_duration']
            }
        
        return {'should_stop': False, 'reason': 'Continue experiment'}
    
    def analyze_statistical_significance(self) -> Dict[str, Any]:
        """Analyze statistical significance of results"""
        results_with_outcomes = [r for r in self.results if r.actual_outcome is not None]
        
        if len(results_with_outcomes) < 30:  # Minimum for statistical analysis
            return {'overall_significant': False, 'reason': 'Insufficient data'}
        
        # Group by variant
        variant_data = {}
        for result in results_with_outcomes:
            if result.variant not in variant_data:
                variant_data[result.variant] = []
            variant_data[result.variant].append(result.actual_outcome)
        
        # Compare control vs each treatment
        control_name = self.config.control_model
        control_outcomes = variant_data.get(control_name, [])
        
        if len(control_outcomes) < 15:
            return {'overall_significant': False, 'reason': 'Insufficient control data'}
        
        significance_results = {}
        overall_significant = False
        
        for variant_name, outcomes in variant_data.items():
            if variant_name == control_name or len(outcomes) < 15:
                continue
            
            # Perform statistical test
            if len(set(outcomes)) == 2:  # Binary outcome
                # Chi-square test for proportions
                control_successes = sum(control_outcomes)
                treatment_successes = sum(outcomes)
                
                # Create contingency table
                contingency = [
                    [control_successes, len(control_outcomes) - control_successes],
                    [treatment_successes, len(outcomes) - treatment_successes]
                ]
                
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                significance_results[variant_name] = {
                    'test_type': 'chi_square',
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level,
                    'control_rate': control_successes / len(control_outcomes),
                    'treatment_rate': treatment_successes / len(outcomes),
                    'lift': (treatment_successes / len(outcomes)) - (control_successes / len(control_outcomes))
                }
            else:
                # T-test for continuous outcomes
                t_stat, p_value = ttest_ind(control_outcomes, outcomes)
                
                significance_results[variant_name] = {
                    'test_type': 't_test',
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level,
                    'control_mean': np.mean(control_outcomes),
                    'treatment_mean': np.mean(outcomes),
                    'lift': np.mean(outcomes) - np.mean(control_outcomes)
                }
            
            if significance_results[variant_name]['significant']:
                overall_significant = True
        
        return {
            'overall_significant': overall_significant,
            'variant_results': significance_results,
            'control_variant': control_name,
            'sample_sizes': {k: len(v) for k, v in variant_data.items()}
        }

# %% [markdown]
# ## 4. Experiment Design and Setup

# %%
def design_ab_test(models: Dict[str, Any], 
                   primary_metric: str = 'accuracy',
                   minimum_effect_size: float = 0.02,
                   statistical_power: float = 0.8,
                   significance_level: float = 0.05) -> ABTestConfiguration:
    """Design A/B test configuration"""
    
    # Select control and treatment models
    model_names = list(models.keys())
    control_model = model_names[0]  # Best performing model as control
    treatment_models = model_names[1:3]  # Next best models as treatments
    
    # Calculate sample size
    effect_size = minimum_effect_size
    alpha = significance_level
    power = statistical_power
    
    # Sample size calculation for proportions
    from statsmodels.stats.power import zt_ind_solve_power
    
    sample_size_per_group = zt_ind_solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        alternative='two-sided'
    )
    
    # Total sample size
    total_sample_size = int(sample_size_per_group * len(model_names))
    
    # Traffic split (equal allocation)
    traffic_split = {model: 1.0 / len(model_names) for model in model_names}
    
    config = ABTestConfiguration(
        experiment_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description=f"A/B test comparing {len(model_names)} models for {primary_metric}",
        control_model=control_model,
        treatment_models=treatment_models,
        traffic_split=traffic_split,
        success_metrics=[primary_metric, 'precision', 'recall', 'f1_score'],
        minimum_sample_size=total_sample_size,
        significance_level=significance_level,
        statistical_power=statistical_power,
        minimum_effect_size=minimum_effect_size,
        max_duration_days=14,
        early_stopping_enabled=True,
        early_stopping_criteria={
            'significance_threshold': significance_level,
            'minimum_runtime_hours': 24
        }
    )
    
    print(f"A/B Test Configuration:")
    print(f"  Control Model: {control_model}")
    print(f"  Treatment Models: {treatment_models}")
    print(f"  Traffic Split: {traffic_split}")
    print(f"  Minimum Sample Size: {total_sample_size}")
    print(f"  Expected Effect Size: {effect_size}")
    print(f"  Statistical Power: {power}")
    print(f"  Significance Level: {significance_level}")
    
    return config

# Design the A/B test
ab_config = design_ab_test(models, primary_metric='accuracy')

# Create A/B test engine
ab_engine = ABTestEngine(ab_config, models)

print("✓ A/B test engine initialized")

# %% [markdown]
# ## 5. Simulation of A/B Test Execution

# %%
def simulate_ab_test(engine: ABTestEngine, 
                    test_data: pd.DataFrame,
                    test_labels: pd.Series,
                    duration_minutes: int = 5) -> Dict[str, Any]:
    """Simulate A/B test execution"""
    
    print(f"Starting A/B test simulation for {duration_minutes} minutes...")
    
    # Start experiment
    engine.start_experiment()
    
    # Simulation parameters
    requests_per_minute = 50
    total_requests = duration_minutes * requests_per_minute
    
    # Generate synthetic user IDs
    user_ids = [f"user_{i:06d}" for i in range(total_requests)]
    
    # Simulate requests
    for i, user_id in enumerate(user_ids):
        # Select random sample from test data
        sample_idx = np.random.choice(len(test_data))
        features = test_data.iloc[[sample_idx]]
        true_label = test_labels.iloc[sample_idx]
        
        # Get prediction
        result = engine.get_prediction(user_id, features)
        
        # Simulate outcome with some delay (immediate for simulation)
        # In real scenario, this would come later
        engine.update_outcome(user_id, result.timestamp, true_label)
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_requests} requests")
            
            # Check early stopping
            early_stop = engine.check_early_stopping()
            if early_stop['should_stop']:
                print(f"  Early stopping triggered: {early_stop['reason']}")
                break
    
    # Stop experiment
    engine.stop_experiment()
    
    # Get final statistics
    final_stats = engine.get_experiment_stats()
    significance_results = engine.analyze_statistical_significance()
    
    print(f"✓ A/B test simulation completed")
    print(f"  Total predictions: {final_stats['total_predictions']}")
    print(f"  Results with outcomes: {final_stats['results_with_outcomes']}")
    print(f"  Experiment duration: {final_stats['experiment_duration']:.2f} hours")
    
    return {
        'final_stats': final_stats,
        'significance_results': significance_results,
        'experiment_config': ab_config,
        'total_results': len(engine.results)
    }

# Run A/B test simulation
simulation_results = simulate_ab_test(ab_engine, X_test, y_test, duration_minutes=3)

# %% [markdown]
# ## 6. Statistical Analysis and Results

# %%
def analyze_ab_test_results(engine: ABTestEngine) -> Dict[str, Any]:
    """Comprehensive analysis of A/B test results"""
    
    print("Analyzing A/B test results...")
    
    # Get results with outcomes
    results_with_outcomes = [r for r in engine.results if r.actual_outcome is not None]
    
    if len(results_with_outcomes) < 30:
        return {'error': 'Insufficient data for analysis'}
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'variant': r.variant,
            'prediction': r.prediction,
            'actual_outcome': r.actual_outcome,
            'correct': r.prediction == r.actual_outcome,
            'timestamp': r.timestamp,
            'user_id': r.user_id
        }
        for r in results_with_outcomes
    ])
    
    # Calculate metrics by variant
    variant_metrics = {}
    for variant in results_df['variant'].unique():
        variant_data = results_df[results_df['variant'] == variant]
        
        metrics = {
            'sample_size': len(variant_data),
            'accuracy': variant_data['correct'].mean(),
            'precision': precision_score(variant_data['actual_outcome'], variant_data['prediction'], average='weighted', zero_division=0),
            'recall': recall_score(variant_data['actual_outcome'], variant_data['prediction'], average='weighted', zero_division=0),
            'f1_score': f1_score(variant_data['actual_outcome'], variant_data['prediction'], average='weighted', zero_division=0)
        }
        
        # Calculate confidence intervals
        accuracy_ci = calculate_confidence_interval(variant_data['correct'])
        metrics['accuracy_ci'] = accuracy_ci
        
        variant_metrics[variant] = metrics
    
    # Statistical significance testing
    significance_results = engine.analyze_statistical_significance()
    
    # Effect size calculations
    control_name = engine.config.control_model
    control_accuracy = variant_metrics[control_name]['accuracy']
    
    effect_sizes = {}
    for variant, metrics in variant_metrics.items():
        if variant != control_name:
            # Cohen's d for effect size
            control_data = results_df[results_df['variant'] == control_name]['correct']
            treatment_data = results_df[results_df['variant'] == variant]['correct']
            
            pooled_std = np.sqrt(((len(control_data) - 1) * control_data.var() + 
                                 (len(treatment_data) - 1) * treatment_data.var()) / 
                                (len(control_data) + len(treatment_data) - 2))
            
            cohens_d = (treatment_data.mean() - control_data.mean()) / pooled_std
            effect_sizes[variant] = cohens_d
    
    # Power analysis
    power_analysis = {}
    for variant in variant_metrics.keys():
        if variant != control_name:
            observed_effect = variant_metrics[variant]['accuracy'] - control_accuracy
            power = calculate_statistical_power(
                effect_size=observed_effect,
                sample_size=variant_metrics[variant]['sample_size'],
                alpha=engine.config.significance_level
            )
            power_analysis[variant] = power
    
    return {
        'variant_metrics': variant_metrics,
        'significance_results': significance_results,
        'effect_sizes': effect_sizes,
        'power_analysis': power_analysis,
        'control_variant': control_name,
        'total_sample_size': len(results_df)
    }

def calculate_confidence_interval(data: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for binary data"""
    n = len(data)
    mean = data.mean()
    se = np.sqrt(mean * (1 - mean) / n)
    
    # Z-score for confidence level
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    ci_lower = mean - z_score * se
    ci_upper = mean + z_score * se
    
    return (ci_lower, ci_upper)

def calculate_statistical_power(effect_size: float, sample_size: int, alpha: float = 0.05) -> float:
    """Calculate statistical power for observed effect"""
    try:
        power = ttest_power(effect_size, sample_size, alpha)
        return power
    except:
        return 0.0

# Analyze results
analysis_results = analyze_ab_test_results(ab_engine)

# Print analysis summary
if 'error' not in analysis_results:
    print("\nA/B TEST ANALYSIS SUMMARY")
    print("=" * 40)
    
    print("\nVariant Performance:")
    for variant, metrics in analysis_results['variant_metrics'].items():
        print(f"\n{variant}:")
        print(f"  Sample Size: {metrics['sample_size']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f} [{metrics['accuracy_ci'][0]:.4f}, {metrics['accuracy_ci'][1]:.4f}]")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    print("\nStatistical Significance:")
    sig_results = analysis_results['significance_results']
    if sig_results['overall_significant']:
        print("  Overall: SIGNIFICANT")
        for variant, result in sig_results['variant_results'].items():
            print(f"  {variant}: p-value = {result['p_value']:.4f}, lift = {result['lift']:.4f}")
    else:
        print("  Overall: NOT SIGNIFICANT")
    
    print("\nEffect Sizes (Cohen's d):")
    for variant, effect_size in analysis_results['effect_sizes'].items():
        print(f"  {variant}: {effect_size:.4f}")
    
    print("\nPower Analysis:")
    for variant, power in analysis_results['power_analysis'].items():
        print(f"  {variant}: {power:.4f}")

# %% [markdown]
# ## 7. Visualization and Reporting

# %%
def create_ab_test_dashboard(engine: ABTestEngine, analysis: Dict[str, Any]):
    """Create comprehensive A/B test dashboard"""
    
    if 'error' in analysis:
        print("Cannot create dashboard: insufficient data")
        return
    
    # Create results DataFrame
    results_with_outcomes = [r for r in engine.results if r.actual_outcome is not None]
    results_df = pd.DataFrame([
        {
            'variant': r.variant,
            'prediction': r.prediction,
            'actual_outcome': r.actual_outcome,
            'correct': r.prediction == r.actual_outcome,
            'timestamp': r.timestamp
        }
        for r in results_with_outcomes
    ])
    
    # Create dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Accuracy by Variant', 'Sample Size Distribution',
            'Accuracy Over Time', 'Confidence Intervals',
            'Effect Sizes', 'P-Values'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Accuracy by variant
    variant_metrics = analysis['variant_metrics']
    variants = list(variant_metrics.keys())
    accuracies = [variant_metrics[v]['accuracy'] for v in variants]
    
    fig.add_trace(
        go.Bar(
            x=variants,
            y=accuracies,
            name='Accuracy',
            marker_color=['red' if v == engine.config.control_model else 'blue' for v in variants]
        ),
        row=1, col=1
    )
    
    # 2. Sample size distribution
    sample_sizes = [variant_metrics[v]['sample_size'] for v in variants]
    
    fig.add_trace(
        go.Bar(
            x=variants,
            y=sample_sizes,
            name='Sample Size',
            marker_color='green'
        ),
        row=1, col=2
    )
    
    # 3. Accuracy over time
    results_df['hour'] = results_df['timestamp'].dt.floor('H')
    hourly_accuracy = results_df.groupby(['hour', 'variant'])['correct'].mean().reset_index()
    
    for variant in variants:
        variant_data = hourly_accuracy[hourly_accuracy['variant'] == variant]
        fig.add_trace(
            go.Scatter(
                x=variant_data['hour'],
                y=variant_data['correct'],
                mode='lines+markers',
                name=f'{variant} (time)',
                line=dict(dash='dash' if variant == engine.config.control_model else 'solid')
            ),
            row=2, col=1
        )
    
    # 4. Confidence intervals
    ci_lower = [variant_metrics[v]['accuracy_ci'][0] for v in variants]
    ci_upper = [variant_metrics[v]['accuracy_ci'][1] for v in variants]
    
    fig.add_trace(
        go.Scatter(
            x=variants,
            y=accuracies,
            mode='markers',
            name='Accuracy',
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_upper[i] - accuracies[i] for i in range(len(accuracies))],
                arrayminus=[accuracies[i] - ci_lower[i] for i in range(len(accuracies))]
            ),
            marker=dict(size=10, color='orange')
        ),
        row=2, col=2
    )
    
    # 5. Effect sizes
    effect_sizes = analysis['effect_sizes']
    if effect_sizes:
        effect_variants = list(effect_sizes.keys())
        effect_values = list(effect_sizes.values())
        
        fig.add_trace(
            go.Bar(
                x=effect_variants,
                y=effect_values,
                name='Effect Size (Cohen\'s d)',
                marker_color='purple'
            ),
            row=3, col=1
        )
    
    # 6. P-values
    sig_results = analysis['significance_results']
    if sig_results['overall_significant']:
        p_variants = list(sig_results['variant_results'].keys())
        p_values = [sig_results['variant_results'][v]['p_value'] for v in p_variants]
        
        fig.add_trace(
            go.Bar(
                x=p_variants,
                y=p_values,
                name='P-Value',
                marker_color=['green' if p < 0.05 else 'red' for p in p_values]
            ),
            row=3, col=2
        )
        
        # Add significance line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="A/B Test Results Dashboard",
        showlegend=True
    )
    
    # Update axes
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Sample Size", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)
    fig.update_yaxes(title_text="Effect Size", row=3, col=1)
    fig.update_yaxes(title_text="P-Value", row=3, col=2)
    
    fig.show()
    
    print("✓ A/B test dashboard created")

# Create dashboard
create_ab_test_dashboard(ab_engine, analysis_results)

# %% [markdown]
# ## 8. Decision Making Framework

# %%
class ABTestDecisionMaker:
    """Automated decision making for A/B tests"""
    
    def __init__(self, config: ABTestConfiguration):
        self.config = config
        self.decision_criteria = {
            'significance_threshold': config.significance_level,
            'minimum_effect_size': config.minimum_effect_size,
            'minimum_power': config.statistical_power,
            'minimum_sample_size': config.minimum_sample_size
        }
    
    def make_decision(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Make deployment decision based on A/B test results"""
        
        if 'error' in analysis_results:
            return {
                'decision': 'continue_experiment',
                'reason': 'Insufficient data for decision making',
                'confidence': 0.0
            }
        
        # Extract key metrics
        variant_metrics = analysis_results['variant_metrics']
        significance_results = analysis_results['significance_results']
        effect_sizes = analysis_results['effect_sizes']
        power_analysis = analysis_results['power_analysis']
        
        control_name = self.config.control_model
        control_accuracy = variant_metrics[control_name]['accuracy']
        
        # Evaluate each treatment
        treatment_evaluations = {}
        
        for variant in self.config.treatment_models:
            if variant in variant_metrics:
                evaluation = self._evaluate_treatment(
                    variant, variant_metrics, significance_results, 
                    effect_sizes, power_analysis, control_accuracy
                )
                treatment_evaluations[variant] = evaluation
        
        # Make final decision
        decision = self._make_final_decision(treatment_evaluations, control_accuracy)
        
        return decision
    
    def _evaluate_treatment(self, variant: str, variant_metrics: Dict, 
                           significance_results: Dict, effect_sizes: Dict,
                           power_analysis: Dict, control_accuracy: float) -> Dict[str, Any]:
        """Evaluate a single treatment variant"""
        
        treatment_accuracy = variant_metrics[variant]['accuracy']
        sample_size = variant_metrics[variant]['sample_size']
        
        # Check significance
        is_significant = False
        p_value = 1.0
        
        if significance_results['overall_significant'] and variant in significance_results['variant_results']:
            var_result = significance_results['variant_results'][variant]
            is_significant = var_result['significant']
            p_value = var_result['p_value']
        
        # Check effect size
        effect_size = effect_sizes.get(variant, 0.0)
        meaningful_effect = abs(effect_size) >= self.decision_criteria['minimum_effect_size']
        
        # Check power
        power = power_analysis.get(variant, 0.0)
        adequate_power = power >= self.decision_criteria['minimum_power']
        
        # Check sample size
        adequate_sample = sample_size >= self.decision_criteria['minimum_sample_size']
        
        # Calculate improvement
        improvement = treatment_accuracy - control_accuracy
        improvement_pct = (improvement / control_accuracy) * 100
        
        # Overall assessment
        meets_criteria = (
            is_significant and 
            meaningful_effect and 
            adequate_power and 
            adequate_sample and
            improvement > 0
        )
        
        return {
            'variant': variant,
            'accuracy': treatment_accuracy,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'is_significant': is_significant,
            'p_value': p_value,
            'effect_size': effect_size,
            'meaningful_effect': meaningful_effect,
            'power': power,
            'adequate_power': adequate_power,
            'sample_size': sample_size,
            'adequate_sample': adequate_sample,
            'meets_criteria': meets_criteria,
            'confidence_score': self._calculate_confidence_score(
                is_significant, meaningful_effect, adequate_power, adequate_sample
            )
        }
    
    def _calculate_confidence_score(self, significant: bool, meaningful: bool, 
                                   adequate_power: bool, adequate_sample: bool) -> float:
        """Calculate confidence score for decision"""
        score = 0.0
        
        if significant:
            score += 0.4
        if meaningful:
            score += 0.3
        if adequate_power:
            score += 0.2
        if adequate_sample:
            score += 0.1
        
        return score
    
    def _make_final_decision(self, treatment_evaluations: Dict, control_accuracy: float) -> Dict[str, Any]:
        """Make final deployment decision"""
        
        # Find best treatment
        best_treatment = None
        best_score = 0.0
        
        for variant, evaluation in treatment_evaluations.items():
            if evaluation['meets_criteria'] and evaluation['confidence_score'] > best_score:
                best_treatment = variant
                best_score = evaluation['confidence_score']
        
        if best_treatment:
            return {
                'decision': 'deploy_treatment',
                'recommended_variant': best_treatment,
                'reason': f"Treatment {best_treatment} shows significant improvement",
                'confidence': best_score,
                'evaluation': treatment_evaluations[best_treatment],
                'all_evaluations': treatment_evaluations
            }
        
        # Check if any treatment shows promise but needs more data
        promising_treatments = [
            v for v, eval in treatment_evaluations.items()
            if eval['improvement'] > 0 and eval['confidence_score'] > 0.5
        ]
        
        if promising_treatments:
            return {
                'decision': 'continue_experiment',
                'reason': f"Promising treatments ({promising_treatments}) need more data",
                'confidence': 0.6,
                'promising_treatments': promising_treatments,
                'all_evaluations': treatment_evaluations
            }
        
        # No significant improvement found
        return {
            'decision': 'keep_control',
            'reason': "No treatment shows significant improvement over control",
            'confidence': 0.8,
            'all_evaluations': treatment_evaluations
        }

# Make decision
decision_maker = ABTestDecisionMaker(ab_config)
decision = decision_maker.make_decision(analysis_results)

print("\nA/B TEST DECISION")
print("=" * 25)
print(f"Decision: {decision['decision'].upper()}")
print(f"Reason: {decision['reason']}")
print(f"Confidence: {decision['confidence']:.2f}")

if decision['decision'] == 'deploy_treatment':
    print(f"Recommended Variant: {decision['recommended_variant']}")
    eval_details = decision['evaluation']
    print(f"Improvement: {eval_details['improvement_pct']:.2f}%")
    print(f"Statistical Significance: {eval_details['is_significant']}")
    print(f"P-value: {eval_details['p_value']:.4f}")

# %% [markdown]
# ## 9. Continuous Monitoring and Reporting

# %%
class ABTestMonitor:
    """Continuous monitoring for A/B tests"""
    
    def __init__(self, engine: ABTestEngine):
        self.engine = engine
        self.monitoring_history = []
        self.alerts = []
    
    def check_experiment_health(self) -> Dict[str, Any]:
        """Check overall experiment health"""
        stats = self.engine.get_experiment_stats()
        
        # Check for issues
        issues = []
        
        # Check sample size distribution
        variant_counts = stats['variant_distribution']
        total_samples = sum(variant_counts.values())
        
        if total_samples > 0:
            for variant, count in variant_counts.items():
                expected_ratio = self.engine.config.traffic_split[variant]
                actual_ratio = count / total_samples
                
                if abs(actual_ratio - expected_ratio) > 0.1:  # 10% deviation
                    issues.append(f"Traffic split imbalance for {variant}: expected {expected_ratio:.2f}, actual {actual_ratio:.2f}")
        
        # Check conversion rates
        conversion_rates = stats['conversion_rates']
        if conversion_rates:
            # Look for unusual patterns
            rates = list(conversion_rates.values())
            if len(rates) > 1:
                rate_std = np.std(rates)
                if rate_std > 0.2:  # High variance in conversion rates
                    issues.append(f"High variance in conversion rates: {rate_std:.3f}")
        
        # Check experiment duration
        duration_hours = stats['experiment_duration']
        if duration_hours > self.engine.config.max_duration_days * 24:
            issues.append(f"Experiment running too long: {duration_hours:.1f} hours")
        
        health_score = max(0, 1 - len(issues) * 0.2)
        
        return {
            'health_score': health_score,
            'issues': issues,
            'stats': stats,
            'timestamp': datetime.now()
        }
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report"""
        health = self.check_experiment_health()
        stats = health['stats']
        
        report = f"""
A/B TEST MONITORING REPORT
=========================

Experiment: {self.engine.config.experiment_name}
Status: {self.engine.status.value}
Duration: {stats['experiment_duration']:.2f} hours

SAMPLE STATISTICS
-----------------
Total Predictions: {stats['total_predictions']}
Results with Outcomes: {stats['results_with_outcomes']}

Traffic Distribution:
{json.dumps(stats['variant_distribution'], indent=2)}

Conversion Rates:
{json.dumps(stats['conversion_rates'], indent=2)}

HEALTH CHECK
-----------
Health Score: {health['health_score']:.2f}
Issues: {len(health['issues'])}
"""
        
        if health['issues']:
            report += "\nIssues Detected:\n"
            for issue in health['issues']:
                report += f"- {issue}\n"
        
        # Add early stopping check
        early_stop = self.engine.check_early_stopping()
        report += f"\nEARLY STOPPING CHECK\n"
        report += f"Should Stop: {early_stop['should_stop']}\n"
        report += f"Reason: {early_stop['reason']}\n"
        
        return report
    
    def create_monitoring_alerts(self) -> List[Dict[str, Any]]:
        """Create monitoring alerts"""
        alerts = []
        health = self.check_experiment_health()
        
        # Health score alert
        if health['health_score'] < 0.7:
            alerts.append({
                'type': 'health_warning',
                'severity': 'medium',
                'message': f"Experiment health score low: {health['health_score']:.2f}",
                'timestamp': datetime.now()
            })
        
        # Issue alerts
        for issue in health['issues']:
            alerts.append({
                'type': 'experiment_issue',
                'severity': 'high',
                'message': issue,
                'timestamp': datetime.now()
            })
        
        # Early stopping alert
        early_stop = self.engine.check_early_stopping()
        if early_stop['should_stop']:
            alerts.append({
                'type': 'early_stopping',
                'severity': 'high',
                'message': f"Early stopping triggered: {early_stop['reason']}",
                'timestamp': datetime.now()
            })
        
        return alerts

# Initialize monitor
ab_monitor = ABTestMonitor(ab_engine)

# Generate monitoring report
monitoring_report = ab_monitor.generate_monitoring_report()
print(monitoring_report)

# Check for alerts
alerts = ab_monitor.create_monitoring_alerts()
if alerts:
    print("\nALERTS DETECTED:")
    for alert in alerts:
        print(f"  [{alert['severity'].upper()}] {alert['message']}")
else:
    print("\n✓ No alerts detected")

# %% [markdown]
# ## 10. Export Results and Configuration

# %%
def export_ab_test_results(engine: ABTestEngine, analysis: Dict[str, Any], 
                          decision: Dict[str, Any]) -> str:
    """Export A/B test results and configuration"""
    
    # Create output directory
    output_dir = f"{config.output_path}/ab_testing"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export configuration
    config_data = {
        'experiment_name': engine.config.experiment_name,
        'description': engine.config.description,
        'control_model': engine.config.control_model,
        'treatment_models': engine.config.treatment_models,
        'traffic_split': engine.config.traffic_split,
        'success_metrics': engine.config.success_metrics,
        'minimum_sample_size': engine.config.minimum_sample_size,
        'significance_level': engine.config.significance_level,
        'statistical_power': engine.config.statistical_power,
        'minimum_effect_size': engine.config.minimum_effect_size,
        'max_duration_days': engine.config.max_duration_days,
        'early_stopping_enabled': engine.config.early_stopping_enabled
    }
    
    with open(f"{output_dir}/experiment_config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Export results
    results_data = []
    for result in engine.results:
        results_data.append({
            'timestamp': result.timestamp.isoformat(),
            'variant': result.variant,
            'user_id': result.user_id,
            'prediction': result.prediction,
            'actual_outcome': result.actual_outcome,
            'metadata': result.metadata
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f"{output_dir}/experiment_results.csv", index=False)
    
    # Export analysis
    with open(f"{output_dir}/analysis_results.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Export decision
    with open(f"{output_dir}/decision.json", 'w') as f:
        json.dump(decision, f, indent=2, default=str)
    
    # Export summary report
    summary_report = f"""
A/B TEST EXPERIMENT SUMMARY
==========================

Experiment: {engine.config.experiment_name}
Start Time: {engine.start_time.isoformat() if engine.start_time else 'N/A'}
End Time: {engine.end_time.isoformat() if engine.end_time else 'N/A'}
Status: {engine.status.value}

CONFIGURATION
------------
Control Model: {engine.config.control_model}
Treatment Models: {engine.config.treatment_models}
Traffic Split: {engine.config.traffic_split}
Significance Level: {engine.config.significance_level}
Statistical Power: {engine.config.statistical_power}
Minimum Effect Size: {engine.config.minimum_effect_size}

RESULTS
-------
Total Predictions: {len(engine.results)}
Results with Outcomes: {len([r for r in engine.results if r.actual_outcome is not None])}

ANALYSIS SUMMARY
---------------
Statistical Significance: {analysis.get('significance_results', {}).get('overall_significant', False)}
Best Treatment: {decision.get('recommended_variant', 'None')}

DECISION
--------
Decision: {decision['decision']}
Reason: {decision['reason']}
Confidence: {decision['confidence']:.2f}
"""
    
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write(summary_report)
    
    print(f"✓ A/B test results exported to: {output_dir}")
    print(f"  Files created:")
    print(f"    - experiment_config.json")
    print(f"    - experiment_results.csv")
    print(f"    - analysis_results.json")
    print(f"    - decision.json")
    print(f"    - summary_report.txt")
    
    return output_dir

# Export results
export_dir = export_ab_test_results(ab_engine, analysis_results, decision)

# %% [markdown]
# ## 11. MLflow Integration

# %%
# Log A/B test to MLflow
try:
    with mlflow.start_run(run_name="ab_test_experiment"):
        # Log configuration
        mlflow.log_params({
            'experiment_name': ab_config.experiment_name,
            'control_model': ab_config.control_model,
            'treatment_models': ','.join(ab_config.treatment_models),
            'significance_level': ab_config.significance_level,
            'statistical_power': ab_config.statistical_power,
            'minimum_effect_size': ab_config.minimum_effect_size,
            'total_variants': len(ab_config.traffic_split)
        })
        
        # Log results
        final_stats = ab_engine.get_experiment_stats()
        mlflow.log_metrics({
            'total_predictions': final_stats['total_predictions'],
            'results_with_outcomes': final_stats['results_with_outcomes'],
            'experiment_duration_hours': final_stats['experiment_duration']
        })
        
        # Log analysis metrics
        if 'error' not in analysis_results:
            for variant, metrics in analysis_results['variant_metrics'].items():
                mlflow.log_metrics({
                    f'{variant}_accuracy': metrics['accuracy'],
                    f'{variant}_sample_size': metrics['sample_size'],
                    f'{variant}_precision': metrics['precision'],
                    f'{variant}_recall': metrics['recall'],
                    f'{variant}_f1_score': metrics['f1_score']
                })
        
        # Log decision
        mlflow.log_params({
            'final_decision': decision['decision'],
            'decision_confidence': decision['confidence']
        })
        
        if decision['decision'] == 'deploy_treatment':
            mlflow.log_params({
                'recommended_variant': decision['recommended_variant']
            })
        
        # Log artifacts
        mlflow.log_artifacts(export_dir, "ab_test_artifacts")
        
        print("✓ A/B test logged to MLflow")
        
except Exception as e:
    print(f"MLflow logging error: {e}")

# %% [markdown]
# ## 12. A/B Testing Framework Summary

# %%
print("🎯 A/B TESTING FRAMEWORK COMPLETE")
print("=" * 40)

print(f"\n📊 EXPERIMENT RESULTS:")
print(f"  Experiment: {ab_config.experiment_name}")
print(f"  Models tested: {len(models)}")
print(f"  Total predictions: {len(ab_engine.results)}")
print(f"  Duration: {ab_engine.get_experiment_stats()['experiment_duration']:.2f} hours")

print(f"\n🔬 ANALYSIS RESULTS:")
if 'error' not in analysis_results:
    print(f"  Statistical significance: {analysis_results['significance_results']['overall_significant']}")
    control_name = ab_config.control_model
    control_accuracy = analysis_results['variant_metrics'][control_name]['accuracy']
    print(f"  Control accuracy: {control_accuracy:.4f}")
    
    for variant, metrics in analysis_results['variant_metrics'].items():
        if variant != control_name:
            improvement = metrics['accuracy'] - control_accuracy
            print(f"  {variant} improvement: {improvement:.4f} ({improvement/control_accuracy*100:.2f}%)")

print(f"\n🎯 DECISION:")
print(f"  Decision: {decision['decision']}")
print(f"  Confidence: {decision['confidence']:.2f}")
print(f"  Reason: {decision['reason']}")

print(f"\n🔧 FRAMEWORK COMPONENTS:")
print(f"  ✅ Experiment design and configuration")
print(f"  ✅ Traffic splitting with consistent hashing")
print(f"  ✅ Statistical significance testing")
print(f"  ✅ Effect size analysis")
print(f"  ✅ Automated decision making")
print(f"  ✅ Continuous monitoring")
print(f"  ✅ Comprehensive reporting")
print(f"  ✅ MLflow integration")

print(f"\n📁 OUTPUTS:")
print(f"  Directory: {export_dir}")
print(f"  Configuration: experiment_config.json")
print(f"  Results: experiment_results.csv")
print(f"  Analysis: analysis_results.json")
print(f"  Decision: decision.json")
print(f"  Summary: summary_report.txt")

print(f"\n🚀 NEXT STEPS:")
print(f"  1. Review analysis results and decision")
print(f"  2. Implement recommended model deployment")
print(f"  3. Set up continuous A/B testing pipeline")
print(f"  4. Monitor deployed model performance")
print(f"  5. Proceed to 07-automated-retraining.py")

print(f"\n✅ A/B testing framework ready for production use!")
print("   Framework supports multiple models, statistical rigor, and automated decisions")

# %%
# Final A/B test summary
ab_test_summary = {
    'completion_time': datetime.now().isoformat(),
    'experiment_config': ab_config.experiment_name,
    'models_tested': len(models),
    'total_predictions': len(ab_engine.results),
    'analysis_completed': 'error' not in analysis_results,
    'decision_made': decision['decision'],
    'decision_confidence': decision['confidence'],
    'framework_validated': True,
    'ready_for_production': True
}

# Save final summary
with open(f"{export_dir}/ab_test_summary.json", 'w') as f:
    json.dump(ab_test_summary, f, indent=2)

print(f"📋 A/B test summary saved to: {export_dir}/ab_test_summary.json") 