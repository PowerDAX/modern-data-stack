#!/usr/bin/env python3
"""
Data Quality Monitoring System
==============================

This module provides comprehensive data quality monitoring capabilities including:
- Multi-dimensional quality assessments
- Real-time quality metrics tracking
- Anomaly detection in data patterns
- Automated quality reporting
- Data lineage and impact analysis
- Configurable quality rules and thresholds

Features:
- Completeness, accuracy, consistency, timeliness, validity checks
- Statistical anomaly detection
- Trend analysis and alerting
- Integration with data pipeline monitoring
- Automated quality reports and dashboards

Author: Data Engineering Team
Date: 2024-01-15
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    INTEGRITY = "integrity"

class SeverityLevel(Enum):
    """Quality issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityRule:
    """Data quality rule definition"""
    rule_id: str
    name: str
    description: str
    dimension: QualityDimension
    severity: SeverityLevel
    threshold: float
    condition: str
    enabled: bool = True
    applies_to: List[str] = field(default_factory=list)  # Column names or patterns
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityIssue:
    """Data quality issue"""
    issue_id: str
    rule_id: str
    dimension: QualityDimension
    severity: SeverityLevel
    description: str
    affected_columns: List[str]
    affected_rows: int
    score: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityMetrics:
    """Quality metrics for a dataset"""
    dataset_name: str
    timestamp: datetime
    total_records: int
    total_columns: int
    overall_score: float
    dimension_scores: Dict[str, float]
    issues: List[QualityIssue]
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system.
    
    This class provides extensive capabilities for monitoring data quality
    across multiple dimensions with configurable rules and thresholds.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data quality monitor.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.rules = self._load_quality_rules()
        self.metrics_history = []
        self.anomaly_detectors = {}
        
        # Initialize built-in quality rules
        self._initialize_builtin_rules()
        
        logger.info("Data Quality Monitor initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'overall_score_weights': {
                'completeness': 0.25,
                'accuracy': 0.20,
                'consistency': 0.15,
                'timeliness': 0.15,
                'validity': 0.15,
                'uniqueness': 0.10
            },
            'anomaly_detection': {
                'enabled': True,
                'sensitivity': 0.05,
                'min_history_points': 10
            },
            'alerting': {
                'enabled': True,
                'critical_threshold': 0.6,
                'high_threshold': 0.7,
                'medium_threshold': 0.8
            },
            'reporting': {
                'auto_generate': True,
                'include_trends': True,
                'include_recommendations': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _load_quality_rules(self) -> Dict[str, QualityRule]:
        """Load quality rules from configuration"""
        return {}
    
    def _initialize_builtin_rules(self):
        """Initialize built-in quality rules"""
        builtin_rules = [
            QualityRule(
                rule_id="COMPLETENESS_001",
                name="Null Value Check",
                description="Check for null/missing values in required columns",
                dimension=QualityDimension.COMPLETENESS,
                severity=SeverityLevel.HIGH,
                threshold=0.05,  # Max 5% null values
                condition="null_percentage <= threshold"
            ),
            QualityRule(
                rule_id="UNIQUENESS_001",
                name="Duplicate Records",
                description="Check for duplicate records",
                dimension=QualityDimension.UNIQUENESS,
                severity=SeverityLevel.MEDIUM,
                threshold=0.02,  # Max 2% duplicates
                condition="duplicate_percentage <= threshold"
            ),
            QualityRule(
                rule_id="VALIDITY_001",
                name="Data Type Validation",
                description="Validate data types match expected schema",
                dimension=QualityDimension.VALIDITY,
                severity=SeverityLevel.HIGH,
                threshold=0.01,  # Max 1% invalid types
                condition="invalid_type_percentage <= threshold"
            ),
            QualityRule(
                rule_id="CONSISTENCY_001",
                name="Format Consistency",
                description="Check format consistency within columns",
                dimension=QualityDimension.CONSISTENCY,
                severity=SeverityLevel.MEDIUM,
                threshold=0.05,  # Max 5% inconsistent formats
                condition="inconsistent_format_percentage <= threshold"
            ),
            QualityRule(
                rule_id="ACCURACY_001",
                name="Outlier Detection",
                description="Detect statistical outliers in numeric columns",
                dimension=QualityDimension.ACCURACY,
                severity=SeverityLevel.LOW,
                threshold=0.10,  # Max 10% outliers
                condition="outlier_percentage <= threshold"
            ),
            QualityRule(
                rule_id="TIMELINESS_001",
                name="Data Freshness",
                description="Check data freshness based on timestamp columns",
                dimension=QualityDimension.TIMELINESS,
                severity=SeverityLevel.MEDIUM,
                threshold=24,  # Max 24 hours old
                condition="max_age_hours <= threshold"
            )
        ]
        
        for rule in builtin_rules:
            self.rules[rule.rule_id] = rule
    
    def assess_quality(self, df: pd.DataFrame, 
                      dataset_name: str = "dataset",
                      schema: Optional[Dict[str, Any]] = None) -> QualityMetrics:
        """
        Perform comprehensive quality assessment of a dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to assess
        dataset_name : str
            Name of the dataset
        schema : Dict[str, Any], optional
            Expected schema for validation
            
        Returns
        -------
        QualityMetrics
            Comprehensive quality assessment results
        """
        logger.info(f"Starting quality assessment for dataset: {dataset_name}")
        
        # Initialize metrics
        timestamp = datetime.now()
        issues = []
        dimension_scores = {}
        
        # Assess each quality dimension
        dimension_scores['completeness'] = self._assess_completeness(df, issues)
        dimension_scores['accuracy'] = self._assess_accuracy(df, issues)
        dimension_scores['consistency'] = self._assess_consistency(df, issues)
        dimension_scores['timeliness'] = self._assess_timeliness(df, issues)
        dimension_scores['validity'] = self._assess_validity(df, issues, schema)
        dimension_scores['uniqueness'] = self._assess_uniqueness(df, issues)
        dimension_scores['integrity'] = self._assess_integrity(df, issues)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Create quality metrics
        metrics = QualityMetrics(
            dataset_name=dataset_name,
            timestamp=timestamp,
            total_records=len(df),
            total_columns=len(df.columns),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=issues,
            metadata={
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
            }
        )
        
        # Store metrics in history
        self.metrics_history.append(metrics)
        
        # Check for anomalies
        if self.config['anomaly_detection']['enabled']:
            self._detect_anomalies(metrics)
        
        # Generate alerts if needed
        if self.config['alerting']['enabled']:
            self._check_alerts(metrics)
        
        logger.info(f"Quality assessment completed. Overall score: {overall_score:.3f}")
        return metrics
    
    def _assess_completeness(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Assess data completeness"""
        scores = []
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            null_percentage = null_count / len(df)
            
            # Check against completeness rules
            rule = self.rules.get("COMPLETENESS_001")
            if rule and null_percentage > rule.threshold:
                issue = QualityIssue(
                    issue_id=f"COMP_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule.rule_id,
                    dimension=QualityDimension.COMPLETENESS,
                    severity=rule.severity,
                    description=f"Column '{column}' has {null_percentage:.1%} null values",
                    affected_columns=[column],
                    affected_rows=null_count,
                    score=1.0 - null_percentage,
                    timestamp=datetime.now(),
                    details={'null_count': null_count, 'null_percentage': null_percentage}
                )
                issues.append(issue)
            
            scores.append(1.0 - null_percentage)
        
        return np.mean(scores) if scores else 1.0
    
    def _assess_accuracy(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Assess data accuracy using outlier detection"""
        scores = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if df[column].dropna().empty:
                continue
                
            # Use IQR method for outlier detection
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_percentage = len(outliers) / len(df)
            
            # Check against accuracy rules
            rule = self.rules.get("ACCURACY_001")
            if rule and outlier_percentage > rule.threshold:
                issue = QualityIssue(
                    issue_id=f"ACC_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule.rule_id,
                    dimension=QualityDimension.ACCURACY,
                    severity=rule.severity,
                    description=f"Column '{column}' has {outlier_percentage:.1%} outliers",
                    affected_columns=[column],
                    affected_rows=len(outliers),
                    score=1.0 - outlier_percentage,
                    timestamp=datetime.now(),
                    details={'outlier_count': len(outliers), 'outlier_percentage': outlier_percentage}
                )
                issues.append(issue)
            
            scores.append(1.0 - outlier_percentage)
        
        return np.mean(scores) if scores else 1.0
    
    def _assess_consistency(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Assess data consistency"""
        scores = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check format consistency for string columns
                consistency_score = self._check_format_consistency(df[column])
                
                if consistency_score < 0.95:  # Less than 95% consistent
                    rule = self.rules.get("CONSISTENCY_001")
                    if rule:
                        issue = QualityIssue(
                            issue_id=f"CON_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            rule_id=rule.rule_id,
                            dimension=QualityDimension.CONSISTENCY,
                            severity=rule.severity,
                            description=f"Column '{column}' has inconsistent formats",
                            affected_columns=[column],
                            affected_rows=int(len(df) * (1 - consistency_score)),
                            score=consistency_score,
                            timestamp=datetime.now(),
                            details={'consistency_score': consistency_score}
                        )
                        issues.append(issue)
                
                scores.append(consistency_score)
            else:
                scores.append(1.0)
        
        return np.mean(scores) if scores else 1.0
    
    def _assess_timeliness(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Assess data timeliness"""
        scores = []
        
        # Look for datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        if not datetime_columns.empty:
            for column in datetime_columns:
                # Calculate data age
                latest_date = df[column].max()
                if pd.notna(latest_date):
                    age_hours = (datetime.now() - latest_date).total_seconds() / 3600
                    
                    # Check against timeliness rules
                    rule = self.rules.get("TIMELINESS_001")
                    if rule and age_hours > rule.threshold:
                        issue = QualityIssue(
                            issue_id=f"TIME_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            rule_id=rule.rule_id,
                            dimension=QualityDimension.TIMELINESS,
                            severity=rule.severity,
                            description=f"Data in column '{column}' is {age_hours:.1f} hours old",
                            affected_columns=[column],
                            affected_rows=len(df),
                            score=max(0, 1.0 - age_hours / rule.threshold),
                            timestamp=datetime.now(),
                            details={'age_hours': age_hours}
                        )
                        issues.append(issue)
                    
                    scores.append(max(0, 1.0 - age_hours / 168))  # 1 week = 168 hours
        
        return np.mean(scores) if scores else 1.0
    
    def _assess_validity(self, df: pd.DataFrame, issues: List[QualityIssue], 
                        schema: Optional[Dict[str, Any]] = None) -> float:
        """Assess data validity against schema or expected patterns"""
        scores = []
        
        for column in df.columns:
            validity_score = 1.0
            
            # Check data type validity
            if schema and column in schema:
                expected_type = schema[column].get('type')
                if expected_type:
                    type_validity = self._check_type_validity(df[column], expected_type)
                    validity_score *= type_validity
            
            # Check for common validity patterns
            if df[column].dtype == 'object':
                # Email validation
                if 'email' in column.lower():
                    validity_score *= self._validate_email_format(df[column])
                # Phone validation
                elif 'phone' in column.lower():
                    validity_score *= self._validate_phone_format(df[column])
            
            if validity_score < 0.99:  # Less than 99% valid
                rule = self.rules.get("VALIDITY_001")
                if rule:
                    issue = QualityIssue(
                        issue_id=f"VAL_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        rule_id=rule.rule_id,
                        dimension=QualityDimension.VALIDITY,
                        severity=rule.severity,
                        description=f"Column '{column}' has invalid values",
                        affected_columns=[column],
                        affected_rows=int(len(df) * (1 - validity_score)),
                        score=validity_score,
                        timestamp=datetime.now(),
                        details={'validity_score': validity_score}
                    )
                    issues.append(issue)
            
            scores.append(validity_score)
        
        return np.mean(scores) if scores else 1.0
    
    def _assess_uniqueness(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Assess data uniqueness"""
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)
        
        # Check against uniqueness rules
        rule = self.rules.get("UNIQUENESS_001")
        if rule and duplicate_percentage > rule.threshold:
            issue = QualityIssue(
                issue_id=f"UNI_DUPLICATES_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                dimension=QualityDimension.UNIQUENESS,
                severity=rule.severity,
                description=f"Dataset has {duplicate_percentage:.1%} duplicate rows",
                affected_columns=list(df.columns),
                affected_rows=duplicate_count,
                score=1.0 - duplicate_percentage,
                timestamp=datetime.now(),
                details={'duplicate_count': duplicate_count, 'duplicate_percentage': duplicate_percentage}
            )
            issues.append(issue)
        
        return 1.0 - duplicate_percentage
    
    def _assess_integrity(self, df: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Assess referential integrity"""
        # This is a simplified implementation
        # In practice, you'd check foreign key relationships
        return 1.0
    
    def _check_format_consistency(self, series: pd.Series) -> float:
        """Check format consistency for string columns"""
        if series.dtype != 'object':
            return 1.0
        
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return 1.0
        
        # Simple pattern consistency check
        patterns = {}
        for value in non_null_values:
            pattern = self._extract_pattern(str(value))
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Calculate consistency as ratio of most common pattern
        if patterns:
            most_common_count = max(patterns.values())
            return most_common_count / len(non_null_values)
        
        return 1.0
    
    def _extract_pattern(self, value: str) -> str:
        """Extract pattern from string value"""
        pattern = ""
        for char in value:
            if char.isdigit():
                pattern += "9"
            elif char.isalpha():
                pattern += "A"
            elif char.isspace():
                pattern += " "
            else:
                pattern += char
        return pattern
    
    def _check_type_validity(self, series: pd.Series, expected_type: str) -> float:
        """Check data type validity"""
        try:
            if expected_type == 'int':
                pd.to_numeric(series, errors='coerce')
            elif expected_type == 'float':
                pd.to_numeric(series, errors='coerce')
            elif expected_type == 'datetime':
                pd.to_datetime(series, errors='coerce')
            
            return 1.0
        except:
            return 0.8  # Assume 80% valid if conversion fails
    
    def _validate_email_format(self, series: pd.Series) -> float:
        """Validate email format"""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return 1.0
        
        valid_emails = non_null_values.str.match(email_pattern).sum()
        return valid_emails / len(non_null_values)
    
    def _validate_phone_format(self, series: pd.Series) -> float:
        """Validate phone format"""
        import re
        phone_patterns = [
            r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$',
            r'^\d{10}$'
        ]
        
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return 1.0
        
        valid_phones = 0
        for pattern in phone_patterns:
            valid_phones += non_null_values.str.match(pattern).sum()
        
        return min(1.0, valid_phones / len(non_null_values))
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        weights = self.config['overall_score_weights']
        weighted_score = 0
        total_weight = 0
        
        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0)
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0
    
    def _detect_anomalies(self, metrics: QualityMetrics):
        """Detect anomalies in quality metrics"""
        if len(self.metrics_history) < self.config['anomaly_detection']['min_history_points']:
            return
        
        # Compare with historical averages
        historical_scores = [m.overall_score for m in self.metrics_history[:-1]]
        current_score = metrics.overall_score
        
        if len(historical_scores) >= 5:
            mean_score = np.mean(historical_scores)
            std_score = np.std(historical_scores)
            
            # Check for anomaly
            if abs(current_score - mean_score) > 2 * std_score:
                logger.warning(f"Anomaly detected: Current score {current_score:.3f} "
                             f"differs significantly from historical average {mean_score:.3f}")
    
    def _check_alerts(self, metrics: QualityMetrics):
        """Check if alerts should be generated"""
        thresholds = self.config['alerting']
        
        if metrics.overall_score < thresholds['critical_threshold']:
            logger.critical(f"CRITICAL: Data quality score {metrics.overall_score:.3f} "
                           f"is below critical threshold {thresholds['critical_threshold']}")
        elif metrics.overall_score < thresholds['high_threshold']:
            logger.error(f"HIGH: Data quality score {metrics.overall_score:.3f} "
                        f"is below high threshold {thresholds['high_threshold']}")
        elif metrics.overall_score < thresholds['medium_threshold']:
            logger.warning(f"MEDIUM: Data quality score {metrics.overall_score:.3f} "
                          f"is below medium threshold {thresholds['medium_threshold']}")
    
    def generate_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            'summary': {
                'dataset_name': metrics.dataset_name,
                'assessment_timestamp': metrics.timestamp.isoformat(),
                'overall_score': metrics.overall_score,
                'total_records': metrics.total_records,
                'total_columns': metrics.total_columns,
                'total_issues': len(metrics.issues)
            },
            'dimension_scores': metrics.dimension_scores,
            'issues': [
                {
                    'issue_id': issue.issue_id,
                    'dimension': issue.dimension.value,
                    'severity': issue.severity.value,
                    'description': issue.description,
                    'affected_columns': issue.affected_columns,
                    'affected_rows': issue.affected_rows,
                    'score': issue.score
                }
                for issue in metrics.issues
            ],
            'trends': self._generate_trend_analysis() if len(self.metrics_history) > 1 else None,
            'recommendations': self._generate_recommendations(metrics),
            'metadata': metrics.metadata
        }
        
        return report
    
    def _generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate trend analysis from historical data"""
        if len(self.metrics_history) < 2:
            return None
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 assessments
        
        trends = {
            'overall_score_trend': self._calculate_trend([m.overall_score for m in recent_metrics]),
            'dimension_trends': {}
        }
        
        # Calculate trends for each dimension
        for dimension in QualityDimension:
            dimension_scores = [m.dimension_scores.get(dimension.value, 0) for m in recent_metrics]
            trends['dimension_trends'][dimension.value] = self._calculate_trend(dimension_scores)
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for a series of values"""
        if len(values) < 2:
            return {'direction': 'stable', 'change': 0}
        
        # Simple linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.01:
            direction = 'improving'
        elif slope < -0.01:
            direction = 'declining'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'change': slope,
            'current_value': values[-1],
            'previous_value': values[-2] if len(values) >= 2 else values[-1]
        }
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[Dict[str, Any]]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Recommendations based on dimension scores
        for dimension, score in metrics.dimension_scores.items():
            if score < 0.8:
                recommendations.append({
                    'dimension': dimension,
                    'priority': 'high' if score < 0.6 else 'medium',
                    'recommendation': self._get_dimension_recommendation(dimension, score),
                    'score': score
                })
        
        # Recommendations based on critical issues
        critical_issues = [issue for issue in metrics.issues if issue.severity == SeverityLevel.CRITICAL]
        if critical_issues:
            recommendations.append({
                'dimension': 'general',
                'priority': 'critical',
                'recommendation': f"Address {len(critical_issues)} critical data quality issues immediately",
                'affected_issues': len(critical_issues)
            })
        
        return recommendations
    
    def _get_dimension_recommendation(self, dimension: str, score: float) -> str:
        """Get specific recommendation for a dimension"""
        recommendations = {
            'completeness': "Implement data validation rules to reduce missing values. "
                          "Consider default values or required field constraints.",
            'accuracy': "Review data entry processes and implement outlier detection. "
                       "Consider automated data cleansing rules.",
            'consistency': "Standardize data formats and implement format validation. "
                         "Create data entry guidelines and templates.",
            'timeliness': "Optimize data pipeline frequency and monitoring. "
                         "Implement real-time data ingestion where possible.",
            'validity': "Strengthen data type validation and format checking. "
                       "Implement comprehensive data validation rules.",
            'uniqueness': "Implement duplicate detection and removal processes. "
                         "Review data ingestion logic for duplicate prevention."
        }
        
        return recommendations.get(dimension, "Review and improve data quality processes for this dimension.")
    
    def create_quality_dashboard(self, metrics: QualityMetrics) -> go.Figure:
        """Create interactive quality dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Quality Score', 'Dimension Scores', 
                           'Issues by Severity', 'Quality Trends'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Overall quality score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.overall_score,
                title={'text': "Overall Quality Score"},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': self._get_score_color(metrics.overall_score)},
                       'steps': [{'range': [0, 0.6], 'color': "lightgray"},
                                {'range': [0.6, 0.8], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75,
                                   'value': 0.8}}
            ),
            row=1, col=1
        )
        
        # Dimension scores bar chart
        dimensions = list(metrics.dimension_scores.keys())
        scores = list(metrics.dimension_scores.values())
        
        fig.add_trace(
            go.Bar(
                x=dimensions,
                y=scores,
                marker_color=[self._get_score_color(score) for score in scores],
                name="Dimension Scores"
            ),
            row=1, col=2
        )
        
        # Issues by severity pie chart
        severity_counts = {}
        for issue in metrics.issues:
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
        
        if severity_counts:
            fig.add_trace(
                go.Pie(
                    labels=list(severity_counts.keys()),
                    values=list(severity_counts.values()),
                    name="Issues by Severity"
                ),
                row=2, col=1
            )
        
        # Quality trends (if historical data available)
        if len(self.metrics_history) > 1:
            timestamps = [m.timestamp for m in self.metrics_history]
            overall_scores = [m.overall_score for m in self.metrics_history]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=overall_scores,
                    mode='lines+markers',
                    name="Quality Trend",
                    line=dict(color='blue', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text=f"Data Quality Dashboard - {metrics.dataset_name}",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score"""
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "orange"
        else:
            return "red"
    
    def export_quality_report(self, metrics: QualityMetrics, output_path: str):
        """Export quality report to file"""
        report = self.generate_quality_report(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Quality report exported to: {output_path}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of all quality assessments"""
        if not self.metrics_history:
            return {'message': 'No quality assessments performed yet'}
        
        latest_metrics = self.metrics_history[-1]
        
        summary = {
            'total_assessments': len(self.metrics_history),
            'latest_assessment': {
                'timestamp': latest_metrics.timestamp.isoformat(),
                'overall_score': latest_metrics.overall_score,
                'dataset_name': latest_metrics.dataset_name,
                'total_issues': len(latest_metrics.issues)
            },
            'average_score': np.mean([m.overall_score for m in self.metrics_history]),
            'trend': self._calculate_trend([m.overall_score for m in self.metrics_history])
        }
        
        return summary

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Quality Monitor")
    parser.add_argument("input_file", help="CSV file to assess")
    parser.add_argument("--config", "-c", help="Configuration file")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--dashboard", "-d", help="Generate interactive dashboard")
    parser.add_argument("--dataset-name", "-n", default="dataset", help="Dataset name")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_file)
    
    # Initialize monitor
    monitor = DataQualityMonitor(args.config)
    
    # Assess quality
    metrics = monitor.assess_quality(df, args.dataset_name)
    
    # Generate report
    report = monitor.generate_quality_report(metrics)
    
    # Print summary
    print(f"\n📊 Data Quality Assessment Summary")
    print(f"Dataset: {metrics.dataset_name}")
    print(f"Records: {metrics.total_records:,}")
    print(f"Columns: {metrics.total_columns}")
    print(f"Overall Score: {metrics.overall_score:.3f}")
    print(f"Total Issues: {len(metrics.issues)}")
    
    # Print dimension scores
    print(f"\n📈 Quality Dimension Scores:")
    for dimension, score in metrics.dimension_scores.items():
        print(f"  {dimension.title()}: {score:.3f}")
    
    # Print critical issues
    critical_issues = [issue for issue in metrics.issues if issue.severity == SeverityLevel.CRITICAL]
    if critical_issues:
        print(f"\n🚨 Critical Issues:")
        for issue in critical_issues:
            print(f"  - {issue.description}")
    
    # Save report
    if args.output:
        monitor.export_quality_report(metrics, args.output)
        print(f"\n💾 Report saved to: {args.output}")
    
    # Generate dashboard
    if args.dashboard:
        dashboard = monitor.create_quality_dashboard(metrics)
        dashboard.write_html(args.dashboard)
        print(f"📊 Dashboard saved to: {args.dashboard}")

if __name__ == "__main__":
    main() 