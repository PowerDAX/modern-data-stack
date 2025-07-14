#!/usr/bin/env python3
"""
Data Masking and Anonymization Utilities
========================================

This module provides comprehensive data masking and anonymization capabilities
for protecting sensitive data while maintaining statistical properties and
referential integrity for testing and development purposes.

Features:
- Multiple masking strategies (substitution, shuffling, noise addition)
- Referential integrity preservation
- Statistical property preservation
- Configurable anonymization levels
- Format-preserving encryption
- Synthetic data generation for sensitive fields

Author: Data Engineering Team
Date: 2024-01-15
"""

import pandas as pd
import numpy as np
import hashlib
import secrets
import string
import re
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from faker import Faker
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MaskingConfig:
    """Configuration for data masking operations"""
    # Masking strategies
    default_strategy: str = "substitution"
    preserve_nulls: bool = True
    preserve_format: bool = True
    preserve_statistics: bool = True
    
    # Anonymization levels
    anonymization_level: str = "medium"  # low, medium, high
    
    # Encryption settings
    encryption_key: Optional[str] = None
    use_deterministic_masking: bool = True
    
    # Statistical preservation
    preserve_distributions: bool = True
    preserve_correlations: bool = True
    correlation_threshold: float = 0.7
    
    # Referential integrity
    preserve_referential_integrity: bool = True
    foreign_key_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Output settings
    output_format: str = "csv"
    chunk_size: int = 10000
    
    # Audit and compliance
    audit_trail: bool = True
    compliance_mode: str = "GDPR"  # GDPR, HIPAA, PCI-DSS

class DataMaskingEngine:
    """
    Comprehensive data masking engine with multiple strategies and
    compliance-aware anonymization capabilities.
    """
    
    def __init__(self, config: MaskingConfig = None):
        """
        Initialize the data masking engine.
        
        Parameters
        ----------
        config : MaskingConfig, optional
            Configuration for masking operations
        """
        self.config = config or MaskingConfig()
        self.fake = Faker()
        self.fake.seed_instance(42)
        
        # Initialize masking strategies
        self.strategies = {
            'substitution': self._substitute_masking,
            'shuffling': self._shuffle_masking,
            'noise_addition': self._noise_addition_masking,
            'format_preserving': self._format_preserving_masking,
            'synthetic_replacement': self._synthetic_replacement_masking,
            'tokenization': self._tokenization_masking,
            'encryption': self._encryption_masking,
            'nullification': self._nullification_masking
        }
        
        # Initialize field type detectors
        self.field_types = {
            'email': self._detect_email_field,
            'phone': self._detect_phone_field,
            'ssn': self._detect_ssn_field,
            'credit_card': self._detect_credit_card_field,
            'name': self._detect_name_field,
            'address': self._detect_address_field,
            'date': self._detect_date_field,
            'numeric_id': self._detect_numeric_id_field,
            'string_id': self._detect_string_id_field
        }
        
        # Initialize compliance rules
        self.compliance_rules = self._load_compliance_rules()
        
        # Audit trail
        self.audit_log = []
        
    def mask_dataset(self, df: pd.DataFrame, 
                    masking_rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply masking to an entire dataset according to specified rules.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to mask
        masking_rules : Dict[str, Dict[str, Any]]
            Masking rules for each column
            
        Returns
        -------
        pd.DataFrame
            Masked dataset
        """
        logger.info(f"Starting dataset masking for {len(df)} records, {len(df.columns)} columns")
        
        # Create copy to avoid modifying original
        masked_df = df.copy()
        
        # Auto-detect sensitive fields if not specified
        if not masking_rules:
            masking_rules = self._auto_detect_sensitive_fields(df)
        
        # Apply masking rules
        for column, rules in masking_rules.items():
            if column in masked_df.columns:
                logger.info(f"Masking column: {column} with strategy: {rules.get('strategy', 'default')}")
                masked_df[column] = self._apply_column_masking(masked_df[column], rules)
        
        # Preserve referential integrity
        if self.config.preserve_referential_integrity:
            masked_df = self._preserve_referential_integrity(masked_df, masking_rules)
        
        # Preserve statistical properties
        if self.config.preserve_statistics:
            masked_df = self._preserve_statistical_properties(df, masked_df, masking_rules)
        
        # Log masking operation
        if self.config.audit_trail:
            self._log_masking_operation(df, masked_df, masking_rules)
        
        logger.info(f"Dataset masking completed successfully")
        return masked_df
    
    def _auto_detect_sensitive_fields(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Automatically detect sensitive fields in the dataset"""
        logger.info("Auto-detecting sensitive fields...")
        
        masking_rules = {}
        
        for column in df.columns:
            field_type = self._detect_field_type(df[column])
            
            if field_type:
                # Apply appropriate masking strategy based on field type
                strategy = self._get_strategy_for_field_type(field_type)
                masking_rules[column] = {
                    'strategy': strategy,
                    'field_type': field_type,
                    'auto_detected': True
                }
                logger.info(f"Detected {field_type} field: {column}")
        
        return masking_rules
    
    def _detect_field_type(self, series: pd.Series) -> Optional[str]:
        """Detect the type of sensitive field"""
        # Sample some values for detection
        sample_values = series.dropna().head(100).astype(str)
        
        if len(sample_values) == 0:
            return None
        
        # Check each field type
        for field_type, detector in self.field_types.items():
            if detector(sample_values):
                return field_type
        
        return None
    
    def _detect_email_field(self, values: pd.Series) -> bool:
        """Detect email fields"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        matches = values.str.match(email_pattern).sum()
        return matches / len(values) > 0.8
    
    def _detect_phone_field(self, values: pd.Series) -> bool:
        """Detect phone number fields"""
        phone_patterns = [
            r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$',
            r'^\d{10}$',
            r'^\d{3}-\d{3}-\d{4}$'
        ]
        
        for pattern in phone_patterns:
            matches = values.str.match(pattern).sum()
            if matches / len(values) > 0.8:
                return True
        return False
    
    def _detect_ssn_field(self, values: pd.Series) -> bool:
        """Detect SSN fields"""
        ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'
        matches = values.str.match(ssn_pattern).sum()
        return matches / len(values) > 0.8
    
    def _detect_credit_card_field(self, values: pd.Series) -> bool:
        """Detect credit card fields"""
        cc_pattern = r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'
        matches = values.str.match(cc_pattern).sum()
        return matches / len(values) > 0.8
    
    def _detect_name_field(self, values: pd.Series) -> bool:
        """Detect name fields"""
        # Look for common name patterns
        name_indicators = ['name', 'first', 'last', 'middle', 'full']
        column_name = values.name.lower() if values.name else ""
        
        return any(indicator in column_name for indicator in name_indicators)
    
    def _detect_address_field(self, values: pd.Series) -> bool:
        """Detect address fields"""
        address_indicators = ['address', 'street', 'addr', 'location']
        column_name = values.name.lower() if values.name else ""
        
        return any(indicator in column_name for indicator in address_indicators)
    
    def _detect_date_field(self, values: pd.Series) -> bool:
        """Detect date fields"""
        try:
            pd.to_datetime(values.head(10))
            return True
        except:
            return False
    
    def _detect_numeric_id_field(self, values: pd.Series) -> bool:
        """Detect numeric ID fields"""
        if not pd.api.types.is_numeric_dtype(values):
            return False
        
        # Check if values are sequential or have ID-like patterns
        unique_ratio = len(values.unique()) / len(values)
        return unique_ratio > 0.9  # High uniqueness suggests ID field
    
    def _detect_string_id_field(self, values: pd.Series) -> bool:
        """Detect string ID fields"""
        if not pd.api.types.is_string_dtype(values):
            return False
        
        # Check for ID patterns
        id_patterns = [r'^[A-Z]\d+$', r'^[A-Z]{2,3}\d+$', r'^[A-Z0-9]{8,}$']
        
        for pattern in id_patterns:
            matches = values.str.match(pattern).sum()
            if matches / len(values) > 0.8:
                return True
        
        return False
    
    def _get_strategy_for_field_type(self, field_type: str) -> str:
        """Get appropriate masking strategy for field type"""
        strategy_mapping = {
            'email': 'synthetic_replacement',
            'phone': 'format_preserving',
            'ssn': 'format_preserving',
            'credit_card': 'tokenization',
            'name': 'synthetic_replacement',
            'address': 'synthetic_replacement',
            'date': 'noise_addition',
            'numeric_id': 'shuffling',
            'string_id': 'tokenization'
        }
        
        return strategy_mapping.get(field_type, self.config.default_strategy)
    
    def _apply_column_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Apply masking to a single column"""
        strategy = rules.get('strategy', self.config.default_strategy)
        
        if strategy in self.strategies:
            return self.strategies[strategy](series, rules)
        else:
            logger.warning(f"Unknown strategy: {strategy}, using default")
            return self.strategies[self.config.default_strategy](series, rules)
    
    def _substitute_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Simple substitution masking"""
        masked_series = series.copy()
        
        if pd.api.types.is_string_dtype(series):
            # Replace with asterisks, preserving length
            masked_series = series.str.replace(r'.', '*', regex=True)
        elif pd.api.types.is_numeric_dtype(series):
            # Replace with zeros
            masked_series = pd.Series([0] * len(series), index=series.index)
        
        return masked_series
    
    def _shuffle_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Shuffle masking to preserve distribution"""
        if self.config.use_deterministic_masking:
            # Use deterministic shuffling
            np.random.seed(42)
        
        masked_series = series.copy()
        non_null_mask = masked_series.notna()
        
        if non_null_mask.any():
            # Shuffle non-null values
            non_null_values = masked_series[non_null_mask].values
            np.random.shuffle(non_null_values)
            masked_series[non_null_mask] = non_null_values
        
        return masked_series
    
    def _noise_addition_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Add noise while preserving statistical properties"""
        if not pd.api.types.is_numeric_dtype(series):
            # For non-numeric data, fall back to shuffling
            return self._shuffle_masking(series, rules)
        
        masked_series = series.copy()
        non_null_mask = masked_series.notna()
        
        if non_null_mask.any():
            # Calculate noise parameters
            std_dev = masked_series[non_null_mask].std()
            noise_factor = rules.get('noise_factor', 0.1)
            
            # Add Gaussian noise
            noise = np.random.normal(0, std_dev * noise_factor, non_null_mask.sum())
            masked_series[non_null_mask] += noise
        
        return masked_series
    
    def _format_preserving_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Format-preserving masking"""
        masked_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            str_value = str(value)
            
            # Preserve format but change digits/letters
            masked_value = ""
            for char in str_value:
                if char.isdigit():
                    masked_value += str(np.random.randint(0, 10))
                elif char.isalpha():
                    if char.isupper():
                        masked_value += np.random.choice(list(string.ascii_uppercase))
                    else:
                        masked_value += np.random.choice(list(string.ascii_lowercase))
                else:
                    masked_value += char
            
            masked_series[idx] = masked_value
        
        return masked_series
    
    def _synthetic_replacement_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Replace with synthetic data"""
        field_type = rules.get('field_type', 'generic')
        masked_series = series.copy()
        
        # Generate synthetic data based on field type
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            if field_type == 'email':
                masked_series[idx] = self.fake.email()
            elif field_type == 'phone':
                masked_series[idx] = self.fake.phone_number()
            elif field_type == 'name':
                masked_series[idx] = self.fake.name()
            elif field_type == 'address':
                masked_series[idx] = self.fake.address()
            else:
                # Generic replacement
                masked_series[idx] = f"MASKED_{hash(str(value)) % 10000}"
        
        return masked_series
    
    def _tokenization_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Tokenization masking"""
        masked_series = series.copy()
        token_map = {}
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            str_value = str(value)
            
            # Create deterministic token
            if str_value not in token_map:
                if self.config.use_deterministic_masking:
                    hash_value = hashlib.sha256(str_value.encode()).hexdigest()[:8]
                    token_map[str_value] = f"TOKEN_{hash_value}"
                else:
                    token_map[str_value] = f"TOKEN_{secrets.token_hex(4)}"
            
            masked_series[idx] = token_map[str_value]
        
        return masked_series
    
    def _encryption_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Encryption-based masking"""
        # This is a simplified encryption for demo purposes
        # In production, use proper encryption libraries
        
        masked_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            # Simple hash-based "encryption"
            encrypted = hashlib.sha256(str(value).encode()).hexdigest()[:16]
            masked_series[idx] = f"ENC_{encrypted}"
        
        return masked_series
    
    def _nullification_masking(self, series: pd.Series, rules: Dict[str, Any]) -> pd.Series:
        """Replace with null values"""
        return pd.Series([None] * len(series), index=series.index)
    
    def _preserve_referential_integrity(self, df: pd.DataFrame, 
                                      masking_rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Preserve referential integrity across related columns"""
        # This is a simplified implementation
        # In production, you'd need more sophisticated relationship tracking
        
        # Create mapping for foreign key relationships
        for column, rules in masking_rules.items():
            if 'foreign_key' in rules:
                parent_column = rules['foreign_key']
                if parent_column in df.columns:
                    # Ensure foreign key values exist in parent column
                    parent_values = df[parent_column].dropna().unique()
                    df[column] = df[column].map(
                        lambda x: np.random.choice(parent_values) if pd.notna(x) else x
                    )
        
        return df
    
    def _preserve_statistical_properties(self, original_df: pd.DataFrame, 
                                       masked_df: pd.DataFrame,
                                       masking_rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Preserve statistical properties where possible"""
        if not self.config.preserve_statistics:
            return masked_df
        
        # For numeric columns, ensure distributions are preserved
        for column in masked_df.columns:
            if (pd.api.types.is_numeric_dtype(original_df[column]) and 
                pd.api.types.is_numeric_dtype(masked_df[column])):
                
                # Check if distribution preservation is requested
                rules = masking_rules.get(column, {})
                if rules.get('preserve_distribution', True):
                    # Adjust masked values to match original distribution
                    masked_df[column] = self._adjust_distribution(
                        original_df[column], masked_df[column]
                    )
        
        return masked_df
    
    def _adjust_distribution(self, original: pd.Series, masked: pd.Series) -> pd.Series:
        """Adjust masked values to match original distribution"""
        # Simple distribution matching - scale to match mean and std
        orig_mean = original.mean()
        orig_std = original.std()
        
        masked_mean = masked.mean()
        masked_std = masked.std()
        
        if masked_std != 0:
            # Scale and shift to match original distribution
            adjusted = (masked - masked_mean) / masked_std * orig_std + orig_mean
            return adjusted
        
        return masked
    
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance-specific masking rules"""
        rules = {
            'GDPR': {
                'personal_identifiers': ['full_anonymization'],
                'sensitive_categories': ['pseudonymization'],
                'audit_requirements': True
            },
            'HIPAA': {
                'protected_health_info': ['de_identification'],
                'audit_requirements': True,
                'minimum_anonymization': 'safe_harbor'
            },
            'PCI-DSS': {
                'payment_card_data': ['tokenization'],
                'audit_requirements': True,
                'encryption_required': True
            }
        }
        
        return rules
    
    def _log_masking_operation(self, original_df: pd.DataFrame, 
                             masked_df: pd.DataFrame,
                             masking_rules: Dict[str, Dict[str, Any]]):
        """Log masking operation for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'data_masking',
            'original_record_count': len(original_df),
            'masked_record_count': len(masked_df),
            'columns_masked': list(masking_rules.keys()),
            'strategies_used': {col: rules.get('strategy', 'unknown') 
                              for col, rules in masking_rules.items()},
            'compliance_mode': self.config.compliance_mode,
            'config': {
                'preserve_nulls': self.config.preserve_nulls,
                'preserve_format': self.config.preserve_format,
                'preserve_statistics': self.config.preserve_statistics,
                'anonymization_level': self.config.anonymization_level
            }
        }
        
        self.audit_log.append(log_entry)
    
    def generate_masking_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate a comprehensive masking report"""
        report = {
            'masking_summary': {
                'total_operations': len(self.audit_log),
                'total_records_processed': sum(log['original_record_count'] for log in self.audit_log),
                'compliance_mode': self.config.compliance_mode,
                'configuration': self.config.__dict__
            },
            'operations': self.audit_log,
            'compliance_validation': self._validate_compliance(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _validate_compliance(self) -> Dict[str, Any]:
        """Validate compliance with specified regulations"""
        compliance_rules = self.compliance_rules.get(self.config.compliance_mode, {})
        
        validation_results = {
            'compliant': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check audit trail requirement
        if compliance_rules.get('audit_requirements', False):
            if not self.config.audit_trail:
                validation_results['compliant'] = False
                validation_results['issues'].append("Audit trail is required for compliance")
        
        return validation_results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving masking strategy"""
        recommendations = []
        
        if self.config.anonymization_level == 'low':
            recommendations.append("Consider increasing anonymization level for better privacy protection")
        
        if not self.config.preserve_referential_integrity:
            recommendations.append("Enable referential integrity preservation for better data quality")
        
        return recommendations

def create_sample_masking_rules() -> Dict[str, Dict[str, Any]]:
    """Create sample masking rules for common scenarios"""
    return {
        'customer_id': {
            'strategy': 'tokenization',
            'field_type': 'string_id',
            'preserve_format': True
        },
        'first_name': {
            'strategy': 'synthetic_replacement',
            'field_type': 'name',
            'preserve_distribution': True
        },
        'last_name': {
            'strategy': 'synthetic_replacement',
            'field_type': 'name',
            'preserve_distribution': True
        },
        'email': {
            'strategy': 'synthetic_replacement',
            'field_type': 'email',
            'preserve_format': True
        },
        'phone': {
            'strategy': 'format_preserving',
            'field_type': 'phone',
            'preserve_format': True
        },
        'address': {
            'strategy': 'synthetic_replacement',
            'field_type': 'address',
            'preserve_distribution': True
        },
        'birth_date': {
            'strategy': 'noise_addition',
            'field_type': 'date',
            'noise_factor': 0.1,
            'preserve_distribution': True
        },
        'income': {
            'strategy': 'noise_addition',
            'field_type': 'numeric',
            'noise_factor': 0.15,
            'preserve_distribution': True
        },
        'credit_card': {
            'strategy': 'tokenization',
            'field_type': 'credit_card',
            'preserve_format': False
        },
        'ssn': {
            'strategy': 'format_preserving',
            'field_type': 'ssn',
            'preserve_format': True
        }
    }

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Masking and Anonymization Utility")
    parser.add_argument("input_file", help="Input CSV file to mask")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--rules", "-r", help="JSON file containing masking rules")
    parser.add_argument("--compliance", "-c", choices=['GDPR', 'HIPAA', 'PCI-DSS'], 
                       default='GDPR', help="Compliance mode")
    parser.add_argument("--level", "-l", choices=['low', 'medium', 'high'], 
                       default='medium', help="Anonymization level")
    parser.add_argument("--auto-detect", "-a", action="store_true", 
                       help="Auto-detect sensitive fields")
    parser.add_argument("--report", help="Generate masking report")
    
    args = parser.parse_args()
    
    # Load input data
    df = pd.read_csv(args.input_file)
    
    # Create configuration
    config = MaskingConfig(
        compliance_mode=args.compliance,
        anonymization_level=args.level,
        audit_trail=True
    )
    
    # Create masking engine
    engine = DataMaskingEngine(config)
    
    # Load or create masking rules
    if args.rules:
        with open(args.rules, 'r') as f:
            masking_rules = json.load(f)
    elif args.auto_detect:
        masking_rules = {}  # Auto-detection will be used
    else:
        masking_rules = create_sample_masking_rules()
    
    # Apply masking
    masked_df = engine.mask_dataset(df, masking_rules)
    
    # Save output
    output_path = args.output or f"masked_{Path(args.input_file).stem}.csv"
    masked_df.to_csv(output_path, index=False)
    
    print(f"Masking completed successfully!")
    print(f"Input records: {len(df)}")
    print(f"Output records: {len(masked_df)}")
    print(f"Masked file saved to: {output_path}")
    
    # Generate report
    if args.report:
        report = engine.generate_masking_report(args.report)
        print(f"Masking report saved to: {args.report}")

if __name__ == "__main__":
    main() 