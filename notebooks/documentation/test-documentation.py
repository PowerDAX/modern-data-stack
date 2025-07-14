#!/usr/bin/env python3
"""
Documentation Testing Framework
==============================

This script provides comprehensive testing for Jupyter notebook documentation
to ensure compliance with the Modern Data Stack Showcase standards.

Features:
- Documentation completeness validation
- Code quality assessment
- Cross-reference validation
- Performance testing
- Accessibility testing
- Link validation
- Image validation
- Automated quality scoring

Usage:
    python test-documentation.py --notebook NOTEBOOK_PATH
    python test-documentation.py --directory DIRECTORY_PATH
    python test-documentation.py --generate-report

Author: Data Science Team
Date: 2024-01-15
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
import ast
import nbformat
import requests
from urllib.parse import urlparse
import subprocess
import time
import hashlib
from dataclasses import dataclass, asdict
import yaml
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('documentation-testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data class."""
    test_name: str
    passed: bool
    score: float
    message: str
    details: Dict[str, Any]
    timestamp: str

@dataclass
class NotebookTestSuite:
    """Test suite results for a notebook."""
    notebook_path: str
    notebook_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    test_results: List[TestResult]
    timestamp: str

class DocumentationTester:
    """
    Comprehensive documentation testing framework.
    
    This class provides various tests to validate notebook documentation
    quality and compliance with standards.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the documentation tester.
        
        Parameters
        ----------
        config_file : str, optional
            Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.test_results = {}
        
        logger.info("Documentation tester initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration settings."""
        default_config = {
            'documentation_standards': {
                'min_header_fields': 10,
                'min_documentation_ratio': 0.3,
                'min_function_docstring_ratio': 0.8,
                'min_visualization_documentation_ratio': 0.9,
                'required_header_fields': [
                    'Category', 'Author', 'Created', 'Last Updated', 
                    'Purpose', 'Prerequisites', 'Key Outcomes'
                ],
                'required_sections': [
                    'Table of Contents', 'Environment Setup', 
                    'Conclusions', 'References'
                ]
            },
            'code_quality': {
                'max_cell_complexity': 15,
                'max_function_complexity': 10,
                'min_comment_ratio': 0.2,
                'max_line_length': 88,
                'required_imports_documentation': True
            },
            'performance': {
                'max_execution_time': 300,  # 5 minutes
                'max_memory_usage': 2048,   # 2GB
                'performance_warnings': True
            },
            'accessibility': {
                'alt_text_required': True,
                'color_contrast_check': True,
                'screen_reader_compatibility': True
            },
            'links': {
                'validate_external_links': True,
                'validate_internal_links': True,
                'timeout': 10
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def test_notebook(self, notebook_path: str) -> NotebookTestSuite:
        """
        Run comprehensive tests on a notebook.
        
        Parameters
        ----------
        notebook_path : str
            Path to the notebook to test
            
        Returns
        -------
        NotebookTestSuite
            Complete test results
        """
        notebook_path = Path(notebook_path)
        
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        
        logger.info(f"Testing notebook: {notebook_path}")
        
        # Load notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Initialize test suite
        test_suite = NotebookTestSuite(
            notebook_path=str(notebook_path),
            notebook_name=notebook_path.stem,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            overall_score=0.0,
            test_results=[],
            timestamp=datetime.now().isoformat()
        )
        
        # Run all tests
        test_methods = [
            self._test_header_completeness,
            self._test_documentation_structure,
            self._test_code_quality,
            self._test_function_documentation,
            self._test_visualization_documentation,
            self._test_markdown_quality,
            self._test_cross_references,
            self._test_performance_indicators,
            self._test_accessibility,
            self._test_link_validation,
            self._test_image_validation,
            self._test_notebook_execution,
            self._test_cell_ordering,
            self._test_imports_documentation,
            self._test_output_documentation
        ]
        
        for test_method in test_methods:
            try:
                result = test_method(nb, notebook_path)
                test_suite.test_results.append(result)
                test_suite.total_tests += 1
                
                if result.passed:
                    test_suite.passed_tests += 1
                else:
                    test_suite.failed_tests += 1
                
                logger.info(f"Test {result.test_name}: {'PASSED' if result.passed else 'FAILED'} ({result.score:.1f})")
                
            except Exception as e:
                logger.error(f"Error running test {test_method.__name__}: {e}")
                error_result = TestResult(
                    test_name=test_method.__name__,
                    passed=False,
                    score=0.0,
                    message=f"Test failed with error: {str(e)}",
                    details={},
                    timestamp=datetime.now().isoformat()
                )
                test_suite.test_results.append(error_result)
                test_suite.total_tests += 1
                test_suite.failed_tests += 1
        
        # Calculate overall score
        if test_suite.test_results:
            test_suite.overall_score = sum(r.score for r in test_suite.test_results) / len(test_suite.test_results)
        
        logger.info(f"Testing completed: {test_suite.passed_tests}/{test_suite.total_tests} tests passed")
        logger.info(f"Overall score: {test_suite.overall_score:.1f}/100")
        
        return test_suite
    
    def _test_header_completeness(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test header completeness."""
        test_name = "Header Completeness"
        
        if not nb.cells or nb.cells[0].cell_type != 'markdown':
            return TestResult(
                test_name=test_name,
                passed=False,
                score=0.0,
                message="No header cell found",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        header_content = nb.cells[0].source
        required_fields = self.config['documentation_standards']['required_header_fields']
        
        found_fields = []
        missing_fields = []
        
        for field in required_fields:
            field_patterns = [
                f"**📊 {field}**",
                f"**👤 {field}**",
                f"**📅 {field}**",
                f"**🔄 {field}**",
                f"**🎯 {field}**",
                f"**📋 {field}**",
                f"**📈 {field}**"
            ]
            
            if any(pattern in header_content for pattern in field_patterns):
                found_fields.append(field)
            else:
                missing_fields.append(field)
        
        score = (len(found_fields) / len(required_fields)) * 100
        passed = len(missing_fields) == 0
        
        message = f"Found {len(found_fields)}/{len(required_fields)} required fields"
        if missing_fields:
            message += f". Missing: {', '.join(missing_fields)}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'found_fields': found_fields,
                'missing_fields': missing_fields,
                'total_required': len(required_fields)
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_documentation_structure(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test documentation structure."""
        test_name = "Documentation Structure"
        
        markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        
        if not markdown_cells:
            return TestResult(
                test_name=test_name,
                passed=False,
                score=0.0,
                message="No markdown cells found",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        documentation_ratio = len(markdown_cells) / len(nb.cells)
        min_ratio = self.config['documentation_standards']['min_documentation_ratio']
        
        # Check for required sections
        required_sections = self.config['documentation_standards']['required_sections']
        all_content = ' '.join(cell.source for cell in markdown_cells)
        
        found_sections = []
        missing_sections = []
        
        for section in required_sections:
            if section.lower() in all_content.lower():
                found_sections.append(section)
            else:
                missing_sections.append(section)
        
        # Calculate score
        ratio_score = min(documentation_ratio / min_ratio, 1.0) * 50
        section_score = (len(found_sections) / len(required_sections)) * 50
        score = ratio_score + section_score
        
        passed = documentation_ratio >= min_ratio and len(missing_sections) == 0
        
        message = f"Documentation ratio: {documentation_ratio:.2f} (min: {min_ratio:.2f})"
        if missing_sections:
            message += f". Missing sections: {', '.join(missing_sections)}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'documentation_ratio': documentation_ratio,
                'found_sections': found_sections,
                'missing_sections': missing_sections,
                'total_cells': len(nb.cells),
                'markdown_cells': len(markdown_cells),
                'code_cells': len(code_cells)
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_code_quality(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test code quality."""
        test_name = "Code Quality"
        
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        
        if not code_cells:
            return TestResult(
                test_name=test_name,
                passed=True,
                score=100.0,
                message="No code cells to evaluate",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        quality_metrics = {
            'total_cells': len(code_cells),
            'cells_with_comments': 0,
            'complex_cells': 0,
            'long_lines': 0,
            'syntax_errors': 0,
            'max_complexity': 0,
            'avg_complexity': 0
        }
        
        complexities = []
        
        for cell in code_cells:
            source = cell.source
            
            # Check for comments
            if '#' in source:
                quality_metrics['cells_with_comments'] += 1
            
            # Check line length
            for line in source.split('\n'):
                if len(line) > self.config['code_quality']['max_line_length']:
                    quality_metrics['long_lines'] += 1
            
            # Calculate complexity
            try:
                complexity = self._calculate_cell_complexity(source)
                complexities.append(complexity)
                
                if complexity > self.config['code_quality']['max_cell_complexity']:
                    quality_metrics['complex_cells'] += 1
                
            except SyntaxError:
                quality_metrics['syntax_errors'] += 1
        
        if complexities:
            quality_metrics['max_complexity'] = max(complexities)
            quality_metrics['avg_complexity'] = sum(complexities) / len(complexities)
        
        # Calculate score
        comment_ratio = quality_metrics['cells_with_comments'] / quality_metrics['total_cells']
        min_comment_ratio = self.config['code_quality']['min_comment_ratio']
        
        comment_score = min(comment_ratio / min_comment_ratio, 1.0) * 40
        complexity_score = max(0, 40 - (quality_metrics['complex_cells'] * 5))
        line_length_score = max(0, 20 - (quality_metrics['long_lines'] * 2))
        
        score = comment_score + complexity_score + line_length_score
        
        passed = (
            comment_ratio >= min_comment_ratio and
            quality_metrics['complex_cells'] == 0 and
            quality_metrics['long_lines'] < 5 and
            quality_metrics['syntax_errors'] == 0
        )
        
        message = f"Comment ratio: {comment_ratio:.2f}, Complex cells: {quality_metrics['complex_cells']}, Long lines: {quality_metrics['long_lines']}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details=quality_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_cell_complexity(self, source: str) -> int:
        """Calculate cell complexity."""
        complexity = 0
        
        # Count control structures
        complexity += source.count('if ')
        complexity += source.count('for ')
        complexity += source.count('while ')
        complexity += source.count('try:')
        complexity += source.count('except')
        complexity += source.count('def ')
        complexity += source.count('class ')
        
        return complexity
    
    def _test_function_documentation(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test function documentation."""
        test_name = "Function Documentation"
        
        functions = []
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                try:
                    tree = ast.parse(cell.source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            functions.append({
                                'name': node.name,
                                'docstring': ast.get_docstring(node),
                                'args': [arg.arg for arg in node.args.args],
                                'line_number': node.lineno
                            })
                except:
                    pass
        
        if not functions:
            return TestResult(
                test_name=test_name,
                passed=True,
                score=100.0,
                message="No functions to evaluate",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        documented_functions = sum(1 for func in functions if func['docstring'])
        documentation_ratio = documented_functions / len(functions)
        min_ratio = self.config['documentation_standards']['min_function_docstring_ratio']
        
        score = (documentation_ratio / min_ratio) * 100
        score = min(score, 100.0)
        
        passed = documentation_ratio >= min_ratio
        
        message = f"Documented functions: {documented_functions}/{len(functions)} ({documentation_ratio:.2f})"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'total_functions': len(functions),
                'documented_functions': documented_functions,
                'documentation_ratio': documentation_ratio,
                'functions': functions
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_visualization_documentation(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test visualization documentation."""
        test_name = "Visualization Documentation"
        
        visualizations = []
        viz_keywords = ['plt.', 'fig.', 'ax.', 'sns.', 'px.', 'go.', 'plotly']
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                source = cell.source
                for keyword in viz_keywords:
                    if keyword in source:
                        viz_info = {
                            'cell_index': i,
                            'has_title': 'title' in source.lower(),
                            'has_labels': any(label in source.lower() for label in ['xlabel', 'ylabel', 'labels']),
                            'has_legend': 'legend' in source.lower(),
                            'library': self._identify_viz_library(source)
                        }
                        visualizations.append(viz_info)
                        break
        
        if not visualizations:
            return TestResult(
                test_name=test_name,
                passed=True,
                score=100.0,
                message="No visualizations to evaluate",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        well_documented = sum(1 for viz in visualizations if viz['has_title'] and viz['has_labels'])
        documentation_ratio = well_documented / len(visualizations)
        min_ratio = self.config['documentation_standards']['min_visualization_documentation_ratio']
        
        score = (documentation_ratio / min_ratio) * 100
        score = min(score, 100.0)
        
        passed = documentation_ratio >= min_ratio
        
        message = f"Well-documented visualizations: {well_documented}/{len(visualizations)} ({documentation_ratio:.2f})"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'total_visualizations': len(visualizations),
                'well_documented': well_documented,
                'documentation_ratio': documentation_ratio,
                'visualizations': visualizations
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _identify_viz_library(self, source: str) -> str:
        """Identify visualization library."""
        if 'px.' in source or 'plotly' in source:
            return 'plotly'
        elif 'sns.' in source:
            return 'seaborn'
        elif 'plt.' in source:
            return 'matplotlib'
        return 'unknown'
    
    def _test_markdown_quality(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test markdown quality."""
        test_name = "Markdown Quality"
        
        markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
        
        if not markdown_cells:
            return TestResult(
                test_name=test_name,
                passed=False,
                score=0.0,
                message="No markdown cells found",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        quality_metrics = {
            'total_cells': len(markdown_cells),
            'cells_with_headers': 0,
            'cells_with_lists': 0,
            'cells_with_links': 0,
            'cells_with_emphasis': 0,
            'total_words': 0,
            'avg_words_per_cell': 0
        }
        
        for cell in markdown_cells:
            source = cell.source
            
            # Check for headers
            if re.search(r'^#+\s', source, re.MULTILINE):
                quality_metrics['cells_with_headers'] += 1
            
            # Check for lists
            if re.search(r'^[-*+]\s', source, re.MULTILINE):
                quality_metrics['cells_with_lists'] += 1
            
            # Check for links
            if re.search(r'\[.*?\]\(.*?\)', source):
                quality_metrics['cells_with_links'] += 1
            
            # Check for emphasis
            if re.search(r'\*\*.*?\*\*|__.*?__|_.*?_|\*.*?\*', source):
                quality_metrics['cells_with_emphasis'] += 1
            
            # Count words
            words = len(source.split())
            quality_metrics['total_words'] += words
        
        quality_metrics['avg_words_per_cell'] = quality_metrics['total_words'] / quality_metrics['total_cells']
        
        # Calculate score
        header_score = (quality_metrics['cells_with_headers'] / quality_metrics['total_cells']) * 25
        list_score = (quality_metrics['cells_with_lists'] / quality_metrics['total_cells']) * 25
        link_score = (quality_metrics['cells_with_links'] / quality_metrics['total_cells']) * 25
        emphasis_score = (quality_metrics['cells_with_emphasis'] / quality_metrics['total_cells']) * 25
        
        score = header_score + list_score + link_score + emphasis_score
        
        passed = (
            quality_metrics['cells_with_headers'] > 0 and
            quality_metrics['cells_with_lists'] > 0 and
            quality_metrics['avg_words_per_cell'] >= 10
        )
        
        message = f"Headers: {quality_metrics['cells_with_headers']}, Lists: {quality_metrics['cells_with_lists']}, Avg words: {quality_metrics['avg_words_per_cell']:.1f}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details=quality_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def _test_cross_references(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test cross-references."""
        test_name = "Cross References"
        
        # This is a simplified implementation
        # In a real scenario, you would check against a database of available notebooks
        
        markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
        
        cross_refs = []
        internal_links = []
        
        for cell in markdown_cells:
            source = cell.source
            
            # Look for notebook references
            notebook_refs = re.findall(r'\[.*?\]\(.*?\.ipynb\)', source)
            cross_refs.extend(notebook_refs)
            
            # Look for internal links
            internal_ref_pattern = r'\[.*?\]\(#.*?\)'
            internal_links.extend(re.findall(internal_ref_pattern, source))
        
        score = min(len(cross_refs) * 20, 100)  # Up to 5 cross-references for full score
        passed = len(cross_refs) > 0 or len(internal_links) > 2
        
        message = f"Cross-references: {len(cross_refs)}, Internal links: {len(internal_links)}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'cross_references': cross_refs,
                'internal_links': internal_links
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_performance_indicators(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test performance indicators."""
        test_name = "Performance Indicators"
        
        performance_issues = []
        performance_warnings = []
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                
                # Check for performance issues
                if 'iterrows' in source:
                    performance_issues.append("Using iterrows() - consider vectorization")
                
                if 'for ' in source and 'range(len(' in source:
                    performance_issues.append("Using range(len()) - consider enumerate() or direct iteration")
                
                if 'read_csv' in source and 'chunksize' not in source:
                    performance_warnings.append("Reading CSV without chunking - may cause memory issues")
                
                if '+=' in source and 'DataFrame' in source:
                    performance_issues.append("Growing DataFrame in loop - consider list comprehension")
        
        # Score based on performance issues
        issue_penalty = len(performance_issues) * 20
        warning_penalty = len(performance_warnings) * 10
        
        score = max(0, 100 - issue_penalty - warning_penalty)
        passed = len(performance_issues) == 0
        
        message = f"Performance issues: {len(performance_issues)}, Warnings: {len(performance_warnings)}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'performance_issues': performance_issues,
                'performance_warnings': performance_warnings
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_accessibility(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test accessibility features."""
        test_name = "Accessibility"
        
        accessibility_features = {
            'alt_text_count': 0,
            'heading_structure': False,
            'descriptive_links': 0,
            'color_descriptions': 0
        }
        
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                source = cell.source
                
                # Check for alt text in images
                img_pattern = r'!\[([^\]]*)\]\([^)]*\)'
                images = re.findall(img_pattern, source)
                accessibility_features['alt_text_count'] += sum(1 for alt in images if alt.strip())
                
                # Check for heading structure
                if re.search(r'^#+\s', source, re.MULTILINE):
                    accessibility_features['heading_structure'] = True
                
                # Check for descriptive links
                link_pattern = r'\[([^\]]+)\]\([^)]*\)'
                links = re.findall(link_pattern, source)
                accessibility_features['descriptive_links'] += sum(1 for link in links if len(link) > 5)
        
        # Calculate score
        score = 0
        if accessibility_features['alt_text_count'] > 0:
            score += 30
        if accessibility_features['heading_structure']:
            score += 40
        if accessibility_features['descriptive_links'] > 0:
            score += 30
        
        passed = accessibility_features['heading_structure'] and accessibility_features['descriptive_links'] > 0
        
        message = f"Alt text: {accessibility_features['alt_text_count']}, Headings: {accessibility_features['heading_structure']}, Descriptive links: {accessibility_features['descriptive_links']}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details=accessibility_features,
            timestamp=datetime.now().isoformat()
        )
    
    def _test_link_validation(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test link validation."""
        test_name = "Link Validation"
        
        if not self.config['links']['validate_external_links']:
            return TestResult(
                test_name=test_name,
                passed=True,
                score=100.0,
                message="Link validation disabled",
                details={},
                timestamp=datetime.now().isoformat()
            )
        
        links = []
        broken_links = []
        
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                source = cell.source
                
                # Extract links
                link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                cell_links = re.findall(link_pattern, source)
                
                for link_text, link_url in cell_links:
                    if link_url.startswith('http'):
                        links.append((link_text, link_url))
        
        # Validate external links
        for link_text, link_url in links:
            try:
                response = requests.head(link_url, timeout=self.config['links']['timeout'])
                if response.status_code >= 400:
                    broken_links.append((link_text, link_url, response.status_code))
            except Exception as e:
                broken_links.append((link_text, link_url, str(e)))
        
        if not links:
            score = 100.0
            passed = True
            message = "No external links to validate"
        else:
            score = ((len(links) - len(broken_links)) / len(links)) * 100
            passed = len(broken_links) == 0
            message = f"Validated {len(links)} links, {len(broken_links)} broken"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'total_links': len(links),
                'broken_links': broken_links,
                'all_links': links
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_image_validation(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test image validation."""
        test_name = "Image Validation"
        
        images = []
        missing_images = []
        
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                source = cell.source
                
                # Extract image references
                img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                cell_images = re.findall(img_pattern, source)
                
                for alt_text, img_path in cell_images:
                    images.append((alt_text, img_path))
                    
                    # Check if local image exists
                    if not img_path.startswith('http'):
                        img_full_path = notebook_path.parent / img_path
                        if not img_full_path.exists():
                            missing_images.append((alt_text, img_path))
        
        if not images:
            score = 100.0
            passed = True
            message = "No images to validate"
        else:
            score = ((len(images) - len(missing_images)) / len(images)) * 100
            passed = len(missing_images) == 0
            message = f"Validated {len(images)} images, {len(missing_images)} missing"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'total_images': len(images),
                'missing_images': missing_images,
                'all_images': images
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_notebook_execution(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test notebook execution."""
        test_name = "Notebook Execution"
        
        # This is a simplified test - in practice, you'd want to use nbconvert or papermill
        execution_errors = []
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                try:
                    # Check for basic syntax errors
                    compile(cell.source, f'<cell {i}>', 'exec')
                except SyntaxError as e:
                    execution_errors.append(f"Cell {i}: {str(e)}")
                except Exception as e:
                    execution_errors.append(f"Cell {i}: {str(e)}")
        
        score = max(0, 100 - len(execution_errors) * 20)
        passed = len(execution_errors) == 0
        
        message = f"Execution errors: {len(execution_errors)}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'execution_errors': execution_errors
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_cell_ordering(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test logical cell ordering."""
        test_name = "Cell Ordering"
        
        # Check for imports at the beginning
        import_cells = []
        first_code_cell_idx = None
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                if first_code_cell_idx is None:
                    first_code_cell_idx = i
                
                if 'import ' in cell.source or 'from ' in cell.source:
                    import_cells.append(i)
        
        # Imports should generally be at the beginning
        early_imports = sum(1 for idx in import_cells if idx <= first_code_cell_idx + 2)
        import_score = (early_imports / max(len(import_cells), 1)) * 100 if import_cells else 100
        
        passed = import_score >= 80
        score = import_score
        
        message = f"Import organization score: {import_score:.1f}%"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'import_cells': import_cells,
                'first_code_cell': first_code_cell_idx,
                'early_imports': early_imports
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_imports_documentation(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test imports documentation."""
        test_name = "Imports Documentation"
        
        import_cells = []
        documented_imports = 0
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and ('import ' in cell.source or 'from ' in cell.source):
                import_cells.append(i)
                
                # Check if previous cell or same cell has documentation
                has_doc = False
                
                # Check same cell for comments
                if '#' in cell.source:
                    has_doc = True
                
                # Check previous cell for documentation
                if i > 0 and nb.cells[i-1].cell_type == 'markdown':
                    prev_content = nb.cells[i-1].source.lower()
                    if any(word in prev_content for word in ['import', 'library', 'libraries', 'dependencies']):
                        has_doc = True
                
                if has_doc:
                    documented_imports += 1
        
        if not import_cells:
            score = 100.0
            passed = True
            message = "No imports to document"
        else:
            score = (documented_imports / len(import_cells)) * 100
            passed = documented_imports / len(import_cells) >= 0.8
            message = f"Documented imports: {documented_imports}/{len(import_cells)}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'import_cells': import_cells,
                'documented_imports': documented_imports
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _test_output_documentation(self, nb: nbformat.NotebookNode, notebook_path: Path) -> TestResult:
        """Test output documentation."""
        test_name = "Output Documentation"
        
        cells_with_outputs = []
        documented_outputs = 0
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and hasattr(cell, 'outputs') and cell.outputs:
                cells_with_outputs.append(i)
                
                # Check if next cell explains the output
                if i + 1 < len(nb.cells) and nb.cells[i+1].cell_type == 'markdown':
                    next_content = nb.cells[i+1].source.lower()
                    if any(word in next_content for word in ['output', 'result', 'shows', 'displays']):
                        documented_outputs += 1
        
        if not cells_with_outputs:
            score = 100.0
            passed = True
            message = "No outputs to document"
        else:
            score = (documented_outputs / len(cells_with_outputs)) * 100
            passed = documented_outputs / len(cells_with_outputs) >= 0.5
            message = f"Documented outputs: {documented_outputs}/{len(cells_with_outputs)}"
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message,
            details={
                'cells_with_outputs': cells_with_outputs,
                'documented_outputs': documented_outputs
            },
            timestamp=datetime.now().isoformat()
        )
    
    def test_directory(self, directory_path: str, recursive: bool = True) -> List[NotebookTestSuite]:
        """
        Test all notebooks in a directory.
        
        Parameters
        ----------
        directory_path : str
            Path to directory containing notebooks
        recursive : bool
            Whether to search recursively
            
        Returns
        -------
        List[NotebookTestSuite]
            Test results for all notebooks
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find notebooks
        if recursive:
            notebook_files = list(directory_path.rglob("*.ipynb"))
        else:
            notebook_files = list(directory_path.glob("*.ipynb"))
        
        logger.info(f"Found {len(notebook_files)} notebooks to test")
        
        test_suites = []
        
        for notebook_path in notebook_files:
            try:
                test_suite = self.test_notebook(str(notebook_path))
                test_suites.append(test_suite)
            except Exception as e:
                logger.error(f"Error testing {notebook_path}: {e}")
        
        return test_suites
    
    def generate_report(self, test_suites: List[NotebookTestSuite], output_file: str = "documentation_test_report.json") -> str:
        """
        Generate comprehensive test report.
        
        Parameters
        ----------
        test_suites : List[NotebookTestSuite]
            Test results to include in report
        output_file : str
            Output file for the report
            
        Returns
        -------
        str
            Path to the generated report
        """
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_notebooks': len(test_suites),
                'total_tests': sum(suite.total_tests for suite in test_suites),
                'total_passed': sum(suite.passed_tests for suite in test_suites),
                'total_failed': sum(suite.failed_tests for suite in test_suites),
                'average_score': sum(suite.overall_score for suite in test_suites) / len(test_suites) if test_suites else 0
            },
            'test_suites': [asdict(suite) for suite in test_suites],
            'summary': {
                'notebooks_by_score': {},
                'common_issues': {},
                'recommendations': []
            }
        }
        
        # Analyze common issues
        all_results = []
        for suite in test_suites:
            all_results.extend(suite.test_results)
        
        failed_tests = [r for r in all_results if not r.passed]
        test_failure_counts = {}
        
        for result in failed_tests:
            test_name = result.test_name
            if test_name not in test_failure_counts:
                test_failure_counts[test_name] = 0
            test_failure_counts[test_name] += 1
        
        report['summary']['common_issues'] = test_failure_counts
        
        # Generate recommendations
        recommendations = []
        if test_failure_counts.get('Header Completeness', 0) > 0:
            recommendations.append("Add comprehensive header cells to notebooks")
        if test_failure_counts.get('Function Documentation', 0) > 0:
            recommendations.append("Add docstrings to functions")
        if test_failure_counts.get('Code Quality', 0) > 0:
            recommendations.append("Improve code quality with comments and better structure")
        
        report['summary']['recommendations'] = recommendations
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report generated: {output_file}")
        return output_file

def main():
    """Main function to run documentation tests."""
    parser = argparse.ArgumentParser(description="Test Jupyter notebook documentation quality")
    parser.add_argument("--notebook", help="Path to a single notebook to test")
    parser.add_argument("--directory", help="Path to directory containing notebooks")
    parser.add_argument("--recursive", action="store_true", help="Search recursively in subdirectories")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", default="documentation_test_report.json", help="Output file for test report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = DocumentationTester(config_file=args.config)
    
    try:
        if args.notebook:
            # Test single notebook
            test_suite = tester.test_notebook(args.notebook)
            print(f"Test Results for {test_suite.notebook_name}:")
            print(f"  Passed: {test_suite.passed_tests}/{test_suite.total_tests}")
            print(f"  Overall Score: {test_suite.overall_score:.1f}/100")
            
            # Generate report
            report_file = tester.generate_report([test_suite], args.output)
            print(f"Report saved to: {report_file}")
            
        elif args.directory:
            # Test directory of notebooks
            test_suites = tester.test_directory(args.directory, args.recursive)
            
            print(f"Test Results for {len(test_suites)} notebooks:")
            total_tests = sum(suite.total_tests for suite in test_suites)
            total_passed = sum(suite.passed_tests for suite in test_suites)
            avg_score = sum(suite.overall_score for suite in test_suites) / len(test_suites)
            
            print(f"  Total Tests: {total_passed}/{total_tests}")
            print(f"  Average Score: {avg_score:.1f}/100")
            
            # Generate report
            report_file = tester.generate_report(test_suites, args.output)
            print(f"Report saved to: {report_file}")
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 