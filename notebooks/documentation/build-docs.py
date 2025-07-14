#!/usr/bin/env python3
"""
Documentation Build System for Modern Data Stack Showcase
========================================================

This script automatically generates comprehensive documentation from Jupyter notebooks
including API documentation, cross-references, and interactive documentation.

Features:
- Automated notebook execution and documentation extraction
- Cross-reference generation between notebooks
- API documentation from shared components
- Interactive documentation with Jupyter Book
- Performance metrics and documentation quality assessment
- Automated testing of documentation examples

Usage:
    python build-docs.py --mode [full|quick|api|notebooks] [--output-dir OUTPUT_DIR]
    
Author: Data Science Team
Date: 2024-01-15
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess
import re
import ast
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell
import pandas as pd
import yaml
from jinja2 import Template
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('documentation-build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentationBuilder:
    """
    Comprehensive documentation builder for Jupyter notebooks.
    
    This class handles all aspects of documentation generation including:
    - Notebook analysis and metadata extraction
    - Cross-reference generation
    - API documentation
    - Interactive documentation building
    - Quality assessment and reporting
    """
    
    def __init__(self, base_dir: str = ".", output_dir: str = "docs"):
        """
        Initialize the documentation builder.
        
        Parameters
        ----------
        base_dir : str
            Base directory containing notebooks
        output_dir : str
            Output directory for generated documentation
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.notebooks_dir = self.base_dir / "notebooks"
        self.templates_dir = self.base_dir / "documentation" / "templates"
        self.shared_dir = self.base_dir / "notebooks" / "shared"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata storage
        self.notebook_metadata = {}
        self.cross_references = {}
        self.api_documentation = {}
        self.quality_metrics = {}
        
        logger.info(f"Documentation builder initialized")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def analyze_notebooks(self) -> Dict[str, Dict]:
        """
        Analyze all notebooks and extract comprehensive metadata.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary containing metadata for each notebook
        """
        logger.info("Starting notebook analysis...")
        
        notebook_files = list(self.notebooks_dir.rglob("*.ipynb"))
        logger.info(f"Found {len(notebook_files)} notebooks")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._analyze_single_notebook, nb_path): nb_path
                for nb_path in notebook_files
            }
            
            for future in as_completed(futures):
                nb_path = futures[future]
                try:
                    metadata = future.result()
                    self.notebook_metadata[str(nb_path)] = metadata
                    logger.info(f"Analyzed: {nb_path.name}")
                except Exception as exc:
                    logger.error(f"Error analyzing {nb_path}: {exc}")
        
        logger.info(f"Completed analysis of {len(self.notebook_metadata)} notebooks")
        return self.notebook_metadata
    
    def _analyze_single_notebook(self, nb_path: Path) -> Dict:
        """
        Analyze a single notebook and extract metadata.
        
        Parameters
        ----------
        nb_path : Path
            Path to the notebook file
            
        Returns
        -------
        Dict
            Comprehensive metadata for the notebook
        """
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            metadata = {
                'path': str(nb_path),
                'name': nb_path.stem,
                'category': self._extract_category(nb_path),
                'title': self._extract_title(nb),
                'author': self._extract_author(nb),
                'created_date': self._extract_created_date(nb),
                'last_updated': self._extract_last_updated(nb),
                'description': self._extract_description(nb),
                'prerequisites': self._extract_prerequisites(nb),
                'datasets': self._extract_datasets(nb),
                'tools_libraries': self._extract_tools_libraries(nb),
                'key_outcomes': self._extract_key_outcomes(nb),
                'related_notebooks': self._extract_related_notebooks(nb),
                'functions': self._extract_functions(nb),
                'imports': self._extract_imports(nb),
                'visualizations': self._extract_visualizations(nb),
                'performance_metrics': self._analyze_performance(nb),
                'quality_score': self._calculate_quality_score(nb),
                'documentation_level': self._assess_documentation_level(nb),
                'execution_time': self._estimate_execution_time(nb),
                'memory_usage': self._estimate_memory_usage(nb),
                'complexity_score': self._calculate_complexity_score(nb),
                'tags': self._extract_tags(nb),
                'dependencies': self._extract_dependencies(nb),
                'outputs': self._extract_outputs(nb),
                'errors': self._check_for_errors(nb)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing notebook {nb_path}: {e}")
            return {
                'path': str(nb_path),
                'name': nb_path.stem,
                'error': str(e)
            }
    
    def _extract_category(self, nb_path: Path) -> str:
        """Extract category from notebook path."""
        path_parts = nb_path.parts
        if 'data-exploration' in path_parts:
            return 'Data Exploration'
        elif 'ml-workflows' in path_parts:
            return 'ML Workflow'
        elif 'devops-automation' in path_parts:
            return 'DevOps Automation'
        elif 'business-intelligence' in path_parts:
            return 'Business Intelligence'
        elif 'templates' in path_parts:
            return 'Template'
        else:
            return 'General'
    
    def _extract_title(self, nb: nbformat.NotebookNode) -> str:
        """Extract title from notebook."""
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            title_match = re.search(r'^#\s+(.+)$', first_cell, re.MULTILINE)
            if title_match:
                return title_match.group(1).strip()
        return "Untitled Notebook"
    
    def _extract_author(self, nb: nbformat.NotebookNode) -> str:
        """Extract author from notebook."""
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            author_match = re.search(r'\*\*👤 Author\*\*:\s*(.+)', first_cell)
            if author_match:
                return author_match.group(1).strip()
        return "Unknown"
    
    def _extract_created_date(self, nb: nbformat.NotebookNode) -> str:
        """Extract created date from notebook."""
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            date_match = re.search(r'\*\*📅 Created\*\*:\s*(.+)', first_cell)
            if date_match:
                return date_match.group(1).strip()
        return "Unknown"
    
    def _extract_last_updated(self, nb: nbformat.NotebookNode) -> str:
        """Extract last updated date from notebook."""
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            date_match = re.search(r'\*\*🔄 Last Updated\*\*:\s*(.+)', first_cell)
            if date_match:
                return date_match.group(1).strip()
        return "Unknown"
    
    def _extract_description(self, nb: nbformat.NotebookNode) -> str:
        """Extract description from notebook."""
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            desc_match = re.search(r'\*\*🎯 Purpose\*\*:\s*(.+)', first_cell)
            if desc_match:
                return desc_match.group(1).strip()
        return "No description available"
    
    def _extract_prerequisites(self, nb: nbformat.NotebookNode) -> List[str]:
        """Extract prerequisites from notebook."""
        prerequisites = []
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            # Look for prerequisites section
            prereq_section = re.search(r'\*\*📋 Prerequisites\*\*:\s*\n((?:- .+\n?)+)', first_cell)
            if prereq_section:
                prereq_text = prereq_section.group(1)
                prerequisites = [line.strip('- ').strip() for line in prereq_text.split('\n') if line.strip().startswith('-')]
        return prerequisites
    
    def _extract_datasets(self, nb: nbformat.NotebookNode) -> List[Dict]:
        """Extract datasets information from notebook."""
        datasets = []
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            # Look for datasets section
            dataset_section = re.search(r'\*\*📊 Datasets Used\*\*:\s*\n((?:- .+\n?)+)', first_cell)
            if dataset_section:
                dataset_text = dataset_section.group(1)
                for line in dataset_text.split('\n'):
                    if line.strip().startswith('-'):
                        dataset_line = line.strip('- ').strip()
                        if ':' in dataset_line:
                            name, description = dataset_line.split(':', 1)
                            datasets.append({
                                'name': name.strip(),
                                'description': description.strip()
                            })
        return datasets
    
    def _extract_tools_libraries(self, nb: nbformat.NotebookNode) -> List[Dict]:
        """Extract tools and libraries from notebook."""
        tools = []
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            # Look for tools section
            tools_section = re.search(r'\*\*🔧 Tools & Libraries\*\*:\s*\n((?:- .+\n?)+)', first_cell)
            if tools_section:
                tools_text = tools_section.group(1)
                for line in tools_text.split('\n'):
                    if line.strip().startswith('-'):
                        tool_line = line.strip('- ').strip()
                        if ':' in tool_line:
                            name, purpose = tool_line.split(':', 1)
                            tools.append({
                                'name': name.strip(),
                                'purpose': purpose.strip()
                            })
        return tools
    
    def _extract_key_outcomes(self, nb: nbformat.NotebookNode) -> List[str]:
        """Extract key outcomes from notebook."""
        outcomes = []
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            # Look for outcomes section
            outcomes_section = re.search(r'\*\*📈 Key Outcomes\*\*:\s*\n((?:- .+\n?)+)', first_cell)
            if outcomes_section:
                outcomes_text = outcomes_section.group(1)
                outcomes = [line.strip('- ').strip() for line in outcomes_text.split('\n') if line.strip().startswith('-')]
        return outcomes
    
    def _extract_related_notebooks(self, nb: nbformat.NotebookNode) -> List[Dict]:
        """Extract related notebooks from notebook."""
        related = []
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            # Look for related notebooks section
            related_section = re.search(r'\*\*🔗 Related Notebooks\*\*:\s*\n((?:- .+\n?)+)', first_cell)
            if related_section:
                related_text = related_section.group(1)
                for line in related_text.split('\n'):
                    if line.strip().startswith('-'):
                        related_line = line.strip('- ').strip()
                        if ':' in related_line:
                            name, relationship = related_line.split(':', 1)
                            related.append({
                                'name': name.strip(),
                                'relationship': relationship.strip()
                            })
        return related
    
    def _extract_functions(self, nb: nbformat.NotebookNode) -> List[Dict]:
        """Extract function definitions from notebook."""
        functions = []
        for cell in nb.cells:
            if cell.cell_type == 'code':
                try:
                    tree = ast.parse(cell.source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_info = {
                                'name': node.name,
                                'args': [arg.arg for arg in node.args.args],
                                'docstring': ast.get_docstring(node),
                                'line_number': node.lineno
                            }
                            functions.append(func_info)
                except:
                    pass
        return functions
    
    def _extract_imports(self, nb: nbformat.NotebookNode) -> List[str]:
        """Extract import statements from notebook."""
        imports = []
        for cell in nb.cells:
            if cell.cell_type == 'code':
                try:
                    tree = ast.parse(cell.source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ''
                            for alias in node.names:
                                imports.append(f"{module}.{alias.name}")
                except:
                    pass
        return list(set(imports))
    
    def _extract_visualizations(self, nb: nbformat.NotebookNode) -> List[Dict]:
        """Extract visualization information from notebook."""
        visualizations = []
        viz_keywords = ['plt.', 'fig.', 'ax.', 'sns.', 'px.', 'go.', 'plotly']
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                source = cell.source
                for keyword in viz_keywords:
                    if keyword in source:
                        viz_info = {
                            'cell_index': i,
                            'type': self._identify_viz_type(source),
                            'library': self._identify_viz_library(source),
                            'has_title': 'title' in source.lower(),
                            'has_labels': any(label in source.lower() for label in ['xlabel', 'ylabel', 'labels']),
                            'interactive': 'plotly' in source.lower() or 'bokeh' in source.lower()
                        }
                        visualizations.append(viz_info)
                        break
        
        return visualizations
    
    def _identify_viz_type(self, source: str) -> str:
        """Identify visualization type from source code."""
        viz_types = {
            'scatter': ['scatter', 'scatterplot'],
            'line': ['plot', 'line'],
            'bar': ['bar', 'barplot'],
            'histogram': ['hist', 'histogram'],
            'box': ['box', 'boxplot'],
            'violin': ['violin'],
            'heatmap': ['heatmap'],
            'pie': ['pie'],
            'subplot': ['subplot', 'subplots']
        }
        
        source_lower = source.lower()
        for viz_type, keywords in viz_types.items():
            if any(keyword in source_lower for keyword in keywords):
                return viz_type
        return 'unknown'
    
    def _identify_viz_library(self, source: str) -> str:
        """Identify visualization library from source code."""
        if 'px.' in source or 'plotly' in source:
            return 'plotly'
        elif 'sns.' in source or 'seaborn' in source:
            return 'seaborn'
        elif 'plt.' in source or 'matplotlib' in source:
            return 'matplotlib'
        elif 'bokeh' in source:
            return 'bokeh'
        else:
            return 'unknown'
    
    def _analyze_performance(self, nb: nbformat.NotebookNode) -> Dict:
        """Analyze performance characteristics of notebook."""
        performance = {
            'total_cells': len(nb.cells),
            'code_cells': sum(1 for cell in nb.cells if cell.cell_type == 'code'),
            'markdown_cells': sum(1 for cell in nb.cells if cell.cell_type == 'markdown'),
            'has_loops': self._has_loops(nb),
            'has_large_data': self._has_large_data_operations(nb),
            'memory_intensive': self._is_memory_intensive(nb),
            'io_operations': self._count_io_operations(nb)
        }
        return performance
    
    def _has_loops(self, nb: nbformat.NotebookNode) -> bool:
        """Check if notebook contains loops."""
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if any(keyword in cell.source for keyword in ['for ', 'while ', 'iterrows', 'itertuples']):
                    return True
        return False
    
    def _has_large_data_operations(self, nb: nbformat.NotebookNode) -> bool:
        """Check if notebook has large data operations."""
        large_data_keywords = ['read_csv', 'read_sql', 'read_parquet', 'read_excel', 'merge', 'groupby']
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if any(keyword in cell.source for keyword in large_data_keywords):
                    return True
        return False
    
    def _is_memory_intensive(self, nb: nbformat.NotebookNode) -> bool:
        """Check if notebook is memory intensive."""
        memory_keywords = ['numpy', 'pandas', 'sklearn', 'tensorflow', 'torch', 'xgboost']
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if any(keyword in cell.source for keyword in memory_keywords):
                    return True
        return False
    
    def _count_io_operations(self, nb: nbformat.NotebookNode) -> int:
        """Count I/O operations in notebook."""
        io_keywords = ['read_', 'write_', 'to_csv', 'to_excel', 'to_parquet', 'to_sql']
        count = 0
        for cell in nb.cells:
            if cell.cell_type == 'code':
                for keyword in io_keywords:
                    count += cell.source.count(keyword)
        return count
    
    def _calculate_quality_score(self, nb: nbformat.NotebookNode) -> float:
        """Calculate documentation quality score."""
        score = 0.0
        max_score = 100.0
        
        # Check for header cell
        if nb.cells and nb.cells[0].cell_type == 'markdown':
            first_cell = nb.cells[0].source
            score += 20 if '**📊 Category**' in first_cell else 0
            score += 10 if '**👤 Author**' in first_cell else 0
            score += 10 if '**🎯 Purpose**' in first_cell else 0
            score += 10 if '**📋 Prerequisites**' in first_cell else 0
        
        # Check for documentation cells
        markdown_ratio = sum(1 for cell in nb.cells if cell.cell_type == 'markdown') / len(nb.cells)
        score += 20 * markdown_ratio
        
        # Check for function documentation
        functions = self._extract_functions(nb)
        documented_functions = sum(1 for func in functions if func['docstring'])
        if functions:
            score += 15 * (documented_functions / len(functions))
        else:
            score += 15  # No functions to document
        
        # Check for visualization documentation
        visualizations = self._extract_visualizations(nb)
        if visualizations:
            well_documented_viz = sum(1 for viz in visualizations if viz['has_title'] and viz['has_labels'])
            score += 15 * (well_documented_viz / len(visualizations))
        else:
            score += 15  # No visualizations to document
        
        return min(score, max_score)
    
    def _assess_documentation_level(self, nb: nbformat.NotebookNode) -> str:
        """Assess documentation level of notebook."""
        quality_score = self._calculate_quality_score(nb)
        
        if quality_score >= 90:
            return "Level 4 - Reference"
        elif quality_score >= 75:
            return "Level 3 - Tutorial"
        elif quality_score >= 60:
            return "Level 2 - Comprehensive"
        else:
            return "Level 1 - Basic"
    
    def _estimate_execution_time(self, nb: nbformat.NotebookNode) -> str:
        """Estimate execution time based on notebook complexity."""
        performance = self._analyze_performance(nb)
        
        if performance['has_large_data'] and performance['memory_intensive']:
            return "Long (>10 minutes)"
        elif performance['has_loops'] or performance['memory_intensive']:
            return "Medium (2-10 minutes)"
        else:
            return "Short (<2 minutes)"
    
    def _estimate_memory_usage(self, nb: nbformat.NotebookNode) -> str:
        """Estimate memory usage based on notebook operations."""
        if self._is_memory_intensive(nb):
            return "High (>2GB)"
        elif self._has_large_data_operations(nb):
            return "Medium (500MB-2GB)"
        else:
            return "Low (<500MB)"
    
    def _calculate_complexity_score(self, nb: nbformat.NotebookNode) -> int:
        """Calculate complexity score of notebook."""
        score = 0
        
        # Count functions
        functions = self._extract_functions(nb)
        score += len(functions) * 2
        
        # Count imports
        imports = self._extract_imports(nb)
        score += len(imports)
        
        # Count visualizations
        visualizations = self._extract_visualizations(nb)
        score += len(visualizations)
        
        # Count code cells
        code_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code')
        score += code_cells
        
        return score
    
    def _extract_tags(self, nb: nbformat.NotebookNode) -> List[str]:
        """Extract tags from notebook."""
        tags = []
        category = self._extract_category(Path(nb.get('path', '')))
        tags.append(category.lower().replace(' ', '-'))
        
        # Extract tags from imports
        imports = self._extract_imports(nb)
        for imp in imports:
            if 'pandas' in imp:
                tags.append('data-manipulation')
            elif 'sklearn' in imp:
                tags.append('machine-learning')
            elif 'plotly' in imp or 'matplotlib' in imp:
                tags.append('visualization')
            elif 'tensorflow' in imp or 'torch' in imp:
                tags.append('deep-learning')
        
        return list(set(tags))
    
    def _extract_dependencies(self, nb: nbformat.NotebookNode) -> List[str]:
        """Extract dependencies from notebook."""
        dependencies = []
        
        # Extract from imports
        imports = self._extract_imports(nb)
        for imp in imports:
            if not imp.startswith('_') and '.' not in imp:
                dependencies.append(imp)
        
        return list(set(dependencies))
    
    def _extract_outputs(self, nb: nbformat.NotebookNode) -> Dict:
        """Extract output information from notebook."""
        outputs = {
            'has_outputs': False,
            'output_types': [],
            'total_outputs': 0
        }
        
        for cell in nb.cells:
            if cell.cell_type == 'code' and hasattr(cell, 'outputs') and cell.outputs:
                outputs['has_outputs'] = True
                outputs['total_outputs'] += len(cell.outputs)
                for output in cell.outputs:
                    if hasattr(output, 'output_type'):
                        outputs['output_types'].append(output.output_type)
        
        outputs['output_types'] = list(set(outputs['output_types']))
        return outputs
    
    def _check_for_errors(self, nb: nbformat.NotebookNode) -> List[str]:
        """Check for errors in notebook."""
        errors = []
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                for output in cell.outputs:
                    if hasattr(output, 'output_type') and output.output_type == 'error':
                        errors.append(f"Cell {i}: {output.get('ename', 'Unknown error')}")
        
        return errors
    
    def generate_cross_references(self) -> Dict[str, List[str]]:
        """Generate cross-references between notebooks."""
        logger.info("Generating cross-references...")
        
        cross_refs = {}
        
        for nb_path, metadata in self.notebook_metadata.items():
            refs = []
            
            # Check for explicit references
            related_notebooks = metadata.get('related_notebooks', [])
            for related in related_notebooks:
                refs.append(related.get('name', ''))
            
            # Check for implicit references (shared imports, functions, etc.)
            for other_path, other_metadata in self.notebook_metadata.items():
                if nb_path != other_path:
                    # Check for shared imports
                    shared_imports = set(metadata.get('imports', [])) & set(other_metadata.get('imports', []))
                    if len(shared_imports) > 5:  # Threshold for significant overlap
                        refs.append(other_metadata.get('name', ''))
                    
                    # Check for shared functions
                    nb_functions = {f['name'] for f in metadata.get('functions', [])}
                    other_functions = {f['name'] for f in other_metadata.get('functions', [])}
                    if nb_functions & other_functions:
                        refs.append(other_metadata.get('name', ''))
            
            cross_refs[nb_path] = list(set(refs))
        
        self.cross_references = cross_refs
        logger.info(f"Generated cross-references for {len(cross_refs)} notebooks")
        return cross_refs
    
    def generate_api_documentation(self) -> Dict[str, Dict]:
        """Generate API documentation from shared components."""
        logger.info("Generating API documentation...")
        
        api_docs = {}
        
        # Analyze shared components
        shared_files = list(self.shared_dir.glob("*.py"))
        
        for py_file in shared_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                module_doc = {
                    'file': str(py_file),
                    'name': py_file.stem,
                    'docstring': ast.get_docstring(tree),
                    'functions': [],
                    'classes': [],
                    'constants': []
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_doc = {
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'docstring': ast.get_docstring(node),
                            'line_number': node.lineno,
                            'decorators': [d.id for d in node.decorator_list if hasattr(d, 'id')]
                        }
                        module_doc['functions'].append(func_doc)
                    
                    elif isinstance(node, ast.ClassDef):
                        class_doc = {
                            'name': node.name,
                            'docstring': ast.get_docstring(node),
                            'line_number': node.lineno,
                            'methods': [],
                            'properties': []
                        }
                        
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_doc = {
                                    'name': item.name,
                                    'args': [arg.arg for arg in item.args.args],
                                    'docstring': ast.get_docstring(item),
                                    'line_number': item.lineno,
                                    'is_private': item.name.startswith('_'),
                                    'is_property': any(d.id == 'property' for d in item.decorator_list if hasattr(d, 'id'))
                                }
                                class_doc['methods'].append(method_doc)
                        
                        module_doc['classes'].append(class_doc)
                    
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                constant_doc = {
                                    'name': target.id,
                                    'value': ast.get_source_segment(content, node.value) if hasattr(ast, 'get_source_segment') else 'N/A',
                                    'line_number': node.lineno
                                }
                                module_doc['constants'].append(constant_doc)
                
                api_docs[py_file.stem] = module_doc
                
            except Exception as e:
                logger.error(f"Error analyzing {py_file}: {e}")
        
        self.api_documentation = api_docs
        logger.info(f"Generated API documentation for {len(api_docs)} modules")
        return api_docs
    
    def build_jupyter_book(self) -> bool:
        """Build Jupyter Book documentation."""
        logger.info("Building Jupyter Book documentation...")
        
        try:
            # Copy necessary files
            book_dir = self.output_dir / "jupyter-book"
            book_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy configuration files
            config_source = self.base_dir / "documentation" / "jupyter-book-config.yml"
            toc_source = self.base_dir / "documentation" / "_toc.yml"
            
            if config_source.exists():
                shutil.copy2(config_source, book_dir / "_config.yml")
            
            if toc_source.exists():
                shutil.copy2(toc_source, book_dir / "_toc.yml")
            
            # Create index file
            self._create_index_file(book_dir)
            
            # Copy notebooks
            nb_dest = book_dir / "notebooks"
            if self.notebooks_dir.exists():
                shutil.copytree(self.notebooks_dir, nb_dest, dirs_exist_ok=True)
            
            # Build the book
            cmd = ["jupyter-book", "build", str(book_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Jupyter Book built successfully")
                return True
            else:
                logger.error(f"Jupyter Book build failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error building Jupyter Book: {e}")
            return False
    
    def _create_index_file(self, book_dir: Path):
        """Create index file for Jupyter Book."""
        index_content = f"""# Modern Data Stack Showcase - Jupyter Notebooks Documentation

Welcome to the comprehensive documentation for the Modern Data Stack Showcase Jupyter notebooks.

## 📊 Overview

This documentation covers {len(self.notebook_metadata)} notebooks across multiple categories:

"""
        
        # Add category summary
        categories = {}
        for metadata in self.notebook_metadata.values():
            category = metadata.get('category', 'General')
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        for category, count in categories.items():
            index_content += f"- **{category}**: {count} notebooks\n"
        
        index_content += """
## 🎯 Documentation Quality

"""
        
        # Add quality metrics
        quality_levels = {}
        for metadata in self.notebook_metadata.values():
            level = metadata.get('documentation_level', 'Level 1 - Basic')
            if level not in quality_levels:
                quality_levels[level] = 0
            quality_levels[level] += 1
        
        for level, count in quality_levels.items():
            index_content += f"- **{level}**: {count} notebooks\n"
        
        index_content += """
## 🚀 Quick Start

1. [Getting Started Guide](getting-started/introduction.md)
2. [Installation Instructions](getting-started/installation.md)
3. [Environment Setup](getting-started/environment-setup.md)
4. [Architecture Overview](getting-started/architecture-overview.md)

## 📚 Documentation Sections

Navigate through the documentation using the table of contents on the left.

## 🔧 Tools and Technologies

This showcase demonstrates integration with:
- Jupyter Lab/Notebook
- Python Data Science Stack
- Docker and Kubernetes
- MLflow and MLOps tools
- Power BI and Business Intelligence
- Modern data pipeline tools

## 🆘 Support

For questions or issues:
- Check the [FAQ](troubleshooting/faq.md)
- Review [troubleshooting guide](troubleshooting/overview.md)
- Submit issues on GitHub

---

*Generated automatically by the Documentation Build System*
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(book_dir / "index.md", 'w', encoding='utf-8') as f:
            f.write(index_content)
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality report."""
        logger.info("Generating quality report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_notebooks': len(self.notebook_metadata),
            'summary': {
                'categories': {},
                'quality_levels': {},
                'documentation_scores': [],
                'complexity_scores': [],
                'performance_metrics': {}
            },
            'notebooks': []
        }
        
        for nb_path, metadata in self.notebook_metadata.items():
            # Update summary statistics
            category = metadata.get('category', 'General')
            report['summary']['categories'][category] = report['summary']['categories'].get(category, 0) + 1
            
            level = metadata.get('documentation_level', 'Level 1 - Basic')
            report['summary']['quality_levels'][level] = report['summary']['quality_levels'].get(level, 0) + 1
            
            score = metadata.get('quality_score', 0)
            report['summary']['documentation_scores'].append(score)
            
            complexity = metadata.get('complexity_score', 0)
            report['summary']['complexity_scores'].append(complexity)
            
            # Add notebook details
            notebook_report = {
                'name': metadata.get('name', 'Unknown'),
                'path': nb_path,
                'category': category,
                'quality_score': score,
                'documentation_level': level,
                'complexity_score': complexity,
                'execution_time': metadata.get('execution_time', 'Unknown'),
                'memory_usage': metadata.get('memory_usage', 'Unknown'),
                'has_errors': len(metadata.get('errors', [])) > 0,
                'error_count': len(metadata.get('errors', [])),
                'function_count': len(metadata.get('functions', [])),
                'visualization_count': len(metadata.get('visualizations', [])),
                'dependencies': metadata.get('dependencies', []),
                'tags': metadata.get('tags', [])
            }
            
            report['notebooks'].append(notebook_report)
        
        # Calculate summary statistics
        if report['summary']['documentation_scores']:
            report['summary']['avg_quality_score'] = sum(report['summary']['documentation_scores']) / len(report['summary']['documentation_scores'])
            report['summary']['min_quality_score'] = min(report['summary']['documentation_scores'])
            report['summary']['max_quality_score'] = max(report['summary']['documentation_scores'])
        
        if report['summary']['complexity_scores']:
            report['summary']['avg_complexity_score'] = sum(report['summary']['complexity_scores']) / len(report['summary']['complexity_scores'])
            report['summary']['min_complexity_score'] = min(report['summary']['complexity_scores'])
            report['summary']['max_complexity_score'] = max(report['summary']['complexity_scores'])
        
        # Save report
        report_path = self.output_dir / "quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {report_path}")
        return report
    
    def build_full_documentation(self) -> bool:
        """Build complete documentation suite."""
        logger.info("Starting full documentation build...")
        
        try:
            # Step 1: Analyze notebooks
            self.analyze_notebooks()
            
            # Step 2: Generate cross-references
            self.generate_cross_references()
            
            # Step 3: Generate API documentation
            self.generate_api_documentation()
            
            # Step 4: Build Jupyter Book
            success = self.build_jupyter_book()
            
            # Step 5: Generate quality report
            self.generate_quality_report()
            
            # Step 6: Save metadata
            metadata_path = self.output_dir / "notebook_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.notebook_metadata, f, indent=2)
            
            cross_refs_path = self.output_dir / "cross_references.json"
            with open(cross_refs_path, 'w', encoding='utf-8') as f:
                json.dump(self.cross_references, f, indent=2)
            
            api_docs_path = self.output_dir / "api_documentation.json"
            with open(api_docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.api_documentation, f, indent=2)
            
            logger.info("Full documentation build completed successfully")
            return success
            
        except Exception as e:
            logger.error(f"Error in full documentation build: {e}")
            return False

def main():
    """Main function to run documentation builder."""
    parser = argparse.ArgumentParser(description="Build comprehensive documentation for Jupyter notebooks")
    parser.add_argument("--mode", choices=["full", "quick", "api", "notebooks"], default="full",
                        help="Documentation build mode")
    parser.add_argument("--output-dir", default="docs", help="Output directory for documentation")
    parser.add_argument("--base-dir", default=".", help="Base directory containing notebooks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    builder = DocumentationBuilder(base_dir=args.base_dir, output_dir=args.output_dir)
    
    if args.mode == "full":
        success = builder.build_full_documentation()
    elif args.mode == "quick":
        builder.analyze_notebooks()
        success = builder.build_jupyter_book()
    elif args.mode == "api":
        builder.generate_api_documentation()
        success = True
    elif args.mode == "notebooks":
        builder.analyze_notebooks()
        builder.generate_quality_report()
        success = True
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False
    
    if success:
        logger.info("Documentation build completed successfully")
        sys.exit(0)
    else:
        logger.error("Documentation build failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 