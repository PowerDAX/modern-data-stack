#!/usr/bin/env python3
"""
Notebook Documentation Enhancement System
========================================

This script automatically enhances existing Jupyter notebooks with comprehensive
documentation following the Modern Data Stack Showcase documentation standards.

Features:
- Automated header cell generation with metadata
- Documentation cell insertion between code cells
- Function docstring enhancement
- Visualization documentation
- Performance and complexity analysis
- Cross-reference generation
- Quality assessment and scoring
- Automated table of contents generation

Usage:
    python enhance-notebook-docs.py --notebook NOTEBOOK_PATH [--output OUTPUT_PATH]
    python enhance-notebook-docs.py --directory DIRECTORY_PATH [--recursive]
    python enhance-notebook-docs.py --template TEMPLATE_TYPE --output OUTPUT_PATH

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
from nbformat.v4 import new_markdown_cell, new_code_cell
import pandas as pd
import yaml
from jinja2 import Template
import shutil
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('notebook-enhancement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NotebookMetadata:
    """Data class for notebook metadata."""
    title: str
    author: str
    category: str
    created_date: str
    last_updated: str
    description: str
    prerequisites: List[str]
    datasets: List[Dict[str, str]]
    tools_libraries: List[Dict[str, str]]
    key_outcomes: List[str]
    related_notebooks: List[Dict[str, str]]
    estimated_runtime: str
    complexity_level: str
    tags: List[str]

class NotebookEnhancer:
    """
    Comprehensive notebook documentation enhancer.
    
    This class analyzes existing notebooks and automatically adds comprehensive
    documentation following the established standards.
    """
    
    def __init__(self, template_dir: str = "templates", output_dir: str = "enhanced"):
        """
        Initialize the notebook enhancer.
        
        Parameters
        ----------
        template_dir : str
            Directory containing documentation templates
        output_dir : str
            Directory for enhanced notebooks
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load documentation templates
        self.templates = self._load_templates()
        
        # Initialize analysis results
        self.analysis_results = {}
        
        logger.info("Notebook enhancer initialized")
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load Jinja2 templates for documentation generation."""
        templates = {}
        
        # Header template
        header_template = """# {{ metadata.title }}

**📊 Category**: {{ metadata.category }}

**👤 Author**: {{ metadata.author }}

**📅 Created**: {{ metadata.created_date }}

**🔄 Last Updated**: {{ metadata.last_updated }}

**⏱️ Estimated Runtime**: {{ metadata.estimated_runtime }}

**🎯 Purpose**: {{ metadata.description }}

**📋 Prerequisites**: 
{% for prereq in metadata.prerequisites %}
- {{ prereq }}
{% endfor %}

**📊 Datasets Used**:
{% for dataset in metadata.datasets %}
- **{{ dataset.name }}**: {{ dataset.description }}
{% endfor %}

**🔧 Tools & Libraries**:
{% for tool in metadata.tools_libraries %}
- **{{ tool.name }}**: {{ tool.purpose }}
{% endfor %}

**📈 Key Outcomes**:
{% for outcome in metadata.key_outcomes %}
- {{ outcome }}
{% endfor %}

**🔗 Related Notebooks**:
{% for related in metadata.related_notebooks %}
- **{{ related.name }}**: {{ related.relationship }}
{% endfor %}

**🏷️ Tags**: {{ metadata.tags | join(', ') }}

**📊 Complexity Level**: {{ metadata.complexity_level }}

---

## 📚 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Loading & Validation](#data-loading--validation)
3. [Analysis & Processing](#analysis--processing)
4. [Results & Visualization](#results--visualization)
5. [Insights & Conclusions](#insights--conclusions)
6. [Next Steps](#next-steps)
7. [References](#references)

---

## ⚠️ Important Notes

- **Performance**: This notebook may require significant memory and processing time
- **Data Privacy**: Ensure all datasets comply with privacy regulations
- **Reproducibility**: Set random seeds for consistent results
- **Dependencies**: Install all required libraries before execution

---

## 📋 Change Log

### v1.0.0 - {{ metadata.last_updated }}
- Initial documentation enhancement
- Automated header generation
- Comprehensive documentation added

---
"""
        
        # Code cell documentation template
        code_doc_template = """## 🔧 {{ section_title }}

### Objective
{{ objective }}

### Implementation Details
{{ implementation_details }}

### Expected Output
{{ expected_output }}

### Performance Considerations
{{ performance_notes }}

---
"""
        
        # Visualization documentation template
        viz_doc_template = """### 📈 {{ chart_title }}

**Purpose**: {{ purpose }}

**Data Source**: {{ data_source }}

**Visualization Type**: {{ viz_type }}

**Key Insights**:
{% for insight in insights %}
- {{ insight }}
{% endfor %}

**Interpretation Guide**:
{{ interpretation_guide }}

**Technical Notes**:
{% for note in technical_notes %}
- {{ note }}
{% endfor %}

---
"""
        
        # Function documentation template
        func_doc_template = """### 🔨 Function: `{{ function_name }}`

**Purpose**: {{ purpose }}

**Parameters**:
{% for param in parameters %}
- `{{ param.name }}` ({{ param.type }}): {{ param.description }}
{% endfor %}

**Returns**: {{ returns }}

**Example Usage**:
```python
{{ example_usage }}
```

**Implementation Notes**:
{% for note in notes %}
- {{ note }}
{% endfor %}

---
"""
        
        # Results section template
        results_template = """## 📊 Results & Analysis

### Summary Statistics
{{ summary_stats }}

### Key Findings
{% for finding in key_findings %}
- {{ finding }}
{% endfor %}

### Statistical Significance
{{ statistical_significance }}

### Limitations
{% for limitation in limitations %}
- {{ limitation }}
{% endfor %}

### Recommendations
{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}

---
"""
        
        # Conclusion template
        conclusion_template = """## 🎯 Conclusions & Next Steps

### Main Conclusions
{% for conclusion in conclusions %}
- {{ conclusion }}
{% endfor %}

### Business Impact
{{ business_impact }}

### Technical Learnings
{% for learning in technical_learnings %}
- {{ learning }}
{% endfor %}

### Recommended Actions
{% for action in recommended_actions %}
- {{ action }}
{% endfor %}

### Future Work
{% for future_work in future_work_items %}
- {{ future_work }}
{% endfor %}

---

## 📚 References

### Data Sources
{% for source in data_sources %}
- {{ source }}
{% endfor %}

### Methodological References
{% for ref in methodological_references %}
- {{ ref }}
{% endfor %}

### Additional Resources
{% for resource in additional_resources %}
- {{ resource }}
{% endfor %}

---

*This notebook was enhanced with automated documentation on {{ enhancement_date }}*
"""
        
        templates['header'] = Template(header_template)
        templates['code_doc'] = Template(code_doc_template)
        templates['viz_doc'] = Template(viz_doc_template)
        templates['func_doc'] = Template(func_doc_template)
        templates['results'] = Template(results_template)
        templates['conclusion'] = Template(conclusion_template)
        
        return templates
    
    def enhance_notebook(self, notebook_path: str, output_path: Optional[str] = None) -> str:
        """
        Enhance a single notebook with comprehensive documentation.
        
        Parameters
        ----------
        notebook_path : str
            Path to the input notebook
        output_path : str, optional
            Path for the enhanced notebook (default: auto-generated)
            
        Returns
        -------
        str
            Path to the enhanced notebook
        """
        notebook_path = Path(notebook_path)
        
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        
        logger.info(f"Enhancing notebook: {notebook_path}")
        
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Analyze the notebook
        analysis = self._analyze_notebook(nb, notebook_path)
        
        # Generate metadata
        metadata = self._generate_metadata(analysis, notebook_path)
        
        # Create enhanced notebook
        enhanced_nb = self._create_enhanced_notebook(nb, metadata, analysis)
        
        # Determine output path
        if output_path is None:
            output_path = self.output_dir / f"{notebook_path.stem}_enhanced.ipynb"
        else:
            output_path = Path(output_path)
        
        # Save enhanced notebook
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(enhanced_nb, f)
        
        logger.info(f"Enhanced notebook saved to: {output_path}")
        
        # Generate quality report
        self._generate_quality_report(analysis, output_path)
        
        return str(output_path)
    
    def _analyze_notebook(self, nb: nbformat.NotebookNode, notebook_path: Path) -> Dict[str, Any]:
        """Analyze notebook content and structure."""
        analysis = {
            'path': str(notebook_path),
            'name': notebook_path.stem,
            'total_cells': len(nb.cells),
            'code_cells': [],
            'markdown_cells': [],
            'functions': [],
            'classes': [],
            'imports': [],
            'visualizations': [],
            'data_operations': [],
            'complexity_metrics': {},
            'performance_indicators': {},
            'quality_assessment': {},
            'documentation_gaps': []
        }
        
        # Analyze each cell
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                code_analysis = self._analyze_code_cell(cell, i)
                analysis['code_cells'].append(code_analysis)
                
                # Extract functions, classes, imports
                analysis['functions'].extend(code_analysis.get('functions', []))
                analysis['classes'].extend(code_analysis.get('classes', []))
                analysis['imports'].extend(code_analysis.get('imports', []))
                analysis['visualizations'].extend(code_analysis.get('visualizations', []))
                analysis['data_operations'].extend(code_analysis.get('data_operations', []))
                
            elif cell.cell_type == 'markdown':
                markdown_analysis = self._analyze_markdown_cell(cell, i)
                analysis['markdown_cells'].append(markdown_analysis)
        
        # Calculate complexity metrics
        analysis['complexity_metrics'] = self._calculate_complexity_metrics(analysis)
        
        # Assess performance indicators
        analysis['performance_indicators'] = self._assess_performance_indicators(analysis)
        
        # Quality assessment
        analysis['quality_assessment'] = self._assess_quality(analysis)
        
        # Identify documentation gaps
        analysis['documentation_gaps'] = self._identify_documentation_gaps(analysis)
        
        return analysis
    
    def _analyze_code_cell(self, cell: nbformat.NotebookNode, index: int) -> Dict[str, Any]:
        """Analyze a code cell."""
        analysis = {
            'index': index,
            'source': cell.source,
            'line_count': len(cell.source.split('\n')),
            'functions': [],
            'classes': [],
            'imports': [],
            'visualizations': [],
            'data_operations': [],
            'complexity_score': 0,
            'has_docstring': False,
            'has_comments': False,
            'performance_notes': []
        }
        
        try:
            # Parse the code
            tree = ast.parse(cell.source)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'line_number': node.lineno,
                        'complexity': self._calculate_function_complexity(node)
                    }
                    analysis['functions'].append(func_info)
                    analysis['has_docstring'] = func_info['docstring'] is not None
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': [],
                        'line_number': node.lineno
                    }
                    analysis['classes'].append(class_info)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
            
            # Check for comments
            analysis['has_comments'] = '#' in cell.source
            
            # Identify visualizations
            viz_keywords = ['plt.', 'fig.', 'ax.', 'sns.', 'px.', 'go.', 'plotly']
            for keyword in viz_keywords:
                if keyword in cell.source:
                    analysis['visualizations'].append({
                        'type': self._identify_viz_type(cell.source),
                        'library': self._identify_viz_library(cell.source),
                        'has_title': 'title' in cell.source.lower(),
                        'has_labels': any(label in cell.source.lower() for label in ['xlabel', 'ylabel', 'labels'])
                    })
                    break
            
            # Identify data operations
            data_keywords = ['read_csv', 'read_sql', 'read_parquet', 'merge', 'groupby', 'pivot']
            for keyword in data_keywords:
                if keyword in cell.source:
                    analysis['data_operations'].append({
                        'operation': keyword,
                        'complexity': 'high' if keyword in ['merge', 'groupby'] else 'medium'
                    })
            
            # Calculate complexity score
            analysis['complexity_score'] = self._calculate_cell_complexity(cell.source)
            
            # Performance notes
            if any(keyword in cell.source for keyword in ['for ', 'while ', 'iterrows']):
                analysis['performance_notes'].append("Contains loops - consider vectorization")
            
            if 'read_csv' in cell.source and 'chunksize' not in cell.source:
                analysis['performance_notes'].append("Large file reading - consider chunking")
                
        except SyntaxError:
            analysis['syntax_error'] = True
            logger.warning(f"Syntax error in cell {index}")
        
        return analysis
    
    def _analyze_markdown_cell(self, cell: nbformat.NotebookNode, index: int) -> Dict[str, Any]:
        """Analyze a markdown cell."""
        analysis = {
            'index': index,
            'source': cell.source,
            'word_count': len(cell.source.split()),
            'has_headers': bool(re.search(r'^#+\s', cell.source, re.MULTILINE)),
            'has_lists': bool(re.search(r'^[-*+]\s', cell.source, re.MULTILINE)),
            'has_links': bool(re.search(r'\[.*?\]\(.*?\)', cell.source)),
            'has_images': bool(re.search(r'!\[.*?\]\(.*?\)', cell.source)),
            'has_code_blocks': bool(re.search(r'```', cell.source)),
            'documentation_quality': 'low'
        }
        
        # Assess documentation quality
        if analysis['word_count'] > 50 and analysis['has_headers']:
            analysis['documentation_quality'] = 'high'
        elif analysis['word_count'] > 20:
            analysis['documentation_quality'] = 'medium'
        
        return analysis
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate function complexity using cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
        
        return complexity
    
    def _calculate_cell_complexity(self, source: str) -> int:
        """Calculate cell complexity score."""
        complexity = 0
        
        # Count control structures
        complexity += source.count('if ')
        complexity += source.count('for ')
        complexity += source.count('while ')
        complexity += source.count('try:')
        complexity += source.count('except')
        
        # Count function definitions
        complexity += source.count('def ')
        complexity += source.count('class ')
        
        # Count nested structures
        complexity += source.count('    if ')  # Nested if
        complexity += source.count('    for ')  # Nested for
        
        return complexity
    
    def _identify_viz_type(self, source: str) -> str:
        """Identify visualization type."""
        viz_types = {
            'scatter': ['scatter', 'scatterplot'],
            'line': ['plot', 'line'],
            'bar': ['bar', 'barplot'],
            'histogram': ['hist', 'histogram'],
            'box': ['box', 'boxplot'],
            'heatmap': ['heatmap'],
            'pie': ['pie']
        }
        
        source_lower = source.lower()
        for viz_type, keywords in viz_types.items():
            if any(keyword in source_lower for keyword in keywords):
                return viz_type
        return 'unknown'
    
    def _identify_viz_library(self, source: str) -> str:
        """Identify visualization library."""
        if 'px.' in source or 'plotly' in source:
            return 'plotly'
        elif 'sns.' in source:
            return 'seaborn'
        elif 'plt.' in source:
            return 'matplotlib'
        return 'unknown'
    
    def _calculate_complexity_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall complexity metrics."""
        metrics = {
            'total_functions': len(analysis['functions']),
            'total_classes': len(analysis['classes']),
            'total_imports': len(set(analysis['imports'])),
            'total_visualizations': len(analysis['visualizations']),
            'total_data_operations': len(analysis['data_operations']),
            'avg_cell_complexity': 0,
            'max_cell_complexity': 0,
            'documentation_ratio': 0
        }
        
        if analysis['code_cells']:
            complexities = [cell.get('complexity_score', 0) for cell in analysis['code_cells']]
            metrics['avg_cell_complexity'] = sum(complexities) / len(complexities)
            metrics['max_cell_complexity'] = max(complexities)
        
        if analysis['total_cells'] > 0:
            metrics['documentation_ratio'] = len(analysis['markdown_cells']) / analysis['total_cells']
        
        return metrics
    
    def _assess_performance_indicators(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess performance indicators."""
        indicators = {
            'has_loops': False,
            'has_large_data_ops': False,
            'memory_intensive': False,
            'io_operations': 0,
            'performance_risks': []
        }
        
        for cell in analysis['code_cells']:
            if any('loop' in note for note in cell.get('performance_notes', [])):
                indicators['has_loops'] = True
            
            if cell.get('data_operations'):
                indicators['has_large_data_ops'] = True
        
        # Check for memory intensive operations
        memory_keywords = ['numpy', 'pandas', 'sklearn', 'tensorflow', 'torch']
        for imp in analysis['imports']:
            if any(keyword in imp for keyword in memory_keywords):
                indicators['memory_intensive'] = True
                break
        
        return indicators
    
    def _assess_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall notebook quality."""
        quality = {
            'documentation_score': 0,
            'code_quality_score': 0,
            'overall_score': 0,
            'recommendations': []
        }
        
        # Documentation score
        markdown_ratio = analysis['complexity_metrics']['documentation_ratio']
        documented_functions = sum(1 for func in analysis['functions'] if func.get('docstring'))
        total_functions = len(analysis['functions'])
        
        doc_score = (markdown_ratio * 50) + (documented_functions / max(total_functions, 1) * 50)
        quality['documentation_score'] = min(doc_score, 100)
        
        # Code quality score
        has_comments = sum(1 for cell in analysis['code_cells'] if cell.get('has_comments'))
        total_code_cells = len(analysis['code_cells'])
        
        code_score = (has_comments / max(total_code_cells, 1) * 100)
        quality['code_quality_score'] = min(code_score, 100)
        
        # Overall score
        quality['overall_score'] = (quality['documentation_score'] + quality['code_quality_score']) / 2
        
        # Recommendations
        if quality['documentation_score'] < 70:
            quality['recommendations'].append("Add more documentation cells")
        if quality['code_quality_score'] < 70:
            quality['recommendations'].append("Add more code comments")
        if total_functions > 0 and documented_functions / total_functions < 0.8:
            quality['recommendations'].append("Add docstrings to functions")
        
        return quality
    
    def _identify_documentation_gaps(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify documentation gaps."""
        gaps = []
        
        # Check for missing header
        if not analysis['markdown_cells'] or analysis['markdown_cells'][0]['index'] != 0:
            gaps.append("Missing comprehensive header cell")
        
        # Check for undocumented functions
        undocumented_functions = [func for func in analysis['functions'] if not func.get('docstring')]
        if undocumented_functions:
            gaps.append(f"{len(undocumented_functions)} functions without docstrings")
        
        # Check for undocumented visualizations
        undocumented_viz = [viz for viz in analysis['visualizations'] if not viz.get('has_title')]
        if undocumented_viz:
            gaps.append(f"{len(undocumented_viz)} visualizations without titles")
        
        # Check for missing section headers
        if analysis['complexity_metrics']['total_functions'] > 0 and not any(
            'function' in cell.get('source', '').lower() for cell in analysis['markdown_cells']
        ):
            gaps.append("Missing function documentation sections")
        
        return gaps
    
    def _generate_metadata(self, analysis: Dict[str, Any], notebook_path: Path) -> NotebookMetadata:
        """Generate metadata for the notebook."""
        # Extract category from path
        category = self._extract_category_from_path(notebook_path)
        
        # Generate description based on analysis
        description = self._generate_description(analysis)
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(analysis)
        
        # Generate prerequisites
        prerequisites = self._generate_prerequisites(analysis)
        
        # Generate datasets info
        datasets = self._generate_datasets_info(analysis)
        
        # Generate tools and libraries
        tools_libraries = self._generate_tools_libraries(analysis)
        
        # Generate key outcomes
        key_outcomes = self._generate_key_outcomes(analysis)
        
        # Generate tags
        tags = self._generate_tags(analysis)
        
        metadata = NotebookMetadata(
            title=analysis['name'].replace('_', ' ').title(),
            author="Data Science Team",
            category=category,
            created_date=datetime.now().strftime('%Y-%m-%d'),
            last_updated=datetime.now().strftime('%Y-%m-%d'),
            description=description,
            prerequisites=prerequisites,
            datasets=datasets,
            tools_libraries=tools_libraries,
            key_outcomes=key_outcomes,
            related_notebooks=[],
            estimated_runtime=self._estimate_runtime(analysis),
            complexity_level=complexity_level,
            tags=tags
        )
        
        return metadata
    
    def _extract_category_from_path(self, notebook_path: Path) -> str:
        """Extract category from notebook path."""
        path_parts = str(notebook_path).lower()
        if 'data-exploration' in path_parts:
            return 'Data Exploration'
        elif 'ml-workflows' in path_parts:
            return 'ML Workflow'
        elif 'devops-automation' in path_parts:
            return 'DevOps Automation'
        elif 'business-intelligence' in path_parts:
            return 'Business Intelligence'
        elif 'template' in path_parts:
            return 'Template'
        else:
            return 'General'
    
    def _generate_description(self, analysis: Dict[str, Any]) -> str:
        """Generate description based on analysis."""
        if analysis['functions']:
            return f"Comprehensive analysis notebook with {len(analysis['functions'])} custom functions"
        elif analysis['visualizations']:
            return f"Data visualization notebook with {len(analysis['visualizations'])} charts"
        elif analysis['data_operations']:
            return f"Data processing notebook with {len(analysis['data_operations'])} operations"
        else:
            return "Analysis and exploration notebook"
    
    def _determine_complexity_level(self, analysis: Dict[str, Any]) -> str:
        """Determine complexity level."""
        complexity_score = analysis['complexity_metrics']['avg_cell_complexity']
        
        if complexity_score > 10:
            return "🔴 High - Advanced techniques and complex logic"
        elif complexity_score > 5:
            return "🟡 Medium - Moderate complexity with some advanced concepts"
        else:
            return "🟢 Low - Beginner-friendly with basic concepts"
    
    def _generate_prerequisites(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate prerequisites based on analysis."""
        prerequisites = ["Basic Python programming knowledge", "Understanding of data structures"]
        
        # Add based on imports
        imports = analysis['imports']
        if any('pandas' in imp for imp in imports):
            prerequisites.append("Pandas for data manipulation")
        if any('numpy' in imp for imp in imports):
            prerequisites.append("NumPy for numerical computing")
        if any('sklearn' in imp for imp in imports):
            prerequisites.append("Machine learning concepts")
        if any('tensorflow' in imp or 'torch' in imp for imp in imports):
            prerequisites.append("Deep learning fundamentals")
        if any('matplotlib' in imp or 'seaborn' in imp or 'plotly' in imp for imp in imports):
            prerequisites.append("Data visualization principles")
        
        return prerequisites
    
    def _generate_datasets_info(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate datasets information."""
        datasets = []
        
        # Look for data loading operations
        for cell in analysis['code_cells']:
            if 'read_csv' in cell['source']:
                datasets.append({
                    'name': 'CSV Dataset',
                    'description': 'Primary dataset loaded from CSV file'
                })
            elif 'read_sql' in cell['source']:
                datasets.append({
                    'name': 'Database Query',
                    'description': 'Data retrieved from database query'
                })
            elif 'read_parquet' in cell['source']:
                datasets.append({
                    'name': 'Parquet Dataset',
                    'description': 'Structured data from Parquet file'
                })
        
        if not datasets:
            datasets.append({
                'name': 'Sample Dataset',
                'description': 'Dataset used for analysis and exploration'
            })
        
        return datasets
    
    def _generate_tools_libraries(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate tools and libraries information."""
        tools = []
        
        # Map imports to descriptions
        import_descriptions = {
            'pandas': 'Data manipulation and analysis',
            'numpy': 'Numerical computing',
            'matplotlib': 'Static data visualization',
            'seaborn': 'Statistical data visualization',
            'plotly': 'Interactive data visualization',
            'sklearn': 'Machine learning algorithms',
            'tensorflow': 'Deep learning framework',
            'torch': 'Deep learning framework',
            'scipy': 'Scientific computing',
            'statsmodels': 'Statistical modeling'
        }
        
        for imp in analysis['imports']:
            for lib, desc in import_descriptions.items():
                if lib in imp:
                    tools.append({'name': lib.title(), 'purpose': desc})
                    break
        
        if not tools:
            tools.append({'name': 'Python', 'purpose': 'Core programming language'})
        
        return tools
    
    def _generate_key_outcomes(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate key outcomes based on analysis."""
        outcomes = []
        
        if analysis['visualizations']:
            outcomes.append(f"Generate {len(analysis['visualizations'])} insightful visualizations")
        if analysis['functions']:
            outcomes.append(f"Implement {len(analysis['functions'])} reusable functions")
        if analysis['data_operations']:
            outcomes.append("Perform comprehensive data analysis")
        
        outcomes.append("Gain insights from data exploration")
        outcomes.append("Document findings and recommendations")
        
        return outcomes
    
    def _generate_tags(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate tags based on analysis."""
        tags = []
        
        # Add based on imports
        imports = analysis['imports']
        if any('pandas' in imp for imp in imports):
            tags.append('data-manipulation')
        if any('matplotlib' in imp or 'seaborn' in imp or 'plotly' in imp for imp in imports):
            tags.append('visualization')
        if any('sklearn' in imp for imp in imports):
            tags.append('machine-learning')
        if any('tensorflow' in imp or 'torch' in imp for imp in imports):
            tags.append('deep-learning')
        if any('scipy' in imp or 'statsmodels' in imp for imp in imports):
            tags.append('statistics')
        
        # Add based on content
        if analysis['data_operations']:
            tags.append('data-processing')
        if analysis['functions']:
            tags.append('custom-functions')
        
        return tags
    
    def _estimate_runtime(self, analysis: Dict[str, Any]) -> str:
        """Estimate runtime based on analysis."""
        performance = analysis['performance_indicators']
        
        if performance['memory_intensive'] and performance['has_large_data_ops']:
            return "⏱️ Long (>10 minutes)"
        elif performance['has_loops'] or performance['memory_intensive']:
            return "⏱️ Medium (2-10 minutes)"
        else:
            return "⏱️ Short (<2 minutes)"
    
    def _create_enhanced_notebook(self, original_nb: nbformat.NotebookNode, 
                                 metadata: NotebookMetadata, 
                                 analysis: Dict[str, Any]) -> nbformat.NotebookNode:
        """Create enhanced notebook with comprehensive documentation."""
        # Create new notebook
        enhanced_nb = nbformat.v4.new_notebook()
        
        # Add header cell
        header_content = self.templates['header'].render(metadata=metadata)
        enhanced_nb.cells.append(new_markdown_cell(header_content))
        
        # Process original cells
        for i, cell in enumerate(original_nb.cells):
            if cell.cell_type == 'code':
                # Add documentation before code cell
                code_analysis = next((c for c in analysis['code_cells'] if c['index'] == i), {})
                
                if code_analysis.get('functions'):
                    # Add function documentation
                    for func in code_analysis['functions']:
                        func_doc = self._generate_function_documentation(func)
                        enhanced_nb.cells.append(new_markdown_cell(func_doc))
                
                # Add code cell documentation
                code_doc = self._generate_code_cell_documentation(cell, code_analysis)
                enhanced_nb.cells.append(new_markdown_cell(code_doc))
                
                # Add the original code cell
                enhanced_nb.cells.append(cell)
                
                # Add visualization documentation if present
                if code_analysis.get('visualizations'):
                    for viz in code_analysis['visualizations']:
                        viz_doc = self._generate_visualization_documentation(viz)
                        enhanced_nb.cells.append(new_markdown_cell(viz_doc))
                
            elif cell.cell_type == 'markdown':
                # Keep original markdown cells
                enhanced_nb.cells.append(cell)
        
        # Add results section
        results_content = self._generate_results_section(analysis)
        enhanced_nb.cells.append(new_markdown_cell(results_content))
        
        # Add conclusion section
        conclusion_content = self._generate_conclusion_section(analysis, metadata)
        enhanced_nb.cells.append(new_markdown_cell(conclusion_content))
        
        return enhanced_nb
    
    def _generate_function_documentation(self, func: Dict[str, Any]) -> str:
        """Generate documentation for a function."""
        doc_content = f"""### 🔨 Function: `{func['name']}`

**Purpose**: {func.get('docstring', 'Function implementation')}

**Parameters**:
"""
        
        for arg in func.get('args', []):
            doc_content += f"- `{arg}`: Parameter description\n"
        
        doc_content += f"""
**Complexity**: {func.get('complexity', 1)} (Cyclomatic complexity)

**Usage Example**:
```python
# Example usage of {func['name']}
result = {func['name']}(parameters)
```

---
"""
        
        return doc_content
    
    def _generate_code_cell_documentation(self, cell: nbformat.NotebookNode, 
                                        analysis: Dict[str, Any]) -> str:
        """Generate documentation for a code cell."""
        doc_content = f"""### 💻 Code Implementation

**Cell Purpose**: {self._infer_cell_purpose(cell.source)}

**Complexity Score**: {analysis.get('complexity_score', 0)}

**Performance Notes**:
"""
        
        for note in analysis.get('performance_notes', ['Standard execution expected']):
            doc_content += f"- {note}\n"
        
        if analysis.get('data_operations'):
            doc_content += "\n**Data Operations**:\n"
            for op in analysis['data_operations']:
                doc_content += f"- {op['operation']} ({op['complexity']} complexity)\n"
        
        doc_content += "\n---\n"
        
        return doc_content
    
    def _infer_cell_purpose(self, source: str) -> str:
        """Infer the purpose of a code cell."""
        source_lower = source.lower()
        
        if 'import' in source_lower:
            return "Library imports and environment setup"
        elif 'read_' in source_lower:
            return "Data loading and initial inspection"
        elif any(viz in source_lower for viz in ['plt.', 'fig.', 'ax.', 'sns.', 'px.']):
            return "Data visualization and plotting"
        elif 'def ' in source_lower:
            return "Function definition and implementation"
        elif any(ml in source_lower for ml in ['fit', 'predict', 'transform', 'score']):
            return "Machine learning model operations"
        elif any(data in source_lower for data in ['groupby', 'merge', 'pivot']):
            return "Data manipulation and transformation"
        else:
            return "Data processing and analysis"
    
    def _generate_visualization_documentation(self, viz: Dict[str, Any]) -> str:
        """Generate documentation for a visualization."""
        doc_content = f"""### 📊 Visualization: {viz['type'].title()} Chart

**Library**: {viz['library'].title()}

**Chart Type**: {viz['type']}

**Documentation Status**:
- Title: {'✅ Present' if viz.get('has_title') else '❌ Missing'}
- Labels: {'✅ Present' if viz.get('has_labels') else '❌ Missing'}

**Key Insights**:
- Visual representation of data patterns
- Supports analysis conclusions
- Enhances understanding of relationships

**Interpretation Guide**:
- Review axis labels and title for context
- Look for patterns, trends, and outliers
- Consider data distribution and relationships

---
"""
        
        return doc_content
    
    def _generate_results_section(self, analysis: Dict[str, Any]) -> str:
        """Generate results section."""
        results_content = f"""## 📊 Analysis Results & Summary

### 📈 Computational Overview

**Total Cells**: {analysis['total_cells']}
- Code Cells: {len(analysis['code_cells'])}
- Documentation Cells: {len(analysis['markdown_cells'])}

**Functions Implemented**: {len(analysis['functions'])}

**Visualizations Created**: {len(analysis['visualizations'])}

**Data Operations**: {len(analysis['data_operations'])}

### 🎯 Key Findings

Based on the analysis performed in this notebook:

1. **Data Processing**: Successfully implemented data loading and manipulation operations
2. **Visualization**: Created insightful visualizations to understand data patterns
3. **Analysis**: Performed comprehensive analysis with appropriate methods
4. **Documentation**: Enhanced documentation for reproducibility and understanding

### 📊 Quality Metrics

**Documentation Score**: {analysis.get('quality_assessment', {}).get('documentation_score', 0):.1f}/100

**Code Quality Score**: {analysis.get('quality_assessment', {}).get('code_quality_score', 0):.1f}/100

**Overall Score**: {analysis.get('quality_assessment', {}).get('overall_score', 0):.1f}/100

### 🔍 Recommendations

"""
        
        for rec in analysis.get('quality_assessment', {}).get('recommendations', []):
            results_content += f"- {rec}\n"
        
        results_content += "\n---\n"
        
        return results_content
    
    def _generate_conclusion_section(self, analysis: Dict[str, Any], 
                                   metadata: NotebookMetadata) -> str:
        """Generate conclusion section."""
        conclusion_content = f"""## 🎯 Conclusions & Next Steps

### 🔑 Main Conclusions

1. **Analysis Completed**: Successfully executed comprehensive analysis with {len(analysis['code_cells'])} code implementations
2. **Insights Generated**: Created {len(analysis['visualizations'])} visualizations for data understanding
3. **Documentation Enhanced**: Improved notebook documentation following best practices
4. **Quality Achieved**: Reached documentation standards with comprehensive explanations

### 🚀 Business Impact

This analysis provides:
- Clear insights into data patterns and relationships
- Reproducible methodology for future analyses
- Well-documented code for team collaboration
- Actionable recommendations based on findings

### 🎓 Technical Learnings

Key technical implementations:
"""
        
        for func in analysis['functions']:
            conclusion_content += f"- Implemented `{func['name']}` function for enhanced functionality\n"
        
        conclusion_content += f"""
### 📋 Recommended Next Steps

1. **Validation**: Validate findings with additional data sources
2. **Automation**: Consider automating key analysis steps
3. **Monitoring**: Set up monitoring for key metrics identified
4. **Scaling**: Evaluate scalability for larger datasets
5. **Integration**: Integrate insights into business processes

### 🔮 Future Work

- Expand analysis to include additional variables
- Implement real-time analysis capabilities
- Create automated reporting dashboards
- Develop predictive models based on insights
- Establish data quality monitoring systems

---

## 📚 References & Resources

### 📊 Data Sources
- Primary analysis dataset
- Supporting reference data
- External validation sources

### 🔬 Methodological References
- Statistical analysis techniques
- Visualization best practices
- Data science methodologies

### 🛠️ Technical Resources
- Python documentation
- Library-specific guides
- Best practice frameworks

### 🌐 Additional Resources
- Domain-specific knowledge bases
- Industry benchmarks
- Regulatory guidelines

---

## 📋 Appendices

### A. Code Quality Assessment
- Total lines of code: {sum(cell.get('line_count', 0) for cell in analysis['code_cells'])}
- Average complexity per cell: {analysis['complexity_metrics']['avg_cell_complexity']:.2f}
- Documentation coverage: {analysis['complexity_metrics']['documentation_ratio']:.1%}

### B. Performance Characteristics
- Memory intensive operations: {'Yes' if analysis['performance_indicators']['memory_intensive'] else 'No'}
- Large data operations: {'Yes' if analysis['performance_indicators']['has_large_data_ops'] else 'No'}
- Optimization opportunities: {len(analysis['performance_indicators']['performance_risks'])}

### C. Documentation Enhancement Details
- Enhancement date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Documentation standard: Modern Data Stack Showcase v1.0
- Quality score: {analysis.get('quality_assessment', {}).get('overall_score', 0):.1f}/100

---

*This notebook has been enhanced with comprehensive documentation using the Modern Data Stack Showcase documentation standards. All enhancements maintain code functionality while significantly improving readability, maintainability, and educational value.*

---

### 🆘 Support & Contact

For questions about this analysis or the documentation enhancement:
- Review the [documentation standards guide](../documentation/standards/documentation-guidelines.md)
- Check the [troubleshooting guide](../documentation/troubleshooting/overview.md)
- Contact the Data Science Team
- Submit issues via the project repository

---

*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return conclusion_content
    
    def _generate_quality_report(self, analysis: Dict[str, Any], output_path: Path) -> None:
        """Generate quality report for the enhanced notebook."""
        report = {
            'notebook_name': analysis['name'],
            'enhancement_date': datetime.now().isoformat(),
            'original_metrics': {
                'total_cells': analysis['total_cells'],
                'code_cells': len(analysis['code_cells']),
                'markdown_cells': len(analysis['markdown_cells']),
                'functions': len(analysis['functions']),
                'visualizations': len(analysis['visualizations'])
            },
            'quality_assessment': analysis.get('quality_assessment', {}),
            'complexity_metrics': analysis.get('complexity_metrics', {}),
            'performance_indicators': analysis.get('performance_indicators', {}),
            'documentation_gaps': analysis.get('documentation_gaps', []),
            'enhancement_summary': {
                'documentation_added': True,
                'header_enhanced': True,
                'functions_documented': len(analysis['functions']),
                'visualizations_documented': len(analysis['visualizations']),
                'quality_improvements': len(analysis.get('quality_assessment', {}).get('recommendations', []))
            }
        }
        
        # Save report
        report_path = output_path.parent / f"{output_path.stem}_quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to: {report_path}")
    
    def enhance_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """
        Enhance all notebooks in a directory.
        
        Parameters
        ----------
        directory_path : str
            Path to the directory containing notebooks
        recursive : bool
            Whether to search recursively in subdirectories
            
        Returns
        -------
        List[str]
            List of paths to enhanced notebooks
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all notebooks
        if recursive:
            notebook_files = list(directory_path.rglob("*.ipynb"))
        else:
            notebook_files = list(directory_path.glob("*.ipynb"))
        
        logger.info(f"Found {len(notebook_files)} notebooks to enhance")
        
        enhanced_notebooks = []
        
        for notebook_path in notebook_files:
            try:
                # Skip already enhanced notebooks
                if '_enhanced' in notebook_path.name:
                    continue
                
                enhanced_path = self.enhance_notebook(str(notebook_path))
                enhanced_notebooks.append(enhanced_path)
                
            except Exception as e:
                logger.error(f"Error enhancing {notebook_path}: {e}")
        
        logger.info(f"Enhanced {len(enhanced_notebooks)} notebooks")
        return enhanced_notebooks

def main():
    """Main function to run notebook enhancement."""
    parser = argparse.ArgumentParser(description="Enhance Jupyter notebooks with comprehensive documentation")
    parser.add_argument("--notebook", help="Path to a single notebook to enhance")
    parser.add_argument("--directory", help="Path to directory containing notebooks")
    parser.add_argument("--output", help="Output path for enhanced notebook(s)")
    parser.add_argument("--recursive", action="store_true", help="Search recursively in subdirectories")
    parser.add_argument("--template-dir", default="templates", help="Directory containing templates")
    parser.add_argument("--output-dir", default="enhanced", help="Directory for enhanced notebooks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    enhancer = NotebookEnhancer(template_dir=args.template_dir, output_dir=args.output_dir)
    
    try:
        if args.notebook:
            # Enhance single notebook
            enhanced_path = enhancer.enhance_notebook(args.notebook, args.output)
            print(f"Enhanced notebook saved to: {enhanced_path}")
            
        elif args.directory:
            # Enhance directory of notebooks
            enhanced_notebooks = enhancer.enhance_directory(args.directory, args.recursive)
            print(f"Enhanced {len(enhanced_notebooks)} notebooks")
            for path in enhanced_notebooks:
                print(f"  - {path}")
                
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 