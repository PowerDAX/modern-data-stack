#!/usr/bin/env python3
"""
Data Lineage Tracking System
============================

This module provides comprehensive data lineage tracking capabilities for the
modern data stack, including:
- Data flow visualization
- Transformation tracking
- Impact analysis
- Column-level lineage
- Automated lineage discovery
- Lineage-based impact assessment

Features:
- Multi-level lineage tracking (table, column, field)
- Transformation logic capture
- Data flow visualization
- Impact analysis for changes
- Automated lineage discovery from SQL and code
- Integration with dbt, airflow, and other tools

Author: Data Engineering Team
Date: 2024-01-15
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import ast
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, Name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LineageType(Enum):
    """Types of lineage relationships"""
    DIRECT = "direct"
    DERIVED = "derived"
    AGGREGATED = "aggregated"
    JOINED = "joined"
    FILTERED = "filtered"
    TRANSFORMED = "transformed"
    CALCULATED = "calculated"

class EntityType(Enum):
    """Types of data entities"""
    TABLE = "table"
    VIEW = "view"
    COLUMN = "column"
    FIELD = "field"
    DATASET = "dataset"
    REPORT = "report"
    DASHBOARD = "dashboard"
    API = "api"
    FILE = "file"

class ChangeType(Enum):
    """Types of changes for impact analysis"""
    SCHEMA_CHANGE = "schema_change"
    DATA_CHANGE = "data_change"
    TRANSFORMATION_CHANGE = "transformation_change"
    BUSINESS_LOGIC_CHANGE = "business_logic_change"
    DEPRECATED = "deprecated"
    ADDED = "added"
    REMOVED = "removed"

@dataclass
class DataEntity:
    """Represents a data entity in the lineage graph"""
    entity_id: str
    name: str
    entity_type: EntityType
    schema_name: Optional[str] = None
    database_name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner: Optional[str] = None
    steward: Optional[str] = None

@dataclass
class LineageRelationship:
    """Represents a lineage relationship between entities"""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    lineage_type: LineageType
    transformation_logic: Optional[str] = None
    transformation_description: Optional[str] = None
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    discovered_by: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImpactAnalysis:
    """Results of impact analysis"""
    change_id: str
    source_entity_id: str
    change_type: ChangeType
    change_description: str
    impacted_entities: List[str]
    impact_severity: str  # low, medium, high, critical
    recommendations: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class DataLineageTracker:
    """
    Comprehensive data lineage tracking system.
    
    This class provides capabilities for tracking data lineage across the
    modern data stack, including automated discovery, visualization, and
    impact analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data lineage tracker.
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration for the lineage tracker
        """
        self.config = config or self._get_default_config()
        self.entities: Dict[str, DataEntity] = {}
        self.relationships: Dict[str, LineageRelationship] = {}
        self.lineage_graph = nx.DiGraph()
        self.impact_analyses: List[ImpactAnalysis] = []
        
        # Initialize SQL parser
        self.sql_parser = SQLLineageParser()
        
        logger.info("Data Lineage Tracker initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'auto_discovery': {
                'enabled': True,
                'scan_sql_files': True,
                'scan_python_files': True,
                'scan_dbt_models': True
            },
            'visualization': {
                'max_depth': 5,
                'show_column_level': True,
                'layout': 'hierarchical'
            },
            'impact_analysis': {
                'enabled': True,
                'severity_thresholds': {
                    'low': 0.2,
                    'medium': 0.5,
                    'high': 0.8,
                    'critical': 0.9
                }
            }
        }
    
    def register_entity(self, entity: DataEntity) -> str:
        """
        Register a data entity in the lineage graph.
        
        Parameters
        ----------
        entity : DataEntity
            Entity to register
            
        Returns
        -------
        str
            Entity ID
        """
        self.entities[entity.entity_id] = entity
        self.lineage_graph.add_node(entity.entity_id, **entity.__dict__)
        
        logger.info(f"Registered entity: {entity.name} ({entity.entity_type.value})")
        return entity.entity_id
    
    def add_lineage_relationship(self, relationship: LineageRelationship) -> str:
        """
        Add a lineage relationship between entities.
        
        Parameters
        ----------
        relationship : LineageRelationship
            Relationship to add
            
        Returns
        -------
        str
            Relationship ID
        """
        # Validate entities exist
        if relationship.source_entity_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_entity_id} not found")
        if relationship.target_entity_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_entity_id} not found")
        
        self.relationships[relationship.relationship_id] = relationship
        self.lineage_graph.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            **relationship.__dict__
        )
        
        logger.info(f"Added lineage relationship: {relationship.source_entity_id} -> {relationship.target_entity_id}")
        return relationship.relationship_id
    
    def discover_lineage_from_sql(self, sql_content: str, 
                                 source_name: str = "unknown") -> List[LineageRelationship]:
        """
        Discover lineage relationships from SQL content.
        
        Parameters
        ----------
        sql_content : str
            SQL content to parse
        source_name : str
            Name of the source file or query
            
        Returns
        -------
        List[LineageRelationship]
            Discovered relationships
        """
        relationships = []
        
        try:
            # Parse SQL to extract table references
            parsed_sql = self.sql_parser.parse_sql(sql_content)
            
            # Extract source and target tables
            source_tables = parsed_sql.get('source_tables', [])
            target_tables = parsed_sql.get('target_tables', [])
            
            # Create relationships
            for target_table in target_tables:
                for source_table in source_tables:
                    relationship = LineageRelationship(
                        relationship_id=f"sql_{source_name}_{source_table}_{target_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        source_entity_id=source_table,
                        target_entity_id=target_table,
                        lineage_type=LineageType.DERIVED,
                        transformation_logic=sql_content,
                        transformation_description=f"SQL transformation in {source_name}",
                        confidence_score=0.8,
                        discovered_by="sql_parser"
                    )
                    relationships.append(relationship)
            
            logger.info(f"Discovered {len(relationships)} relationships from SQL")
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {e}")
        
        return relationships
    
    def discover_lineage_from_dbt(self, dbt_project_dir: str) -> List[LineageRelationship]:
        """
        Discover lineage from dbt project.
        
        Parameters
        ----------
        dbt_project_dir : str
            Path to dbt project directory
            
        Returns
        -------
        List[LineageRelationship]
            Discovered relationships
        """
        relationships = []
        dbt_dir = Path(dbt_project_dir)
        
        if not dbt_dir.exists():
            logger.warning(f"dbt project directory not found: {dbt_project_dir}")
            return relationships
        
        # Find dbt model files
        model_files = list(dbt_dir.rglob("*.sql"))
        
        for model_file in model_files:
            try:
                with open(model_file, 'r') as f:
                    sql_content = f.read()
                
                # Parse dbt-specific constructs
                model_name = model_file.stem
                
                # Extract dbt references
                dbt_refs = re.findall(r"ref\(['\"]([^'\"]+)['\"]\)", sql_content)
                dbt_sources = re.findall(r"source\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", sql_content)
                
                # Create relationships for dbt refs
                for ref in dbt_refs:
                    relationship = LineageRelationship(
                        relationship_id=f"dbt_{ref}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        source_entity_id=ref,
                        target_entity_id=model_name,
                        lineage_type=LineageType.DERIVED,
                        transformation_logic=sql_content,
                        transformation_description=f"dbt model transformation",
                        confidence_score=0.9,
                        discovered_by="dbt_parser"
                    )
                    relationships.append(relationship)
                
                # Create relationships for dbt sources
                for source_schema, source_table in dbt_sources:
                    source_id = f"{source_schema}.{source_table}"
                    relationship = LineageRelationship(
                        relationship_id=f"dbt_{source_id}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        source_entity_id=source_id,
                        target_entity_id=model_name,
                        lineage_type=LineageType.DIRECT,
                        transformation_logic=sql_content,
                        transformation_description=f"dbt source reference",
                        confidence_score=0.95,
                        discovered_by="dbt_parser"
                    )
                    relationships.append(relationship)
                
            except Exception as e:
                logger.error(f"Error parsing dbt model {model_file}: {e}")
        
        logger.info(f"Discovered {len(relationships)} relationships from dbt project")
        return relationships
    
    def discover_lineage_from_python(self, python_file: str) -> List[LineageRelationship]:
        """
        Discover lineage from Python code.
        
        Parameters
        ----------
        python_file : str
            Path to Python file
            
        Returns
        -------
        List[LineageRelationship]
            Discovered relationships
        """
        relationships = []
        
        try:
            with open(python_file, 'r') as f:
                content = f.read()
            
            # Parse Python AST
            tree = ast.parse(content)
            
            # Find pandas operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Look for pandas read operations
                    if (isinstance(node.func, ast.Attribute) and 
                        node.func.attr in ['read_csv', 'read_sql', 'read_parquet', 'read_excel']):
                        
                        # Extract file/table names
                        if node.args and isinstance(node.args[0], ast.Str):
                            source_name = node.args[0].s
                            
                            relationship = LineageRelationship(
                                relationship_id=f"python_{source_name}_{python_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                source_entity_id=source_name,
                                target_entity_id=f"python_script_{Path(python_file).stem}",
                                lineage_type=LineageType.DIRECT,
                                transformation_description=f"Python data loading",
                                confidence_score=0.7,
                                discovered_by="python_parser"
                            )
                            relationships.append(relationship)
            
            logger.info(f"Discovered {len(relationships)} relationships from Python file")
            
        except Exception as e:
            logger.error(f"Error parsing Python file {python_file}: {e}")
        
        return relationships
    
    def get_lineage_upstream(self, entity_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """
        Get upstream lineage for an entity.
        
        Parameters
        ----------
        entity_id : str
            Entity ID to trace upstream
        max_depth : int
            Maximum depth to trace
            
        Returns
        -------
        Dict[str, Any]
            Upstream lineage information
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not found")
        
        upstream_entities = []
        visited = set()
        
        def _trace_upstream(current_entity: str, depth: int):
            if depth >= max_depth or current_entity in visited:
                return
            
            visited.add(current_entity)
            
            # Get predecessors in the lineage graph
            predecessors = list(self.lineage_graph.predecessors(current_entity))
            
            for pred in predecessors:
                if pred not in visited:
                    entity = self.entities[pred]
                    relationship = self.lineage_graph.get_edge_data(pred, current_entity)
                    
                    upstream_entities.append({
                        'entity_id': pred,
                        'entity_name': entity.name,
                        'entity_type': entity.entity_type.value,
                        'depth': depth,
                        'relationship_type': relationship.get('lineage_type', 'unknown'),
                        'transformation_description': relationship.get('transformation_description', '')
                    })
                    
                    _trace_upstream(pred, depth + 1)
        
        _trace_upstream(entity_id, 0)
        
        return {
            'entity_id': entity_id,
            'entity_name': self.entities[entity_id].name,
            'upstream_entities': upstream_entities,
            'total_upstream_count': len(upstream_entities)
        }
    
    def get_lineage_downstream(self, entity_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """
        Get downstream lineage for an entity.
        
        Parameters
        ----------
        entity_id : str
            Entity ID to trace downstream
        max_depth : int
            Maximum depth to trace
            
        Returns
        -------
        Dict[str, Any]
            Downstream lineage information
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not found")
        
        downstream_entities = []
        visited = set()
        
        def _trace_downstream(current_entity: str, depth: int):
            if depth >= max_depth or current_entity in visited:
                return
            
            visited.add(current_entity)
            
            # Get successors in the lineage graph
            successors = list(self.lineage_graph.successors(current_entity))
            
            for succ in successors:
                if succ not in visited:
                    entity = self.entities[succ]
                    relationship = self.lineage_graph.get_edge_data(current_entity, succ)
                    
                    downstream_entities.append({
                        'entity_id': succ,
                        'entity_name': entity.name,
                        'entity_type': entity.entity_type.value,
                        'depth': depth,
                        'relationship_type': relationship.get('lineage_type', 'unknown'),
                        'transformation_description': relationship.get('transformation_description', '')
                    })
                    
                    _trace_downstream(succ, depth + 1)
        
        _trace_downstream(entity_id, 0)
        
        return {
            'entity_id': entity_id,
            'entity_name': self.entities[entity_id].name,
            'downstream_entities': downstream_entities,
            'total_downstream_count': len(downstream_entities)
        }
    
    def analyze_impact(self, entity_id: str, change_type: ChangeType, 
                      change_description: str) -> ImpactAnalysis:
        """
        Analyze the impact of a change to an entity.
        
        Parameters
        ----------
        entity_id : str
            Entity ID that is changing
        change_type : ChangeType
            Type of change
        change_description : str
            Description of the change
            
        Returns
        -------
        ImpactAnalysis
            Impact analysis results
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not found")
        
        # Get downstream entities
        downstream = self.get_lineage_downstream(entity_id)
        impacted_entities = [e['entity_id'] for e in downstream['downstream_entities']]
        
        # Calculate impact severity
        impact_severity = self._calculate_impact_severity(
            change_type, len(impacted_entities), downstream['downstream_entities']
        )
        
        # Generate recommendations
        recommendations = self._generate_impact_recommendations(
            change_type, impact_severity, impacted_entities
        )
        
        # Create impact analysis
        analysis = ImpactAnalysis(
            change_id=f"impact_{entity_id}_{change_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_entity_id=entity_id,
            change_type=change_type,
            change_description=change_description,
            impacted_entities=impacted_entities,
            impact_severity=impact_severity,
            recommendations=recommendations
        )
        
        self.impact_analyses.append(analysis)
        
        logger.info(f"Impact analysis completed for {entity_id}: {impact_severity} severity, {len(impacted_entities)} entities impacted")
        
        return analysis
    
    def _calculate_impact_severity(self, change_type: ChangeType, 
                                  entity_count: int, 
                                  downstream_entities: List[Dict[str, Any]]) -> str:
        """Calculate impact severity based on change type and affected entities"""
        base_severity = 0.0
        
        # Base severity by change type
        severity_weights = {
            ChangeType.SCHEMA_CHANGE: 0.8,
            ChangeType.DATA_CHANGE: 0.6,
            ChangeType.TRANSFORMATION_CHANGE: 0.7,
            ChangeType.BUSINESS_LOGIC_CHANGE: 0.9,
            ChangeType.DEPRECATED: 0.5,
            ChangeType.ADDED: 0.2,
            ChangeType.REMOVED: 0.9
        }
        
        base_severity = severity_weights.get(change_type, 0.5)
        
        # Adjust based on number of impacted entities
        if entity_count > 20:
            base_severity += 0.2
        elif entity_count > 10:
            base_severity += 0.1
        elif entity_count > 5:
            base_severity += 0.05
        
        # Adjust based on entity types
        critical_types = ['report', 'dashboard', 'api']
        critical_count = sum(1 for e in downstream_entities if e['entity_type'] in critical_types)
        if critical_count > 0:
            base_severity += 0.1 * critical_count
        
        # Normalize to 0-1 range
        final_severity = min(1.0, base_severity)
        
        # Convert to severity level
        thresholds = self.config['impact_analysis']['severity_thresholds']
        if final_severity >= thresholds['critical']:
            return 'critical'
        elif final_severity >= thresholds['high']:
            return 'high'
        elif final_severity >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_impact_recommendations(self, change_type: ChangeType, 
                                       severity: str, 
                                       impacted_entities: List[str]) -> List[str]:
        """Generate recommendations for impact mitigation"""
        recommendations = []
        
        if severity in ['critical', 'high']:
            recommendations.append("Coordinate with all downstream data consumers before making changes")
            recommendations.append("Implement backward compatibility where possible")
            recommendations.append("Plan for staged rollout with rollback capabilities")
        
        if change_type == ChangeType.SCHEMA_CHANGE:
            recommendations.append("Update all dependent data models and transformations")
            recommendations.append("Test all downstream reports and dashboards")
        
        if change_type == ChangeType.DEPRECATED:
            recommendations.append("Provide migration path for dependent systems")
            recommendations.append("Set up sunset timeline with advance notice")
        
        if len(impacted_entities) > 10:
            recommendations.append("Consider automated testing for all impacted entities")
            recommendations.append("Set up monitoring for downstream impact")
        
        return recommendations
    
    def create_lineage_visualization(self, entity_id: str, 
                                   show_upstream: bool = True,
                                   show_downstream: bool = True,
                                   max_depth: int = 3) -> go.Figure:
        """
        Create interactive lineage visualization.
        
        Parameters
        ----------
        entity_id : str
            Central entity ID
        show_upstream : bool
            Whether to show upstream lineage
        show_downstream : bool
            Whether to show downstream lineage
        max_depth : int
            Maximum depth to show
            
        Returns
        -------
        go.Figure
            Interactive lineage visualization
        """
        # Get lineage data
        lineage_data = {'entities': [], 'relationships': []}
        
        # Add central entity
        central_entity = self.entities[entity_id]
        lineage_data['entities'].append({
            'id': entity_id,
            'name': central_entity.name,
            'type': central_entity.entity_type.value,
            'level': 0,
            'is_central': True
        })
        
        # Add upstream entities
        if show_upstream:
            upstream = self.get_lineage_upstream(entity_id, max_depth)
            for entity in upstream['upstream_entities']:
                lineage_data['entities'].append({
                    'id': entity['entity_id'],
                    'name': entity['entity_name'],
                    'type': entity['entity_type'],
                    'level': -entity['depth'],
                    'is_central': False
                })
                
                lineage_data['relationships'].append({
                    'source': entity['entity_id'],
                    'target': entity_id if entity['depth'] == 1 else 'parent',
                    'type': entity['relationship_type']
                })
        
        # Add downstream entities
        if show_downstream:
            downstream = self.get_lineage_downstream(entity_id, max_depth)
            for entity in downstream['downstream_entities']:
                lineage_data['entities'].append({
                    'id': entity['entity_id'],
                    'name': entity['entity_name'],
                    'type': entity['entity_type'],
                    'level': entity['depth'],
                    'is_central': False
                })
                
                lineage_data['relationships'].append({
                    'source': entity_id if entity['depth'] == 1 else 'parent',
                    'target': entity['entity_id'],
                    'type': entity['relationship_type']
                })
        
        # Create visualization
        fig = self._create_lineage_network_plot(lineage_data)
        
        return fig
    
    def _create_lineage_network_plot(self, lineage_data: Dict[str, Any]) -> go.Figure:
        """Create network plot for lineage visualization"""
        # Create networkx graph for layout
        G = nx.DiGraph()
        
        # Add nodes
        for entity in lineage_data['entities']:
            G.add_node(entity['id'], **entity)
        
        # Add edges
        for rel in lineage_data['relationships']:
            if rel['source'] in G.nodes and rel['target'] in G.nodes:
                G.add_edge(rel['source'], rel['target'], **rel)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create traces
        edge_trace = []
        node_trace = []
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_data = G.nodes[node]
            
            color = self._get_node_color(node_data['type'])
            size = 20 if node_data.get('is_central', False) else 15
            
            node_trace.append(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(size=size, color=color, line=dict(width=2, color='black')),
                text=node_data['name'],
                textposition="middle center",
                hoverinfo='text',
                hovertext=f"Name: {node_data['name']}<br>Type: {node_data['type']}<br>ID: {node}",
                showlegend=False
            ))
        
        # Create figure
        fig = go.Figure(data=edge_trace + node_trace)
        
        fig.update_layout(
            title="Data Lineage Visualization",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Data lineage shows the flow of data through your data stack",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            'table': '#3498db',
            'view': '#2ecc71',
            'column': '#f39c12',
            'report': '#e74c3c',
            'dashboard': '#9b59b6',
            'api': '#1abc9c',
            'file': '#34495e'
        }
        return color_map.get(entity_type, '#95a5a6')
    
    def export_lineage_report(self, output_path: str):
        """Export comprehensive lineage report"""
        report = {
            'summary': {
                'total_entities': len(self.entities),
                'total_relationships': len(self.relationships),
                'entity_types': self._get_entity_type_counts(),
                'relationship_types': self._get_relationship_type_counts(),
                'generated_at': datetime.now().isoformat()
            },
            'entities': [
                {
                    'entity_id': entity.entity_id,
                    'name': entity.name,
                    'type': entity.entity_type.value,
                    'schema_name': entity.schema_name,
                    'database_name': entity.database_name,
                    'description': entity.description,
                    'tags': entity.tags,
                    'owner': entity.owner,
                    'created_at': entity.created_at.isoformat(),
                    'updated_at': entity.updated_at.isoformat()
                }
                for entity in self.entities.values()
            ],
            'relationships': [
                {
                    'relationship_id': rel.relationship_id,
                    'source_entity_id': rel.source_entity_id,
                    'target_entity_id': rel.target_entity_id,
                    'lineage_type': rel.lineage_type.value,
                    'transformation_description': rel.transformation_description,
                    'confidence_score': rel.confidence_score,
                    'discovered_by': rel.discovered_by,
                    'created_at': rel.created_at.isoformat()
                }
                for rel in self.relationships.values()
            ],
            'impact_analyses': [
                {
                    'change_id': analysis.change_id,
                    'source_entity_id': analysis.source_entity_id,
                    'change_type': analysis.change_type.value,
                    'change_description': analysis.change_description,
                    'impacted_entities': analysis.impacted_entities,
                    'impact_severity': analysis.impact_severity,
                    'recommendations': analysis.recommendations,
                    'analysis_timestamp': analysis.analysis_timestamp.isoformat()
                }
                for analysis in self.impact_analyses
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Lineage report exported to: {output_path}")
    
    def _get_entity_type_counts(self) -> Dict[str, int]:
        """Get counts of entities by type"""
        counts = {}
        for entity in self.entities.values():
            entity_type = entity.entity_type.value
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts
    
    def _get_relationship_type_counts(self) -> Dict[str, int]:
        """Get counts of relationships by type"""
        counts = {}
        for rel in self.relationships.values():
            rel_type = rel.lineage_type.value
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts

class SQLLineageParser:
    """SQL parser for extracting lineage information"""
    
    def __init__(self):
        self.keywords = ['SELECT', 'FROM', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']
    
    def parse_sql(self, sql_content: str) -> Dict[str, Any]:
        """Parse SQL to extract table references"""
        try:
            # Parse SQL using sqlparse
            parsed = sqlparse.parse(sql_content)[0]
            
            source_tables = set()
            target_tables = set()
            
            # Extract table references
            for token in parsed.flatten():
                if token.ttype is Name:
                    # Simple heuristic: if it's after FROM or JOIN, it's a source table
                    if self._is_table_reference(token, parsed):
                        source_tables.add(token.value)
            
            return {
                'source_tables': list(source_tables),
                'target_tables': list(target_tables),
                'sql_content': sql_content
            }
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {e}")
            return {'source_tables': [], 'target_tables': [], 'sql_content': sql_content}
    
    def _is_table_reference(self, token: Token, parsed: Statement) -> bool:
        """Determine if token is a table reference"""
        # This is a simplified implementation
        # In production, you'd use more sophisticated SQL parsing
        return token.ttype is Name and not token.value.upper() in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR']

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Lineage Tracker")
    parser.add_argument("--dbt-project", help="Path to dbt project for lineage discovery")
    parser.add_argument("--sql-file", help="SQL file to parse for lineage")
    parser.add_argument("--python-file", help="Python file to parse for lineage")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--visualize", "-v", help="Generate lineage visualization")
    parser.add_argument("--entity-id", help="Entity ID for visualization")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = DataLineageTracker()
    
    # Discover lineage
    if args.dbt_project:
        relationships = tracker.discover_lineage_from_dbt(args.dbt_project)
        for rel in relationships:
            tracker.add_lineage_relationship(rel)
        print(f"Discovered {len(relationships)} relationships from dbt project")
    
    if args.sql_file:
        with open(args.sql_file, 'r') as f:
            sql_content = f.read()
        relationships = tracker.discover_lineage_from_sql(sql_content, args.sql_file)
        for rel in relationships:
            tracker.add_lineage_relationship(rel)
        print(f"Discovered {len(relationships)} relationships from SQL file")
    
    if args.python_file:
        relationships = tracker.discover_lineage_from_python(args.python_file)
        for rel in relationships:
            tracker.add_lineage_relationship(rel)
        print(f"Discovered {len(relationships)} relationships from Python file")
    
    # Export report
    if args.output:
        tracker.export_lineage_report(args.output)
        print(f"Lineage report exported to: {args.output}")
    
    # Generate visualization
    if args.visualize and args.entity_id:
        fig = tracker.create_lineage_visualization(args.entity_id)
        fig.write_html(args.visualize)
        print(f"Lineage visualization saved to: {args.visualize}")
    
    # Print summary
    print(f"\n📊 Lineage Tracking Summary:")
    print(f"Total entities: {len(tracker.entities)}")
    print(f"Total relationships: {len(tracker.relationships)}")
    print(f"Entity types: {tracker._get_entity_type_counts()}")
    print(f"Relationship types: {tracker._get_relationship_type_counts()}")

if __name__ == "__main__":
    main() 