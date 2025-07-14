/*
 * ENTERPRISE QUERY METADATA MACRO SUITE
 * =====================================
 * 
 * This macro suite provides comprehensive query metadata tracking, cost optimization,
 * and BigQuery governance capabilities for the Modern Data Stack Showcase.
 * 
 * CAPABILITIES:
 * - Automated query metadata injection for cost tracking
 * - BigQuery label sanitization and governance
 * - Model type classification and tagging
 * - Environment detection and configuration
 * - JSON-formatted metadata for query analysis
 * - Support for BigQuery job labels and query comments
 * 
 * USAGE:
 * 1. Configured globally in dbt_project.yml:
 *    query-comment:
 *      comment: "{{ query_comment() }}"
 *      job-label: true
 * 
 * 2. Automatic metadata injection includes:
 *    - Model name, schema, database
 *    - Environment (dev/test/prod)
 *    - dbt version and profile information
 *    - Model type (fact/dimension/aggregate)
 *    - Materialization strategy
 *    - Incremental configuration
 *    - Partition and cluster fields
 *    - Model tags and metadata
 */

{% macro sanitize_label(label) %}
    {%- if label is none %}
        {%- do return("none") %}
    {%- elif label is undefined %}
        {%- do return("undefined") %}
    {%- elif not label %}
        {%- do return("") %}
    {%- else %}
        {%- set valid_chars = modules.re.sub('[^-\w]', '-', label) %}
        {%- do return((valid_chars | lower)[:63] if valid_chars is not none and valid_chars | length > 63 else valid_chars | lower) %}
    {%- endif %}
{%- endmacro %}

{% macro parse_model_type(node_name) %}
    {% set pattern_prefix = '(^(staging|intermediate|marts)_)' %}
    {% set pattern_suffix = '_(fact|dim|bridge|agg)$' %}
    {% set re = modules.re %}
    
    {% if re.search('fact', node_name) %}
        {% do return('fact') %}
    {% elif re.search('dim', node_name) %}
        {% do return('dimension') %}
    {% elif re.search('bridge', node_name) %}
        {% do return('bridge') %}
    {% elif re.search('agg', node_name) %}
        {% do return('aggregate') %}
    {% else %}
        {% do return('unknown') %}
    {% endif %}
{% endmacro %}

{% macro get_environment() %}
    {%- set environment = target['name'] %}
    {%- if 'dev' in environment %}
        {%- do return('development') %}
    {%- elif 'test' in environment %}
        {%- do return('testing') %}
    {%- elif 'prod' in environment %}
        {%- do return('production') %}
    {%- else %}
        {%- do return('unknown') %}
    {%- endif %}
{% endmacro %}

{% macro query_comment(node) %}
    {%- set environment = get_environment() %}
    
    {%- set comment_dict = {
        'source': 'dbt-showcase',
        'environment': sanitize_label(environment),
        'dbt-version': sanitize_label(dbt_version),
        'dbt-profile': sanitize_label(target['profile_name']),
        'dbt-target': sanitize_label(target['name']),
        'full-refresh': sanitize_label(flags.FULL_REFRESH | string),
        'execution-time': sanitize_label(run_started_at.strftime('%Y-%m-%d %H:%M:%S')),
        'project': 'modern-data-stack-showcase'
    } %}

    {%- if node is not none %}
        {%- set model_type = parse_model_type(node.name) %}
        
        {%- set node_tags = [] %}
        {%- if node.config.tags %}
            {%- for tag in node.config.tags %}
                {%- do node_tags.append(sanitize_label(tag)) %}
            {%- endfor %}
        {%- endif %}
        
        {%- do comment_dict.update({
            'node-name': sanitize_label(node.name),
            'node-type': sanitize_label(node.resource_type),
            'model-type': sanitize_label(model_type),
            'materialization': sanitize_label(node.config.materialized),
            'schema': sanitize_label(node.schema),
            'database': sanitize_label(node.database),
            'package': sanitize_label(node.package_name),
            'tags': ','.join(node_tags[:5]),  # Limit to first 5 tags
            'incremental-strategy': sanitize_label(node.config.incremental_strategy) if node.config.incremental_strategy else 'none',
            'unique-key': sanitize_label(node.config.unique_key) if node.config.unique_key else 'none'
        }) %}
        
        {%- if node.config.partition_by %}
            {%- do comment_dict.update({
                'partition-field': sanitize_label(node.config.partition_by.field) if node.config.partition_by.field else 'none'
            }) %}
        {%- endif %}
        
        {%- if node.config.cluster_by %}
            {%- do comment_dict.update({
                'cluster-fields': ','.join(node.config.cluster_by[:3])  # Limit to first 3 cluster fields
            }) %}
        {%- endif %}
    {%- endif %}
    
    {%- do return(tojson(comment_dict)) %}
{%- endmacro %} 
