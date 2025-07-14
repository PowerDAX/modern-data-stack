/*
 * ENTERPRISE TABLE MANAGEMENT MACRO SUITE
 * ========================================
 * 
 * This macro suite provides comprehensive table governance, monitoring, and 
 * data quality capabilities for enterprise-grade data pipelines.
 * 
 * CAPABILITIES:
 * - Automated BigQuery table labeling for governance and cost tracking
 * - Model execution auditing with comprehensive metadata
 * - Data freshness monitoring with configurable thresholds
 * - Environment-aware configuration management
 * - Automated row count tracking and quality metrics
 * - Post-hook integration for seamless execution
 * 
 * USAGE:
 * 1. Table Labeling (BigQuery):
 *    - Automatically applied via post-hooks
 *    - Includes environment, schema, tags, materialization
 *    - Supports cost center tracking and governance
 * 
 * 2. Model Execution Auditing:
 *    - Tracks execution time, row counts, environment
 *    - Logs comprehensive model metadata
 *    - Integrates with data quality monitoring
 * 
 * 3. Data Freshness Monitoring:
 *    - Configurable thresholds by model tags
 *    - Critical models: 2 hours, Daily: 30 hours
 *    - Returns freshness status and metrics
 * 
 * CONFIGURATION:
 * - Add to post-hooks in model config:
 *   post_hook: ["{{ set_table_labels() }}", "{{ audit_model_execution() }}"]
 * - Control via vars: enable_data_quality_monitoring, freshness_threshold_hours
 */

{% macro set_table_labels() %}
    {%- if target.type == 'bigquery' and execute -%}
        {%- set labels = {} -%}
        
        {%- set environment = get_environment() -%}
        {%- do labels.update({'environment': environment}) -%}
        
        {%- do labels.update({'project': 'showcase'}) -%}
        {%- do labels.update({'dbt_version': dbt_version.replace('.', '_')}) -%}
        {%- do labels.update({'created_at': run_started_at.strftime('%Y_%m_%d')}) -%}
        
        {%- if this.schema -%}
            {%- do labels.update({'schema': this.schema.lower()}) -%}
        {%- endif -%}
        
        {%- if model.config.tags -%}
            {%- for tag in model.config.tags[:3] -%}  {# Limit to 3 tags due to BigQuery label limits #}
                {%- set tag_key = 'tag_' ~ loop.index -%}
                {%- set tag_value = tag.replace(':', '_').replace('-', '_') -%}
                {%- do labels.update({tag_key: tag_value}) -%}
            {%- endfor -%}
        {%- endif -%}
        
        {%- if model.config.materialized -%}
            {%- do labels.update({'materialization': model.config.materialized}) -%}
        {%- endif -%}
        
        {%- set label_sql -%}
            ALTER TABLE {{ this }} SET OPTIONS (
                labels = {{ labels | tojson }}
            )
        {%- endset -%}
        
        {{ log("Setting table labels: " ~ labels, info=true) }}
        {{ run_query(label_sql) }}
    {%- endif -%}
{% endmacro %}

{% macro audit_model_execution() %}
    {%- if execute and var('enable_data_quality_monitoring', true) -%}
        {%- set audit_data = {
            'model_name': this.name,
            'schema': this.schema,
            'database': this.database,
            'materialization': model.config.materialized,
            'execution_time': run_started_at.strftime('%Y-%m-%d %H:%M:%S'),
            'target': target.name,
            'environment': get_environment(),
            'dbt_version': dbt_version
        } -%}
        
        {%- if model.config.tags -%}
            {%- do audit_data.update({'tags': model.config.tags | join(',')}) -%}
        {%- endif -%}
        
        {%- set row_count_sql -%}
            SELECT COUNT(*) as row_count FROM {{ this }}
        {%- endset -%}
        
        {%- if execute -%}
            {%- set results = run_query(row_count_sql) -%}
            {%- if results -%}
                {%- set row_count = results.columns[0].values()[0] -%}
                {%- do audit_data.update({'row_count': row_count}) -%}
            {%- endif -%}
        {%- endif -%}
        
        {{ log("Model execution audit: " ~ audit_data, info=true) }}
        
        {# In a real implementation, this would write to an audit table #}
        {# For the showcase, we'll just log the information #}
    {%- endif -%}
{% endmacro %}

{% macro get_freshness_threshold(model_name) %}
    {%- set default_threshold = var('freshness_threshold_hours', 24) -%}
    
    {%- if 'critical' in model.config.tags -%}
        {%- do return(2) -%}  {# 2 hours for critical models #}
    {%- elif 'hourly' in model.config.tags -%}
        {%- do return(6) -%}   {# 6 hours for hourly models #}
    {%- elif 'daily' in model.config.tags -%}
        {%- do return(30) -%}  {# 30 hours for daily models #}
    {%- else -%}
        {%- do return(default_threshold) -%}
    {%- endif -%}
{% endmacro %}

{% macro check_data_freshness(column_name='updated_at') %}
    {%- set freshness_threshold = get_freshness_threshold(this.name) -%}
    
    SELECT 
        '{{ this }}' as table_name,
        MAX({{ column_name }}) as latest_update,
        CURRENT_TIMESTAMP() as check_time,
        DATETIME_DIFF(CURRENT_TIMESTAMP(), MAX({{ column_name }}), HOUR) as hours_since_update,
        {{ freshness_threshold }} as threshold_hours,
        CASE 
            WHEN DATETIME_DIFF(CURRENT_TIMESTAMP(), MAX({{ column_name }}), HOUR) > {{ freshness_threshold }}
            THEN 'STALE'
            ELSE 'FRESH'
        END as freshness_status
    FROM {{ this }}
    WHERE {{ column_name }} IS NOT NULL
{% endmacro %} 
