-- Incremental loading utilities for enterprise data patterns

{% macro incremental_source_filter(timestamp_column='ingestion_timestamp', lookback_days=1) %}
  {% if is_incremental() %}
    -- Only process records newer than the latest timestamp in the target table
    -- Include lookback period to handle late-arriving data
    WHERE {{ timestamp_column }} > (
      SELECT COALESCE(
        MAX({{ timestamp_column }}) - INTERVAL '{{ lookback_days }} days',
        '1900-01-01'::timestamp
      ) 
      FROM {{ this }}
    )
  {% endif %}
{% endmacro %}

{% macro incremental_fact_filter(date_column='report_date', lookback_days=3) %}
  {% if is_incremental() %}
    -- For fact tables, filter by business date with lookback for late arrivals
    WHERE {{ date_column }} >= (
      SELECT COALESCE(
        MAX({{ date_column }}) - INTERVAL '{{ lookback_days }} days',
        '1900-01-01'::date
      ) 
      FROM {{ this }}
    )
  {% endif %}
{% endmacro %}

{% macro incremental_dimension_filter(updated_column='ingestion_timestamp') %}
  {% if is_incremental() %}
    -- For dimension tables, use updated timestamp for SCD Type 1
    WHERE {{ updated_column }} > (
      SELECT COALESCE(MAX({{ updated_column })), '1900-01-01'::timestamp) 
      FROM {{ this }}
    )
  {% endif %}
{% endmacro %}

{% macro get_incremental_strategy(table_type='fact') %}
  {% if table_type == 'fact' %}
    -- Fact tables use merge strategy for upserts
    {% set strategy = 'merge' %}
  {% elif table_type == 'dimension' %}
    -- Dimension tables use delete+insert for full refresh capability
    {% set strategy = 'delete+insert' %}
  {% else %}
    -- Default to merge for mixed use cases
    {% set strategy = 'merge' %}
  {% endif %}
  
  {{ return(strategy) }}
{% endmacro %}

{% macro incremental_config(
  unique_key, 
  table_type='fact', 
  exclude_columns=['ingestion_timestamp'],
  partition_by=none,
  cluster_by=none
) %}
  
  {% set config_dict = {
    'materialized': 'incremental',
    'unique_key': unique_key,
    'incremental_strategy': get_incremental_strategy(table_type),
    'merge_exclude_columns': exclude_columns
  } %}
  
  {% if partition_by %}
    {% do config_dict.update({'partition_by': partition_by}) %}
  {% endif %}
  
  {% if cluster_by %}
    {% do config_dict.update({'cluster_by': cluster_by}) %}
  {% endif %}
  
  {{ config(**config_dict) }}
  
{% endmacro %} 