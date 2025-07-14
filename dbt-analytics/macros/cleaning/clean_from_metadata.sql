/*
 * ENTERPRISE DATA CLEANING MACRO SUITE
 * ====================================
 * 
 * This macro suite provides comprehensive metadata-driven data cleaning capabilities
 * for enterprise data pipelines, automatically applying appropriate transformations
 * based on BigQuery column metadata and data types.
 * 
 * CAPABILITIES:
 * - Automatic data type-specific cleaning transformations
 * - Metadata-driven column selection and transformation
 * - Connector-specific cleaning logic support
 * - STRING: Trimming, NULL handling, control character removal, case normalization
 * - NUMERIC/FLOAT64/INT64: NULL coalescing with zero defaults
 * - DATE/DATETIME: NULL handling with proper casting
 * - Account/connector ID standardization
 * - Custom filter condition support
 * - BigQuery INFORMATION_SCHEMA integration
 * 
 * USAGE:
 * 1. Standard cleaning:
 *    {{ select_clean_from_metadata(ref('source_table')) }}
 * 
 * 2. With connector-specific logic:
 *    {{ select_clean_from_metadata(ref('source_table'), connector_specific_macro()) }}
 * 
 * 3. With filter conditions:
 *    {{ select_clean_from_metadata(ref('source_table'), filter_condition='date_key >= "2023-01-01"') }}
 * 
 * TRANSFORMATIONS APPLIED:
 * - STRING: UPPER(REPLACE(TRIM(column), control_chars, space)) with NULL handling
 * - NUMERIC: COALESCE(column, 0) for consistent numeric handling
 * - DATE: COALESCE(column, NULL) with proper casting
 * - connector_id: CAST(column AS STRING) for standardization
 * - Control character removal (ASCII 26) for data quality
 * 
 * INTEGRATION:
 * - Used by all cleaned layer models via {{ select_clean_from_metadata() }}
 * - Supports enterprise connector-specific transformations
 * - Integrates with BigQuery metadata for automatic schema discovery
 */

{% macro column_list_from_metadata(relation, connector_specific=None) %}

{% set data_structure_query %}
    with table_schema as (
        select * from `{{relation.database}}`.{{relation.schema}}.INFORMATION_SCHEMA.COLUMNS
    ),
    cleaned as (
        select
            column_name,
            data_type,
            ordinal_position,
            case
                -- account/config
                when column_name = 'account_id' then column_name
                when column_name = 'connector_id' then concat( 'CAST( ', column_name, ' AS STRING) AS ', column_name)

                --connector specific cleaning
                {% if connector_specific is not none %}
                {{ connector_specific() }}
                {% endif %}

                --standard cleaning
                when data_type = 'STRING' then concat('UPPER( REPLACE( CASE WHEN TRIM(`', column_name, '`) = "" THEN CAST( NULL AS STRING) ELSE COALESCE( TRIM(`', column_name, '`), CAST( NULL AS STRING)) END, code_points_to_string([26]), " ") )AS `', column_name, '`')
                when data_type = 'NUMERIC' then concat('COALESCE( `', column_name, '`, 0) AS `', column_name, '`')
                when data_type = 'FLOAT64' then concat('COALESCE( `', column_name, '`, 0) AS `', column_name, '`')
                when data_type = 'INT64' then concat('COALESCE( `', column_name, '`, 0) AS `', column_name, '`')
                when data_type = 'DATE' then concat('COALESCE( `', column_name, '`, CAST( NULL AS DATE)) AS `', column_name, '`')
                when data_type = 'DATETIME' then concat('COALESCE( `', column_name, '`, CAST( NULL AS DATETIME)) AS `', column_name, '`')
                else concat('`',column_name, '`')
                end as cleaned_column
        from table_schema
        where table_name = '{{relation.identifier}}'
    ),
    output as (
        select * from cleaned order by ordinal_position
    )

    select string_agg( distinct cleaned_column) as columns from output

{% endset %}

{% set results = run_query(data_structure_query) %}

{% if execute %}
    {# Return the first column #}
    {% do return(results.columns[0].values()) %}
{% else %}
    {% do return([]) %}
{% endif %}

{% endmacro %}


{% macro clean_from_metadata(relation, connector_specific=None, filter_condition=none) %}
with cleaned_source as (
    {{ select_clean_from_metadata(relation, connector_specific, filter_condition) }}
)
{% endmacro %}

{% macro select_clean_from_metadata(relation, connector_specific=None, filter_condition=none) %}
{% set result_list = column_list_from_metadata(relation, connector_specific) %}
select
    {% for item in result_list -%}
    {{ item -}}
    {% if not loop.last -%}
    ,
    {% endif %}
    {% endfor %}
from {{ relation }}
{% if filter_condition is not none %}
where {{ filter_condition }}
{% endif %}
{% endmacro %}

