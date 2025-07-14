/*
 * ENTERPRISE DEDUPLICATION MACRO
 * ===============================
 * 
 * This macro provides advanced deduplication capabilities with configurable 
 * prioritization strategies for enterprise data integration scenarios.
 * 
 * CAPABILITIES:
 * - Multi-column key-based deduplication
 * - Configurable sorting strategies for data prioritization
 * - NULL handling options (ignore or preserve)
 * - Array aggregation with safe offset handling
 * - Support for complex source priority rules
 * - BigQuery-optimized SQL generation
 * 
 * USAGE:
 * {{ deduplicate_using_prioritization(
 *     table_name=ref('source_table'),
 *     key_columns=['account_id', 'product_key'],
 *     attribute_columns=['product_name', 'brand', 'price'],
 *     sorting=['source_priority ASC', 'report_date DESC'],
 *     ignore_nulls=true
 * ) }}
 * 
 * PARAMETERS:
 * - table_name: Source table reference or name
 * - key_columns: List of columns to group by (deduplication keys)
 * - attribute_columns: List of columns to prioritize and select
 * - sorting: List of ORDER BY clauses for prioritization
 * - ignore_nulls: Boolean to ignore NULL values in aggregation
 * 
 * EXAMPLE USE CASE:
 * Source layer models use this to prioritize data from multiple sources
 * based on source_priority rankings, ensuring highest quality data wins.
 * 
 * NOTE: This does not guarantee ending up with the same row for all columns
 * when ignoring nulls, as different columns may select from different source rows.
 */

{% macro deduplicate_using_prioritization(table_name, key_columns, attribute_columns, sorting, ignore_nulls = false) %}
    SELECT
        {% for key_column in key_columns %}
        {{ key_column }},
        {% endfor %}
        {% for attribute_column in attribute_columns %}
        ARRAY_AGG(
            {{ attribute_column }} {{ "IGNORE NULLS" if ignore_nulls }}
            ORDER BY {{ sorting|join(',')}} LIMIT 1
        )[SAFE_OFFSET(0)] AS {{ attribute_column }}{{ "," if not loop.last else "" }}
        {% endfor %}
    FROM {{ table_name }}
    {{ group_by(key_columns|length) }}
{% endmacro %}
