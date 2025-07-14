-- Normalized sales fact table
-- References dim_location and dim_item

{{
    config(
        materialized='table',
        post_hook=[
            "{{ set_table_labels() }}",
            "{{ audit_model_execution() }}"
        ],
        tags=['critical', 'fact_table']
    )
}}

WITH sales_source AS (
    SELECT * FROM {{ ref('clean_grocery_connector_location_sales') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_key', 'item_key', 'date_key']) }} AS sales_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_key']) }} AS location_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_key']) }} AS item_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        date_key,
        
        -- facts
        sales_dollars,
        sales_units,
        regular_sales_dollars,
        regular_sales_units,
        promotional_sales_dollars,
        promotional_sales_units
    FROM sales_source
)

SELECT
    sales_id,
    location_id,
    item_id,
    account_id,
    connector_id,
    date_key,
    sales_dollars,
    sales_units,
    regular_sales_dollars,
    regular_sales_units,
    promotional_sales_dollars,
    promotional_sales_units
FROM final 


