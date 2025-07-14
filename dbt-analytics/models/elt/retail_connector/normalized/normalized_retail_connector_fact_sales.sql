-- Normalized sales fact table
-- References dim_store and dim_product

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
    SELECT * FROM {{ ref('clean_retail_connector_store_sales') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_key', 'product_key', 'date_key']) }} AS sales_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_key']) }} AS store_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_key']) }} AS product_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        date_key,
        
        -- facts
        sales_amount,
        sales_quantity
    FROM sales_source
)

SELECT
    sales_id,
    store_id,
    product_id,
    account_id,
    connector_id,
    date_key,
    sales_amount,
    sales_quantity
FROM final 
