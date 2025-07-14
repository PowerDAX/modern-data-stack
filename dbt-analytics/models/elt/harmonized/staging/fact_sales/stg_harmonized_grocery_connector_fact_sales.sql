-- Harmonized staging for grocery connector sales
-- Standardizes schema for cross-connector integration

{{
    config(
        materialized='table'
    )
}}

WITH source AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_fact_sales') }}
),

harmonized AS (
    SELECT
        'grocery_connector' AS retailer,
        sales_id,
        location_id AS store_id,
        item_id AS product_id,
        account_id,
        connector_id,
        date_key,
        
        -- Standardized fact columns
        sales_dollars AS sales_amount,
        sales_units AS sales_quantity,
        
        -- Calculated harmonized metrics
        SAFE_DIVIDE(sales_dollars, sales_units) AS avg_selling_price
    FROM source
)

SELECT * FROM harmonized 
