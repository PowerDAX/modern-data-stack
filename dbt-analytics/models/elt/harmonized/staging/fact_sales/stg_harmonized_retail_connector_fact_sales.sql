-- Harmonized staging for retail connector sales
-- Standardizes schema for cross-connector integration

WITH source AS (
    SELECT * FROM {{ ref('normalized_retail_connector_fact_sales') }}
),

harmonized AS (
    SELECT
        'retail_connector' AS retailer,
        sales_id,
        store_id,
        product_id,
        account_id,
        connector_id,
        date_key,
        
        -- Standardized fact columns
        sales_amount,
        sales_quantity,
        
        -- Calculated harmonized metrics
        SAFE_DIVIDE(sales_amount, sales_quantity) AS avg_selling_price
    FROM source
)

SELECT * FROM harmonized 


