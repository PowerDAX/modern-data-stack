-- Normalized dimension for product categories
-- Snowflaked dimension referenced by dim_product_sub_category

{{
    config(
        materialized='table'
    )
}}

WITH product_category_staging AS (
    SELECT * FROM {{ ref('stg_retail_connector_dim_product_category') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_category_key']) }} AS product_category_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        product_category_key,
        product_category
    FROM product_category_staging
)

SELECT
    product_category_id,
    account_id,
    connector_id,
    product_category_key,
    product_category
FROM final 
