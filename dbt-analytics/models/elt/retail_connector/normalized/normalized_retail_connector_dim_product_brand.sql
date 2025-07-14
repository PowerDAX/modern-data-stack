-- Normalized dimension for product categories
-- Snowflaked dimension referenced by dim_product_sub_category

{{
    config(
        materialized='table'
    )
}}

WITH product_brand_staging AS (
    SELECT * FROM {{ ref('stg_retail_connector_dim_product_brand') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_brand_key']) }} AS product_brand_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        product_brand_key,
        product_brand
    FROM product_brand_staging
)

SELECT
    product_brand_id,
    account_id,
    connector_id,
    product_brand_key,
    product_brand
FROM final 
