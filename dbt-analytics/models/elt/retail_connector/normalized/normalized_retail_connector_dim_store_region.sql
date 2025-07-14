-- Normalized dimension for product categories
-- Snowflaked dimension referenced by dim_product_sub_category

{{
    config(
        materialized='table'
    )
}}

WITH store_region_staging AS (
    SELECT * FROM {{ ref('stg_retail_connector_dim_store_region') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_region_key']) }} AS store_region_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        store_region_key,
        store_region
    FROM store_region_staging
)

SELECT
    store_region_id,
    account_id,
    connector_id,
    store_region_key,
    store_region
FROM final 
