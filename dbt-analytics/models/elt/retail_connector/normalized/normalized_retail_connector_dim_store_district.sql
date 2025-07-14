-- Normalized dimension for product categories
-- Snowflaked dimension referenced by dim_product

{{
    config(
        materialized='table'
    )
}}

WITH store_district_staging AS (
    SELECT * FROM {{ ref('stg_retail_connector_dim_product_subcategory') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_district_key']) }} AS store_district_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_region_key']) }} AS store_region_id,

        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        store_district_key,
        store_district
    FROM store_district_staging
)

SELECT
    store_district_id,
    store_region_id,
    account_id,
    connector_id,
    store_district_key,
    store_district
FROM final 
