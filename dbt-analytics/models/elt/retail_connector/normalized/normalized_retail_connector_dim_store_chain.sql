-- Normalized dimension for product categories
-- Snowflaked dimension referenced by dim_product_sub_category

{{
    config(
        materialized='table'
    )
}}

WITH chain_staging AS (
    SELECT * FROM {{ ref('stg_retail_connector_dim_chain') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_chain_key']) }} AS store_chain_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        store_chain_key,
        store_chain
    FROM chain_staging
)

SELECT
    store_chain_id,
    account_id,
    connector_id,
    store_chain_key,
    store_chain
FROM final 
