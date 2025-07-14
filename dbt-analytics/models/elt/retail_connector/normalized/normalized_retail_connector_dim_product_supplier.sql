-- Normalized dimension for suppliers
-- Snowflaked dimension referenced by dim_product

{{
    config(
        materialized='table'
    )
}}

WITH supplier_staging AS (
    SELECT * FROM {{ ref('stg_retail_connector_dim_product_supplier') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_supplier_key']) }} AS product_supplier_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        product_supplier_key,
        product_supplier
    FROM supplier_staging
)

SELECT
    product_supplier_id,
    account_id,
    connector_id,
    product_supplier_key,
    product_supplier
FROM final 
