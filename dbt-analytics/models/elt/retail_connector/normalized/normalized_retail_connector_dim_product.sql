-- Normalized product dimension with snowflaked references
-- References dim_product_category and dim_supplier

{{
    config(
        materialized='table'
    )
}}

WITH product_attributes AS (
    SELECT * FROM {{ ref('stg_retail_connector_product_attributes') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_key']) }} AS product_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_brand_key']) }} AS product_brand_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_subcategory_key']) }} AS product_subcategory_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_supplier_key']) }} AS product_supplier_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        product_key,
        product,
        upc,
        pack_size,
        unit_of_measure,
        product_description,
        product_status
    FROM product_attributes
)

SELECT
    product_id,
    product_brand_id,
    product_subcategory_id,
    product_supplier_id,
    account_id,
    connector_id,
    product_key,
    product,
    upc,
    pack_size,
    unit_of_measure,
    product_description,
    product_status
FROM final 
