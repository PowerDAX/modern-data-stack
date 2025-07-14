-- Normalized item brand dimension

{{
    config(
        materialized='table'
    )
}}

WITH item_brand_data AS (
    SELECT * FROM {{ ref('stg_grocery_connector_dim_item_brand') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_brand_key']) }} AS item_brand_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        item_brand_key,
        item_brand
    FROM item_brand_data
)

SELECT
    item_brand_id,
    account_id,
    connector_id,
    item_brand_key,
    item_brand
FROM final 