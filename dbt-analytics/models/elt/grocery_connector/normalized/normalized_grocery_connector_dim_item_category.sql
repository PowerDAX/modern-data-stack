-- Normalized item category dimension

{{
    config(
        materialized='table'
    )
}}

WITH item_category_data AS (
    SELECT * FROM {{ ref('stg_grocery_connector_dim_item_category') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_category_key']) }} AS item_category_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        item_category_key,
        item_category
    FROM item_category_data
)

SELECT
    item_category_id,
    account_id,
    connector_id,
    item_category_key,
    item_category
FROM final 


