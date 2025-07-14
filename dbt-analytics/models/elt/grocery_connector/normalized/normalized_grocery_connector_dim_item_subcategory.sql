-- Normalized item subcategory dimension

{{
    config(
        materialized='table'
    )
}}

WITH item_subcategory_data AS (
    SELECT * FROM {{ ref('stg_grocery_connector_dim_item_subcategory') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_subcategory_key']) }} AS item_subcategory_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        item_subcategory_key,
        item_subcategory
    FROM item_subcategory_data
)

SELECT
    item_subcategory_id,
    account_id,
    connector_id,
    item_subcategory_key,
    item_subcategory
FROM final 