-- Normalized item supplier dimension

{{
    config(
        materialized='table'
    )
}}

WITH item_supplier_data AS (
    SELECT * FROM {{ ref('stg_grocery_connector_dim_item_supplier') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_supplier_key']) }} AS item_supplier_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        item_supplier_key,
        item_supplier
    FROM item_supplier_data
)

SELECT
    item_supplier_id,
    account_id,
    connector_id,
    item_supplier_key,
    item_supplier
FROM final 