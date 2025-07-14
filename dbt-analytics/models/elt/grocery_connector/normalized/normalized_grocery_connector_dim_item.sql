-- Normalized item dimension with snowflaked references
-- References dim_item_category and dim_supplier

{{
    config(
        materialized='table'
    )
}}

WITH item_attributes AS (
    SELECT * FROM {{ ref('stg_grocery_connector_item_attributes') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_key']) }} AS item_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_brand_key']) }} AS item_brand_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_subcategory_key']) }} AS item_subcategory_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_supplier_key']) }} AS item_supplier_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        item_key,
        item,
        upc,
        pack_size,
        unit_of_measure,
        item_description,
        item_status
    FROM item_attributes
)

SELECT
    item_id,
    item_brand_id,
    item_subcategory_id,
    item_supplier_id,
    account_id,
    connector_id,
    item_key,
    item,
    upc,
    pack_size,
    unit_of_measure,
    item_description,
    item_status
FROM final 


