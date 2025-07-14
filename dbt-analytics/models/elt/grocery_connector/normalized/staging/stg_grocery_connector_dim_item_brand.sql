{{
    config(
        materialized='table'
    )
}}

WITH item_attributes AS (
    SELECT * FROM {{ ref('stg_grocery_connector_item_attributes') }}
),

deduped AS (
    {{ deduplicate_using_prioritization(
        table_name='item_attributes',
        key_columns=['account_id', 'connector_id', 'item_brand_key'],
        attribute_columns=['item_brand'],
        sorting=['date_key DESC NULLS LAST'],
        ignore_nulls=true
    ) }}
)

SELECT
    account_id,
    connector_id,
    item_brand_key,
    item_brand
FROM deduped 