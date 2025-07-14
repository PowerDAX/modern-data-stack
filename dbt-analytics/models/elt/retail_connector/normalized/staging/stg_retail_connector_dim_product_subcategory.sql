{{
    config(
        materialized='table'
    )
}}

WITH product_attributes AS (
    SELECT * FROM {{ ref('stg_retail_connector_product_attributes') }}
),

deduped AS (
    {{ deduplicate_using_prioritization(
        table_name='product_attributes',
        key_columns=['account_id', 'connector_id', 'product_subcategory_key'],
        attribute_columns=['product_subcategory', 'product_category_key'],
        sorting=['date_key DESC NULLS LAST'],
        ignore_nulls=true
    ) }}
)

SELECT
    account_id,
    connector_id,
    product_subcategory_key,
    product_subcategory,
    product_category_key
FROM deduped 
