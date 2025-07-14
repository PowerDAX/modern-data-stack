{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_grocery_connector_item_master') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_grocery_connector_item_master')) }}
)

SELECT
    account_id,
    connector_id,
    DATE(report_date) AS date_key,
    item_key,
    item_name AS item,
    brand_key AS item_brand_key,
    brand AS item_brand,
    category_key AS item_category_key,
    category AS item_category,
    sub_category_key AS item_subcategory_key,
    sub_category AS item_subcategory,
    supplier_key AS item_supplier_key,
    supplier_name AS item_supplier,
    upc,
    pack_size,
    unit_of_measure,
    item_description,
    item_status
FROM cleaned_source 
