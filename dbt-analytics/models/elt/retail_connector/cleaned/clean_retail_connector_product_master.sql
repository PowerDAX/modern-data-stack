{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_retail_connector_product_master') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_retail_connector_product_master')) }}
)

SELECT
    account_id,
    connector_id,
    DATE(report_date) AS date_key,
    product_key,
    product_name AS product,
    brand_key AS product_brand_key,
    brand AS product_brand,
    category_key AS product_category_key,
    category AS product_category,
    subcategory_key AS product_subcategory_key,
    subcategory AS product_subcategory,
    supplier_key AS product_supplier_key,
    supplier_name AS product_supplier,
    upc,
    pack_size,
    unit_of_measure,
    product_description,
    product_status
FROM cleaned_source
