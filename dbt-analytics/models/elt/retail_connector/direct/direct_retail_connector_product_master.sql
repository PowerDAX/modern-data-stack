-- Direct layer extraction for retail connector product master

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'retail_connector_product_master_raw') }}
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    logical_date,
    product_key,
    product_name,
    brand_key,
    brand,
    category_key,
    category,
    sub_category_key,
    sub_category,
    supplier_key,
    supplier_name,
    upc,
    pack_size,
    unit_of_measure,
    product_description,
    product_status
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND product_key IS NOT NULL 
