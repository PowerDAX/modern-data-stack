-- Direct layer extraction for grocery connector item master

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'grocery_connector_item_master_raw') }}
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    item_key,
    item_name,
    brand_key,
    brand,
    category_key,
    category,
    subcategory_key,
    subcategory,
    supplier_key,
    supplier_name,
    upc,
    pack_size,
    unit_of_measure,
    item_description,
    item_status
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND item_key IS NOT NULL
