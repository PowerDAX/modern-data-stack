-- Direct layer extraction for retail connector store inventory

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'retail_connector_store_inventory_raw') }}
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    store_id,
    product_id,
    upc,
    report_date,
    CAST(on_hand_quantity AS INTEGER) AS on_hand_quantity,
    CAST(on_hand_amount AS NUMERIC) AS on_hand_amount,
    CAST(on_order_quantity AS INTEGER) AS on_order_quantity,
    CAST(on_order_amount AS NUMERIC) AS on_order_amount
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND store_id IS NOT NULL
    AND product_id IS NOT NULL 
