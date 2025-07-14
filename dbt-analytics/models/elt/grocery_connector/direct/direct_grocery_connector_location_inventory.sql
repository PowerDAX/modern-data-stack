-- Direct layer extraction for grocery connector location inventory

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'grocery_connector_location_inventory_raw') }}
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    location_key,
    item_key,
    upc,
    on_hand_units,
    on_hand_dollars,
    on_order_units,
    on_order_dollars,
    allocated_units,
    available_units
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND location_key IS NOT NULL
    AND item_key IS NOT NULL
