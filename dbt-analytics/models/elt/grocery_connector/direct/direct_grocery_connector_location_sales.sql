-- Direct layer extraction for grocery connector location sales

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'grocery_connector_location_sales_raw') }}
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    location_key,
    item_key,
    upc,
    begin_date,
    end_date,
    sales_dollars,
    sales_units,
    regular_sales_dollars,
    regular_sales_units,
    promotional_sales_dollars,
    promotional_sales_units
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND location_key IS NOT NULL
    AND item_key IS NOT NULL
