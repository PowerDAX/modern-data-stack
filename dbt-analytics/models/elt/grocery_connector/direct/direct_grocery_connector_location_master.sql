-- Direct layer extraction for grocery connector location master

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'grocery_connector_location_master_raw') }}
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    chain_key,
    chain,
    location_key,
    location_name,
    location_number,
    location_type,
    location_address,
    location_city,
    location_state,
    location_zip,
    location_country,
    location_latitude,
    location_longitude,
    location_manager,
    location_phone,
    location_status,
    district_key,
    district,
    region_key,
    region
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND location_key IS NOT NULL
