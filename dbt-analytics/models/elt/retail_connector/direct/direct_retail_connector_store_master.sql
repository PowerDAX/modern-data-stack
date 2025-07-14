-- Direct layer extraction for retail connector store master

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'retail_connector_store_master_raw') }}
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    chain_key,
    chain,
    store_key,
    store_name,
    store_number,
    store_type,
    store_address,
    store_city,
    store_state,
    store_zip,
    store_country,
    CAST(store_latitude AS NUMERIC) AS store_latitude,
    CAST(store_longitude AS NUMERIC) AS store_longitude,
    store_manager,
    store_phone,
    store_status,
    district_key,
    district,
    region_key,
    region
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND store_id IS NOT NULL 
