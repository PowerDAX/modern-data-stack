-- Source layer deduplication for retail connector store master

{{
    config(
        materialized='table'
    )
}}

WITH direct AS (
    SELECT * FROM {{ ref('direct_retail_connector_store_master') }}
),

deduplicated AS (
    SELECT *
    FROM direct
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY 
            account_id,
            connector_id,
            store_key
        ORDER BY 
            report_date DESC,
            ingestion_timestamp DESC
    ) = 1
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
    store_latitude,
    store_longitude,
    store_manager,
    store_phone,
    store_status,
    district_key,
    district,
    region_key,
    region
FROM deduplicated 
