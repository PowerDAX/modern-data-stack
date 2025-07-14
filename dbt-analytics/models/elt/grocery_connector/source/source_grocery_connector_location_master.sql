-- Source layer deduplication for grocery connector location master

{{
    config(
        materialized='table'
    )
}}

WITH direct AS (
    SELECT * FROM {{ ref('direct_grocery_connector_location_master') }}
),

deduplicated AS (
    SELECT *
    FROM direct
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY 
            account_id,
            connector_id,
            location_key
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
FROM deduplicated
