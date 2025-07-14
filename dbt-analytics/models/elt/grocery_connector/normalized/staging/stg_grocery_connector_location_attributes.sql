{{
    config(
        materialized='table'
    )
}}

WITH location_master AS (
    SELECT * FROM {{ ref('clean_grocery_connector_location_master') }}
),

location_sales AS (
    SELECT * FROM {{ ref('clean_grocery_connector_location_sales') }}
),

location_inventory AS (
    SELECT * FROM {{ ref('clean_grocery_connector_location_inventory') }}
),

unioned AS (
    SELECT
        account_id,
        connector_id,
        location_key,
        location_chain_key,
        location_chain,
        location_district_key,
        location_district,
        location_region_key,
        location_region,
        location,
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
        1 AS source_priority,
        date_key
    FROM location_master
    UNION ALL
    SELECT
        account_id,
        connector_id,
        location_key,
        CAST(NULL AS STRING) AS location_chain_key,
        CAST(NULL AS STRING) AS location_chain,
        CAST(NULL AS STRING) AS location_district_key,
        CAST(NULL AS STRING) AS location_district,
        CAST(NULL AS STRING) AS location_region_key,
        CAST(NULL AS STRING) AS location_region,
        CAST(NULL AS STRING) AS location,
        CAST(NULL AS STRING) AS location_number,
        CAST(NULL AS STRING) AS location_type,
        CAST(NULL AS STRING) AS location_address,
        CAST(NULL AS STRING) AS location_city,
        CAST(NULL AS STRING) AS location_state,
        CAST(NULL AS STRING) AS location_zip,
        CAST(NULL AS STRING) AS location_country,
        CAST(NULL AS NUMERIC) AS location_latitude,
        CAST(NULL AS NUMERIC) AS location_longitude,
        CAST(NULL AS STRING) AS location_manager,
        CAST(NULL AS STRING) AS location_phone,
        CAST(NULL AS STRING) AS location_status,
        2 AS source_priority,
        date_key
    FROM location_sales
    UNION ALL
    SELECT
        account_id,
        connector_id,
        location_key,
        CAST(NULL AS STRING) AS location_chain_key,
        CAST(NULL AS STRING) AS location_chain,
        CAST(NULL AS STRING) AS location_district_key,
        CAST(NULL AS STRING) AS location_district,
        CAST(NULL AS STRING) AS location_region_key,
        CAST(NULL AS STRING) AS location_region,
        CAST(NULL AS STRING) AS location,
        CAST(NULL AS STRING) AS location_number,
        CAST(NULL AS STRING) AS location_type,
        CAST(NULL AS STRING) AS location_address,
        CAST(NULL AS STRING) AS location_city,
        CAST(NULL AS STRING) AS location_state,
        CAST(NULL AS STRING) AS location_zip,
        CAST(NULL AS STRING) AS location_country,
        CAST(NULL AS NUMERIC) AS location_latitude,
        CAST(NULL AS NUMERIC) AS location_longitude,
        CAST(NULL AS STRING) AS location_manager,
        CAST(NULL AS STRING) AS location_phone,
        CAST(NULL AS STRING) AS location_status,
        3 AS source_priority,
        date_key
    FROM location_inventory
),

deduped AS (
    SELECT *
    FROM unioned
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            account_id,
            connector_id,
            location_key
        ORDER BY
            source_priority,
            date_key DESC
    ) = 1
)

SELECT
    account_id,
    connector_id,
    location_key,
    location_chain_key,
    location_chain,
    location_district_key,
    location_district,
    location_region_key,
    location_region,
    location,
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
    source_priority,
    date_key
FROM deduped 


