{{
    config(
        materialized='table'
    )
}}

WITH store_master AS (
    SELECT * FROM {{ ref('clean_retail_connector_store_master') }}
),

store_sales AS (
    SELECT * FROM {{ ref('clean_retail_connector_store_sales') }}
),

store_inventory AS (
    SELECT * FROM {{ ref('clean_retail_connector_store_inventory') }}
),

unioned AS (
    SELECT
        account_id,
        connector_id,
        store_chain_key,
        store_chain,
        store_key,
        store_address,
        store_city,
        store_state,
        store_zip,
        store_country,
        store_latitude,
        store_longitude,
        store,
        store_number,
        store_type,
        store_manager,
        store_phone,
        store_status,
        store_district_key,
        store_district,
        store_region_key,
        store_region,
        report_date AS date_key,
        1 AS source_priority
    FROM store_master
    UNION ALL
    SELECT
        account_id,
        connector_id,
        CAST(NULL AS STRING) AS store_chain_key,
        CAST(NULL AS STRING) AS store_chain,
        store_key,
        CAST(NULL AS STRING) AS store_address,
        CAST(NULL AS STRING) AS store_city,
        CAST(NULL AS STRING) AS store_state,
        CAST(NULL AS STRING) AS store_zip,
        CAST(NULL AS STRING) AS store_country,
        CAST(NULL AS NUMERIC) AS store_latitude,
        CAST(NULL AS NUMERIC) AS store_longitude,
        CAST(NULL AS STRING) AS store,
        CAST(NULL AS STRING) AS store_number,
        CAST(NULL AS STRING) AS store_type,
        CAST(NULL AS STRING) AS store_manager,
        CAST(NULL AS STRING) AS store_phone,
        CAST(NULL AS STRING) AS store_status,
        CAST(NULL AS STRING) AS store_district_key,
        CAST(NULL AS STRING) AS store_district,
        CAST(NULL AS STRING) AS store_region_key,
        CAST(NULL AS STRING) AS store_region,
        date_key,
        2 AS source_priority
    FROM store_sales
    UNION ALL
    SELECT
        account_id,
        connector_id,
        CAST(NULL AS STRING) AS store_chain_key,
        CAST(NULL AS STRING) AS store_chain,
        store_key,
        CAST(NULL AS STRING) AS store_address,
        CAST(NULL AS STRING) AS store_city,
        CAST(NULL AS STRING) AS store_state,
        CAST(NULL AS STRING) AS store_zip,
        CAST(NULL AS STRING) AS store_country,
        CAST(NULL AS NUMERIC) AS store_latitude,
        CAST(NULL AS NUMERIC) AS store_longitude,
        CAST(NULL AS STRING) AS store,
        CAST(NULL AS STRING) AS store_number,
        CAST(NULL AS STRING) AS store_type,
        CAST(NULL AS STRING) AS store_manager,
        CAST(NULL AS STRING) AS store_phone,
        CAST(NULL AS STRING) AS store_status,
        CAST(NULL AS STRING) AS store_district_key,
        CAST(NULL AS STRING) AS store_district,
        CAST(NULL AS STRING) AS store_region_key,
        CAST(NULL AS STRING) AS store_region,
        date_key,
        3 AS source_priority
    FROM store_inventory
),

deduped AS (
    SELECT *
    FROM unioned
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            account_id,
            connector_id,
            store_key
        ORDER BY 
            source_priority,
            period_start DESC
    ) = 1
)

SELECT
    account_id,
    connector_id,
    store_chain_key,
    store_chain,
    store_key,
    store_address,
    store_city,
    store_state,
    store_zip,
    store_country,
    store_latitude,
    store_longitude,
    store,
    store_number,
    store_type,
    store_manager,
    store_phone,
    store_status,
    store_district_key,
    store_district,
    store_region_key,
    store_region,
    date_key,
    source_priority
FROM deduped 
