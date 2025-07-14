{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_retail_connector_store_master') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_retail_connector_store_master')) }}
)

SELECT
    account_id,
    connector_id,
    DATE(report_date) AS date_key,
    chain_key AS store_chain_key,
    chain AS store_chain,
    store_key,
    store_name AS store,
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
    district_key AS store_district_key,
    district AS store_district,
    region_key AS store_region_key,
    region AS store_region
FROM cleaned_source
