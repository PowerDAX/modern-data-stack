{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_grocery_connector_location_master') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_grocery_connector_location_master')) }}
)

SELECT
    account_id,
    connector_id,
    DATE(report_date) AS date_key,
    chain_key AS location_chain_key,
    chain AS location_chain,
    location_key,
    location_name AS location,
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
    district_key AS location_district_key,
    district AS location_district,
    region_key AS location_region_key,
    region AS location_region
FROM cleaned_source 
