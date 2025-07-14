-- Denormalized location dimension
-- Flattens normalized location, location_chain, location_district, and location_region dimensions

{{
    config(
        materialized='table'
    )
}}

WITH dim_location AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_location') }}
),

dim_location_chain AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_location_chain') }}
),

dim_location_district AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_location_district') }}
),

dim_location_region AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_location_region') }}
),

denormalized AS (
    SELECT
        -- location attributes
        dim_location.location_id,
        dim_location.location_key,
        dim_location.location,
        dim_location.location_number,
        dim_location.location_type,
        dim_location.location_address,
        dim_location.location_city,
        dim_location.location_state,
        dim_location.location_zip,
        dim_location.location_country,
        dim_location.location_latitude,
        dim_location.location_longitude,
        dim_location.location_manager,
        dim_location.location_phone,
        dim_location.location_status,

        -- account/config
        dim_location.account_id,
        dim_location.connector_id,

        -- location chain attributes
        dim_location_chain.location_chain_id,
        dim_location_chain.location_chain_key,
        dim_location_chain.location_chain,

        -- location district attributes
        dim_location_district.location_district_id,
        dim_location_district.location_district_key,
        dim_location_district.location_district,

        -- location region attributes
        dim_location_region.location_region_id,
        dim_location_region.location_region_key,
        dim_location_region.location_region
    FROM dim_location 
    INNER JOIN dim_location_chain USING (location_chain_id)
    INNER JOIN dim_location_district USING (location_district_id)
    INNER JOIN dim_location_region USING (location_region_id)
)

SELECT * FROM denormalized


