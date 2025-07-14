-- Normalized location dimension

{{
    config(
        materialized='table'
    )
}}

WITH location_attributes AS (
    SELECT * FROM {{ ref('stg_grocery_connector_location_attributes') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_key']) }} AS location_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_chain_key']) }} AS location_chain_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_district_key']) }} AS location_district_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        location_key,
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
        location_status
    FROM location_attributes
)

SELECT
    location_id,
    location_chain_id,
    location_district_id,
    account_id,
    connector_id,
    location_key,
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
    location_status
FROM final 


