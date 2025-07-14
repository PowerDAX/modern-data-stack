-- Normalized location district dimension

{{
    config(
        materialized='table'
    )
}}

WITH location_district_data AS (
    SELECT * FROM {{ ref('stg_grocery_connector_dim_location_district') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_district_key']) }} AS location_district_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        location_district_key,
        location_district
    FROM location_district_data
)

SELECT
    location_district_id,
    account_id,
    connector_id,
    location_district_key,
    location_district
FROM final 