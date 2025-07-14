-- Normalized location region dimension

{{
    config(
        materialized='table'
    )
}}

WITH location_region_data AS (
    SELECT * FROM {{ ref('stg_grocery_connector_dim_location_region') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_region_key']) }} AS location_region_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        location_region_key,
        location_region
    FROM location_region_data
)

SELECT
    location_region_id,
    account_id,
    connector_id,
    location_region_key,
    location_region
FROM final 