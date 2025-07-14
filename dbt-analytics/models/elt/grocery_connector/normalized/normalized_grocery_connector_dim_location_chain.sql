-- Normalized location chain dimension

{{
    config(
        materialized='table'
    )
}}

WITH location_chain_data AS (
    SELECT * FROM {{ ref('stg_grocery_connector_dim_location_chain') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_chain_key']) }} AS location_chain_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        location_chain_key,
        location_chain
    FROM location_chain_data
)

SELECT
    location_chain_id,
    account_id,
    connector_id,
    location_chain_key,
    location_chain
FROM final 