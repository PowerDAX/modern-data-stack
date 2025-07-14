-- Analytics model for grocery connector location dimension
-- Direct select from denormalized location dimension

{{
    config(
        materialized='table'
    )
}}

WITH denormalized_location AS (
    SELECT * FROM {{ ref('denormalized_grocery_connector_dim_location') }}
)

SELECT * FROM denormalized_location 