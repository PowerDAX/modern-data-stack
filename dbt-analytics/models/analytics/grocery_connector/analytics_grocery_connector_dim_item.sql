-- Analytics model for grocery connector item dimension
-- Direct select from denormalized item dimension

{{
    config(
        materialized='table'
    )
}}

WITH denormalized_item AS (
    SELECT * FROM {{ ref('denormalized_grocery_connector_dim_item') }}
)

SELECT * FROM denormalized_item 