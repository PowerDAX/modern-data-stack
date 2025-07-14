-- Analytics model for retail connector product dimension
-- Direct select from denormalized product dimension

{{
    config(
        materialized='table'
    )
}}

WITH denormalized_product AS (
    SELECT * FROM {{ ref('denormalized_retail_connector_dim_product') }}
)

SELECT * FROM denormalized_product 