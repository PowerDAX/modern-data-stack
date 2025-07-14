-- Analytics model for retail connector store dimension
-- Direct select from denormalized store dimension

{{
    config(
        materialized='table'
    )
}}

WITH denormalized_store AS (
    SELECT * FROM {{ ref('denormalized_retail_connector_dim_store') }}
)

SELECT * FROM denormalized_store 