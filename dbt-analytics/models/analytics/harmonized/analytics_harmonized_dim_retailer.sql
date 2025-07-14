-- Analytics model for harmonized retailer dimension
-- Direct select from denormalized harmonized model

{{
    config(
        materialized='table'
    )
}}

WITH denormalized_harmonized_retailer AS (
    SELECT * FROM {{ ref('denormalized_harmonized_dim_retailer') }}
)

SELECT * FROM denormalized_harmonized_retailer 