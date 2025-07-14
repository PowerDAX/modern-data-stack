-- Analytics model for harmonized product dimension
-- Direct select from denormalized harmonized model

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

WITH denormalized_harmonized_product AS (
    SELECT * FROM {{ ref('denormalized_harmonized_dim_product') }}
)

SELECT * FROM denormalized_harmonized_product 