-- Analytics model for harmonized store dimension
-- Direct select from denormalized harmonized model

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

WITH denormalized_harmonized_store AS (
    SELECT * FROM {{ ref('denormalized_harmonized_dim_store') }}
)

SELECT * FROM denormalized_harmonized_store 