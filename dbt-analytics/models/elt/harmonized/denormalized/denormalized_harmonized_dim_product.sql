-- Harmonized denormalized dim product
-- Direct select from normalized harmonized model (simplified)

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

WITH harmonized_product AS (
    SELECT * FROM {{ ref('normalized_harmonized_dim_product') }}
)

SELECT * FROM harmonized_product 