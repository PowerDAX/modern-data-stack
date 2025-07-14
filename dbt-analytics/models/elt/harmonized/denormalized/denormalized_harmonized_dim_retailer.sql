-- Harmonized denormalized dim retailer
-- Direct select from normalized harmonized model (simplified)

{{
    config(
        materialized='table'
    )
}}

WITH harmonized_retailer AS (
    SELECT * FROM {{ ref('normalized_harmonized_dim_retailer') }}
)

SELECT * FROM harmonized_retailer 