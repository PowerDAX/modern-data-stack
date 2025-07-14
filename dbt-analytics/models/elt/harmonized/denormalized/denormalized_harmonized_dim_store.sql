-- Harmonized denormalized dim store
-- Direct select from normalized harmonized model (simplified)

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

WITH harmonized_store AS (
    SELECT * FROM {{ ref('normalized_harmonized_dim_store') }}
)

SELECT * FROM harmonized_store 