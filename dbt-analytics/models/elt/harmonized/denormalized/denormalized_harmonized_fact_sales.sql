-- Harmonized denormalized fact sales
-- Direct select from normalized harmonized model (simplified)

{{
    config(
        materialized='table',
        cluster_by=["retailer_id"]
    )
}}

WITH harmonized_sales AS (
    SELECT * FROM {{ ref('normalized_harmonized_fact_sales') }}
)

SELECT * FROM harmonized_sales 


