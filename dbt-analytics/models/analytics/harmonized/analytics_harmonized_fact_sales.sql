-- Analytics model for harmonized sales
-- Direct select from normalized harmonized model

{{
    config(
        materialized='table',
        cluster_by=["retailer_id"]
    )
}}

WITH normalized_harmonized_sales AS (
    SELECT * FROM {{ ref('normalized_harmonized_fact_sales') }}
)

SELECT * FROM normalized_harmonized_sales 
