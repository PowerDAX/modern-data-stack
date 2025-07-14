-- Analytics model for grocery connector sales
-- Direct select from normalized fact and dimensions

{{
    config(
        materialized='table'
    )
}}

WITH normalized_sales AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_fact_sales') }}
)

SELECT * FROM normalized_sales 
