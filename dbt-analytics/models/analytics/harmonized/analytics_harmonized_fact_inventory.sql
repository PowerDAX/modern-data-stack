-- Analytics model for harmonized inventory
-- Direct select from normalized harmonized model

{{
    config(
        materialized='table',
        cluster_by=["retailer_id"]
    )
}}

WITH normalized_harmonized_inventory AS (
    SELECT * FROM {{ ref('normalized_harmonized_fact_inventory_store') }}
)

SELECT * FROM normalized_harmonized_inventory 