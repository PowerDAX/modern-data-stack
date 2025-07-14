-- Harmonized denormalized fact inventory store
-- Direct select from normalized harmonized model (simplified)

{{
    config(
        materialized='table',
        cluster_by=["retailer_id"]
    )
}}

WITH harmonized_inventory AS (
    SELECT * FROM {{ ref('normalized_harmonized_fact_inventory_store') }}
)

SELECT * FROM harmonized_inventory 