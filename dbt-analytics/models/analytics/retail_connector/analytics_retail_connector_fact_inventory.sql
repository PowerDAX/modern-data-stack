-- Analytics model for retail connector inventory
-- Direct select from normalized fact and dimensions

{{
    config(
        materialized='table'
    )
}}

WITH normalized_inventory AS (
    SELECT * FROM {{ ref('normalized_retail_connector_fact_inventory_store') }}
)

SELECT * FROM normalized_inventory 