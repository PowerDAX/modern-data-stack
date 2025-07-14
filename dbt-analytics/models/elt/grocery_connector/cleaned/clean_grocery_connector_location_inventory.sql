{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_grocery_connector_location_inventory') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_grocery_connector_location_inventory')) }}
)

SELECT
    account_id,
    connector_id,
    location_key,
    item_key,
    upc,
    report_date AS date_key,
    on_hand_units,
    on_hand_dollars,
    on_order_units,
    on_order_dollars,
    allocated_units,
    available_units
FROM cleaned_source 
