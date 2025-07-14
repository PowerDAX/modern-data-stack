{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_grocery_connector_location_sales') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_grocery_connector_location_sales')) }}
)

SELECT
    account_id,
    connector_id,
    location_key,
    item_key,
    upc,
    end_date AS date_key,
    sales_dollars,
    sales_units,
    regular_sales_dollars,
    regular_sales_units,
    promotional_sales_dollars,
    promotional_sales_units
FROM cleaned_source 
