{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_retail_connector_store_inventory') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_retail_connector_store_inventory')) }}
)

SELECT
    account_id,
    connector_id,
    store_key,
    product_key,
    upc,
    report_date AS date_key,
    on_hand_quantity,
    on_hand_amount,
    on_order_quantity,
    on_order_amount
FROM cleaned_source 
