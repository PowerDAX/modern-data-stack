{{
    config(
        materialized='table'
    )
}}

-- depends_on: {{ ref('source_retail_connector_store_sales') }}

WITH cleaned_source AS (
    {{ select_clean_from_metadata(ref('source_retail_connector_store_sales')) }}
)

SELECT
    account_id,
    connector_id,
    store_key,
    product_key,
    upc,
    end_date AS date_key,
    sales_amount,
    sales_quantity
FROM cleaned_source 
