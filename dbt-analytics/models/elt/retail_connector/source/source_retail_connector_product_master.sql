-- Source layer deduplication for retail connector product master

{{
    config(
        materialized='table'
    )
}}

WITH direct AS (
    SELECT * FROM {{ ref('direct_retail_connector_product_master') }}
),

deduplicated AS (
    SELECT *
    FROM direct
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY 
            account_id,
            connector_id,
            product_key
        ORDER BY 
            report_date DESC,
            ingestion_timestamp DESC
    ) = 1
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    product_key,
    product_name,
    brand_key,
    brand,
    category_key,
    category,
    subcategory_key,
    subcategory,
    supplier_key,
    supplier_name,
    upc,
    pack_size,
    unit_of_measure,
    product_description,
    product_status
FROM deduplicated 
