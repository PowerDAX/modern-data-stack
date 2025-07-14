-- Source layer deduplication for grocery connector item master

{{
    config(
        materialized='table'
    )
}}

WITH direct AS (
    SELECT * FROM {{ ref('direct_grocery_connector_item_master') }}
),

deduplicated AS (
    SELECT *
    FROM direct
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY 
            account_id,
            connector_id,
            item_key
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
    item_key,
    item_name,
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
    item_description,
    item_status
FROM deduplicated
