{{
    config(
        materialized='table'
    )
}}

WITH product_master AS (
    SELECT * FROM {{ ref('clean_retail_connector_product_master') }}
),

store_sales AS (
    SELECT * FROM {{ ref('clean_retail_connector_store_sales') }}
),

store_inventory AS (
    SELECT * FROM {{ ref('clean_retail_connector_store_inventory') }}
),

unioned AS (
    SELECT
        account_id,
        connector_id,
        product_key,
        product_category_key,
        product_category,
        product_subcategory_key,
        product_subcategory,
        product_supplier_key,
        product_supplier,
        product,
        product_brand_key,
        product_brand,
        upc,
        pack_size,
        unit_of_measure,
        product_description,
        product_status,
        1 AS source_priority,
        date_key
    FROM product_master
    UNION ALL
    SELECT
        account_id,
        connector_id,
        product_key,
        CAST(NULL AS STRING) AS product_category_key,
        CAST(NULL AS STRING) AS product_category,
        CAST(NULL AS STRING) AS product_subcategory_key,
        CAST(NULL AS STRING) AS product_subcategory,
        CAST(NULL AS STRING) AS product_supplier_key,
        CAST(NULL AS STRING) AS product_supplier,
        CAST(NULL AS STRING) AS product,
        CAST(NULL AS STRING) AS product_brand_key,
        CAST(NULL AS STRING) AS product_brand,
        upc,
        CAST(NULL AS STRING) AS pack_size,
        CAST(NULL AS STRING) AS unit_of_measure,
        CAST(NULL AS STRING) AS product_description,
        CAST(NULL AS STRING) AS product_status,
        2 AS source_priority,
        date_key
    FROM store_sales
    UNION ALL
    SELECT
        account_id,
        connector_id,
        product_key,
        CAST(NULL AS STRING) AS product_category_key,
        CAST(NULL AS STRING) AS product_category,
        CAST(NULL AS STRING) AS product_subcategory_key,
        CAST(NULL AS STRING) AS product_subcategory,
        CAST(NULL AS STRING) AS product_supplier_key,
        CAST(NULL AS STRING) AS product_supplier,
        CAST(NULL AS STRING) AS product,
        CAST(NULL AS STRING) AS product_brand_key,
        CAST(NULL AS STRING) AS product_brand,
        upc,
        CAST(NULL AS STRING) AS pack_size,
        CAST(NULL AS STRING) AS unit_of_measure,
        CAST(NULL AS STRING) AS product_description,
        CAST(NULL AS STRING) AS product_status,
        3 AS source_priority,
        date_key
    FROM store_inventory
),

deduped AS (
    SELECT *
    FROM unioned
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            account_id,
            connector_id,
            product_key
        ORDER BY
            source_priority,
            date_key DESC
    ) = 1
)

SELECT
    account_id,
    connector_id,
    product_key,
    product_category_key,
    product_category,
    product_subcategory_key,
    product_subcategory,
    product_supplier_key,
    product_supplier,
    product,
    product_brand_key,
    product_brand,
    upc,
    pack_size,
    unit_of_measure,
    product_description,
    product_status,
    source_priority,
    date_key
FROM deduped 
