{{
    config(
        materialized='table'
    )
}}

WITH item_master AS (
    SELECT * FROM {{ ref('clean_grocery_connector_item_master') }}
),

location_sales AS (
    SELECT * FROM {{ ref('clean_grocery_connector_location_sales') }}
),

location_inventory AS (
    SELECT * FROM {{ ref('clean_grocery_connector_location_inventory') }}
),

unioned AS (
    SELECT
        account_id,
        connector_id,
        item_key,
        item_category_key,
        item_category,
        item_subcategory_key,
        item_subcategory,
        item_supplier_key,
        item_supplier,
        item,
        item_brand_key,
        item_brand,
        upc,
        pack_size,
        unit_of_measure,
        item_description,
        item_status,
        1 AS source_priority,
        date_key
    FROM item_master
    UNION ALL
    SELECT
        account_id,
        connector_id,
        item_key,
        CAST(NULL AS STRING) AS item_category_key,
        CAST(NULL AS STRING) AS item_category,
        CAST(NULL AS STRING) AS item_subcategory_key,
        CAST(NULL AS STRING) AS item_subcategory,
        CAST(NULL AS STRING) AS item_supplier_key,
        CAST(NULL AS STRING) AS item_supplier,
        CAST(NULL AS STRING) AS item,
        CAST(NULL AS STRING) AS item_brand_key,
        CAST(NULL AS STRING) AS item_brand,
        upc,
        CAST(NULL AS STRING) AS pack_size,
        CAST(NULL AS STRING) AS unit_of_measure,
        CAST(NULL AS STRING) AS item_description,
        CAST(NULL AS STRING) AS item_status,
        2 AS source_priority,
        date_key
    FROM location_sales
    UNION ALL
    SELECT
        account_id,
        connector_id,
        item_key,
        CAST(NULL AS STRING) AS item_category_key,
        CAST(NULL AS STRING) AS item_category,
        CAST(NULL AS STRING) AS item_subcategory_key,
        CAST(NULL AS STRING) AS item_subcategory,
        CAST(NULL AS STRING) AS item_supplier_key,
        CAST(NULL AS STRING) AS item_supplier,
        CAST(NULL AS STRING) AS item,
        CAST(NULL AS STRING) AS item_brand_key,
        CAST(NULL AS STRING) AS item_brand,
        upc,
        CAST(NULL AS STRING) AS pack_size,
        CAST(NULL AS STRING) AS unit_of_measure,
        CAST(NULL AS STRING) AS item_description,
        CAST(NULL AS STRING) AS item_status,
        3 AS source_priority,
        date_key
    FROM location_inventory
),

deduped AS (
    SELECT *
    FROM unioned
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            account_id,
            connector_id,
            item_key
        ORDER BY
            source_priority,
            date_key DESC
    ) = 1
)

SELECT
    account_id,
    connector_id,
    item_key,
    item_category_key,
    item_category,
    item_subcategory_key,
    item_subcategory,
    item_supplier_key,
    item_supplier,
    item,
    item_brand_key,
    item_brand,
    upc,
    pack_size,
    unit_of_measure,
    item_description,
    item_status,
    source_priority,
    date_key
FROM deduped 


