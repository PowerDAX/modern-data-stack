-- Denormalized item dimension
-- Flattens normalized item, item_brand, item_subcategory, item_category, and supplier dimensions

{{
    config(
        materialized='table'
    )
}}

WITH dim_item AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_item') }}
),

dim_item_brand AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_item_brand') }}
),

dim_item_subcategory AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_item_subcategory') }}
),

dim_item_category AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_item_category') }}
),

dim_item_supplier AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_dim_item_supplier') }}
),

denormalized AS (
    SELECT
        -- item attributes
        dim_item.item_id,
        dim_item.item_key,
        dim_item.item,
        dim_item.upc,
        dim_item.pack_size,
        dim_item.unit_of_measure,
        dim_item.item_description,
        dim_item.item_status,

        -- account/config
        dim_item.account_id,
        dim_item.connector_id,

        -- item brand attributes
        dim_item_brand.item_brand_id,
        dim_item_brand.item_brand_key,
        dim_item_brand.item_brand,

        -- item subcategory attributes
        dim_item_subcategory.item_subcategory_id,
        dim_item_subcategory.item_subcategory_key,
        dim_item_subcategory.item_subcategory,

        -- item category attributes
        dim_item_category.item_category_id,
        dim_item_category.item_category_key,
        dim_item_category.item_category,
        
        -- supplier attributes
        dim_item_supplier.item_supplier_id,
        dim_item_supplier.item_supplier_key,
        dim_item_supplier.item_supplier
    FROM dim_item
    INNER JOIN dim_item_brand USING (item_brand_id)
    INNER JOIN dim_item_subcategory USING (item_subcategory_id)
    INNER JOIN dim_item_category USING (item_category_id)
    INNER JOIN dim_item_supplier USING (item_supplier_id)
)

SELECT * FROM denormalized 


