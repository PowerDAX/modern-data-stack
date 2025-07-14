-- Harmonized staging for grocery connector product dimension
-- Standardizes schema for cross-connector integration

WITH source AS (
    SELECT * FROM {{ ref('denormalized_grocery_connector_dim_item') }}
),

harmonized AS (
    SELECT
        'grocery_connector' AS retailer,
        item_id AS product_id,
        item_key AS product_key,
        item AS product,
        upc,
        pack_size,
        unit_of_measure,
        item_description AS product_description,
        item_status AS product_status,
        account_id,
        connector_id,
        
        -- Brand attributes
        item_brand_key AS product_brand_key,
        item_brand AS product_brand,
        
        -- Category attributes  
        item_category_key AS product_category_key,
        item_category AS product_category,
        item_subcategory_key AS product_subcategory_key,
        item_subcategory AS product_subcategory,
        
        -- Supplier attributes
        item_supplier_key AS product_supplier_key,
        item_supplier AS product_supplier,
        
        -- Harmonized attributes
        COALESCE(item_brand, 'Unknown Brand') AS standardized_brand,
        COALESCE(item_category, 'Unknown Category') AS standardized_category,
        COALESCE(item_subcategory, 'Unknown Subcategory') AS standardized_subcategory,
        COALESCE(item_supplier, 'Unknown Supplier') AS standardized_supplier
    FROM source
)

SELECT * FROM harmonized 