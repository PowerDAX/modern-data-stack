-- Harmonized staging for retail connector product dimension
-- Standardizes schema for cross-connector integration

WITH source AS (
    SELECT * FROM {{ ref('denormalized_retail_connector_dim_product') }}
),

harmonized AS (
    SELECT
        'retail_connector' AS retailer,
        product_id,
        product_key,
        product,
        upc,
        pack_size,
        unit_of_measure,
        product_description,
        product_status,
        account_id,
        connector_id,
        
        -- Brand attributes
        product_brand_key,
        product_brand,
        
        -- Category attributes
        product_category_key,
        product_category,
        product_subcategory_key,
        product_subcategory,
        
        -- Supplier attributes
        product_supplier_key,
        product_supplier,
        
        -- Harmonized attributes
        COALESCE(product_brand, 'Unknown Brand') AS standardized_brand,
        COALESCE(product_category, 'Unknown Category') AS standardized_category,
        COALESCE(product_subcategory, 'Unknown Subcategory') AS standardized_subcategory,
        COALESCE(product_supplier, 'Unknown Supplier') AS standardized_supplier
    FROM source
)

SELECT * FROM harmonized 