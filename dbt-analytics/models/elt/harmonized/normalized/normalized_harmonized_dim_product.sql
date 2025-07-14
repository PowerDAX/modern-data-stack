-- Harmonized normalized dim product
-- Combines staging models from all connectors

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

-- depends_on: {{ ref('stg_harmonized_retail_connector_dim_product') }}
-- depends_on: {{ ref('stg_harmonized_grocery_connector_dim_product') }}

WITH unioned AS (
    {{ dbt_utils.union_relations(
        relations=[
            ref('stg_harmonized_retail_connector_dim_product'),
            ref('stg_harmonized_grocery_connector_dim_product')
        ],
        include=[
            'retailer',
            'product_id',
            'product_key',
            'product',
            'upc',
            'pack_size',
            'unit_of_measure',
            'product_description',
            'product_status',
            'account_id',
            'connector_id',
            'product_brand_id',
            'product_brand_key',
            'product_brand',
            'product_category_id',
            'product_category_key',
            'product_category',
            'product_subcategory_id',
            'product_subcategory_key',
            'product_subcategory',
            'product_supplier_id',
            'product_supplier_key',
            'product_supplier',
            'standardized_brand',
            'standardized_category',
            'standardized_subcategory',
            'standardized_supplier'
        ]
    ) }}
)

SELECT *,
    {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'retailer']) }} AS retailer_id
FROM unioned
WHERE retailer IS NOT NULL