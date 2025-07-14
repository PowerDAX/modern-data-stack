-- Harmonized normalized dim store
-- Combines staging models from all connectors

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

-- depends_on: {{ ref('stg_harmonized_retail_connector_dim_store') }}
-- depends_on: {{ ref('stg_harmonized_grocery_connector_dim_store') }}

WITH unioned AS (
    {{ dbt_utils.union_relations(
        relations=[
            ref('stg_harmonized_retail_connector_dim_store'),
            ref('stg_harmonized_grocery_connector_dim_store')
        ],
        include=[
            'retailer',
            'store_id',
            'store_key',
            'store',
            'store_number',
            'store_type',
            'store_address',
            'store_city',
            'store_state',
            'store_zip',
            'store_country',
            'store_latitude',
            'store_longitude',
            'store_manager',
            'store_phone',
            'store_status',
            'account_id',
            'connector_id',
            'store_chain_id',
            'store_chain_key',
            'store_chain',
            'store_district_id',
            'store_district_key',
            'store_district',
            'store_region_id',
            'store_region_key',
            'store_region',
            'standardized_chain',
            'standardized_district',
            'standardized_region',
            'standardized_store_type',
            'standardized_state',
            'standardized_country'
        ],
        column_override = {
            "store_latitude": "NUMERIC",
            "store_longitude": "NUMERIC"
        }
    ) }}
)

SELECT *,
    {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'retailer']) }} AS retailer_id
FROM unioned
WHERE retailer IS NOT NULL