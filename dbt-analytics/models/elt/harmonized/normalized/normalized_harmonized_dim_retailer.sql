-- Harmonized normalized dim retailer
-- Combines staging models from all connectors

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

-- depends_on: {{ ref('stg_harmonized_retail_connector_dim_retailer') }}
-- depends_on: {{ ref('stg_harmonized_grocery_connector_dim_retailer') }}

WITH unioned AS (
    {{ dbt_utils.union_relations(
        relations=[
            ref('stg_harmonized_retail_connector_dim_retailer'),
            ref('stg_harmonized_grocery_connector_dim_retailer')
        ],
        include=[
            'retailer',
            'retailer_key',
            'retailer_name',
            'retailer_type',
            'retailer_description',
            'retailer_status',
            'channel_type',
            'business_model',
            'primary_region',
            'primary_currency',
            'standardized_retailer_code',
            'standardized_retailer_name',
            'standardized_retailer_type',
            'standardized_channel_type'
        ]
    ) }}
)

SELECT *,
    {{ dbt_utils.surrogate_key(['retailer']) }} AS harmonized_retailer_id
FROM unioned
WHERE retailer IS NOT NULL 