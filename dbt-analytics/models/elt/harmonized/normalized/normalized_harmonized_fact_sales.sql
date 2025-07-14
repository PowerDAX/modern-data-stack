-- Harmonized normalized fact sales
-- Combines staging models from all connectors

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

-- depends_on: {{ ref('stg_harmonized_retail_connector_fact_sales') }}
-- depends_on: {{ ref('stg_harmonized_grocery_connector_fact_sales') }}

WITH unioned AS (
    {{ dbt_utils.union_relations(
        relations=[
            ref('stg_harmonized_retail_connector_fact_sales'),
            ref('stg_harmonized_grocery_connector_fact_sales')
        ],
        include=[
            'retailer',
            'sales_id',
            'account_id',
            'connector_id',
            'store_id',
            'product_id',
            'date_key',
            'sales_quantity',
            'sales_amount',
            'avg_selling_price'
        ],
        column_override = {
            "date_key": "DATE",
            "sales_quantity": "NUMERIC",
            "sales_amount": "NUMERIC",
            "avg_selling_price": "NUMERIC"
        },
        where='CAST(date_key AS DATE) < current_date() AND EXTRACT(year FROM CAST(date_key AS DATE)) > EXTRACT(year FROM current_date()) - 10'
    ) }}
)

SELECT *,
    {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'retailer']) }} AS retailer_id
FROM unioned
WHERE retailer IS NOT NULL
