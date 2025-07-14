-- Harmonized normalized fact inventory store
-- Combines staging models from all connectors

{{
    config(
        materialized='table',
        cluster_by=["retailer"]
    )
}}

-- depends_on: {{ ref('stg_harmonized_retail_connector_fact_inventory_store') }}
-- depends_on: {{ ref('stg_harmonized_grocery_connector_fact_inventory_store') }}

WITH unioned AS (
    {{ dbt_utils.union_relations(
        relations=[
            ref('stg_harmonized_retail_connector_fact_inventory_store'),
            ref('stg_harmonized_grocery_connector_fact_inventory_store')
        ],
        include=[
            'retailer',
            'inventory_id',
            'account_id',
            'connector_id',
            'store_id',
            'product_id',
            'date_key',
            'on_hand_quantity',
            'on_hand_amount',
            'on_order_quantity',
            'on_order_amount',
            'avg_unit_cost',
            'avg_order_unit_cost',
            'total_available_quantity',
            'total_inventory_value'
        ],
        column_override = {
            "date_key": "DATE",
            "on_hand_quantity": "NUMERIC",
            "on_hand_amount": "NUMERIC",
            "on_order_quantity": "NUMERIC",
            "on_order_amount": "NUMERIC",
            "avg_unit_cost": "NUMERIC",
            "avg_order_unit_cost": "NUMERIC",
            "total_available_quantity": "NUMERIC",
            "total_inventory_value": "NUMERIC"
        },
        where='CAST(date_key AS DATE) < current_date() AND EXTRACT(year FROM CAST(date_key AS DATE)) > EXTRACT(year FROM current_date()) - 10'
    ) }}
)

SELECT *,
    {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'retailer']) }} AS retailer_id
FROM unioned
WHERE retailer IS NOT NULL 