-- Normalized inventory fact table
-- References dim_location and dim_item

{{
    config(
        materialized='table',
        post_hook=[
            "{{ set_table_labels() }}",
            "{{ audit_model_execution() }}"
        ],
        tags=['critical', 'fact_table']
    )
}}

WITH inventory_source AS (
    SELECT * FROM {{ ref('clean_grocery_connector_location_inventory') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_key', 'item_key', 'date_key']) }} AS inventory_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'location_key']) }} AS location_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'item_key']) }} AS item_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        date_key,
        
        -- facts
        on_hand_units,
        on_hand_dollars,
        on_order_units,
        on_order_dollars,
        allocated_units,
        available_units
    FROM inventory_source
)

SELECT
    inventory_id,
    location_id,
    item_id,
    account_id,
    connector_id,
    date_key,
    on_hand_units,
    on_hand_dollars,
    on_order_units,
    on_order_dollars,
    allocated_units,
    available_units
FROM final 