-- Harmonized staging for grocery connector inventory
-- Standardizes schema for cross-connector integration

WITH source AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_fact_inventory_location') }}
),

harmonized AS (
    SELECT
        'grocery_connector' AS retailer,
        inventory_id,
        location_id AS store_id,
        item_id AS product_id,
        account_id,
        connector_id,
        date_key,
        
        -- Standardized fact columns
        on_hand_units AS on_hand_quantity,
        on_hand_dollars AS on_hand_amount,
        on_order_units AS on_order_quantity,
        on_order_dollars AS on_order_amount,
        
        -- Calculated harmonized metrics
        SAFE_DIVIDE(on_hand_dollars, on_hand_units) AS avg_unit_cost,
        SAFE_DIVIDE(on_order_dollars, on_order_units) AS avg_order_unit_cost,
        SAFE_ADD(on_hand_units, on_order_units) AS total_available_quantity,
        SAFE_ADD(on_hand_dollars, on_order_dollars) AS total_inventory_value
    FROM source
)

SELECT * FROM harmonized 