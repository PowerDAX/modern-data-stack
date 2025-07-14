-- Harmonized staging for retail connector inventory
-- Standardizes schema for cross-connector integration

WITH source AS (
    SELECT * FROM {{ ref('normalized_retail_connector_fact_inventory_store') }}
),

harmonized AS (
    SELECT
        'retail_connector' AS retailer,
        inventory_id,
        store_id,
        product_id,
        account_id,
        connector_id,
        date_key,
        
        -- Standardized fact columns
        on_hand_quantity,
        on_hand_amount,
        on_order_quantity,
        on_order_amount,
        
        -- Calculated harmonized metrics
        SAFE_DIVIDE(on_hand_amount, on_hand_quantity) AS avg_unit_cost,
        SAFE_DIVIDE(on_order_amount, on_order_quantity) AS avg_order_unit_cost,
        SAFE_ADD(on_hand_quantity, on_order_quantity) AS total_available_quantity,
        SAFE_ADD(on_hand_amount, on_order_amount) AS total_inventory_value
    FROM source
)

SELECT * FROM harmonized 