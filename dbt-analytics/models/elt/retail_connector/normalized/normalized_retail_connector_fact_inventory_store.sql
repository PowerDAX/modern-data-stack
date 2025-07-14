-- Normalized store inventory fact table
-- References dim_store and dim_product

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
    SELECT * FROM {{ ref('clean_retail_connector_store_inventory') }}
),

final AS (
    SELECT
        -- primary key
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_id', 'product_id', 'date_key']) }} AS inventory_id,
        
        -- foreign keys
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'store_id']) }} AS store_id,
        {{ dbt_utils.surrogate_key(['account_id', 'connector_id', 'product_id']) }} AS product_id,
        
        -- account/config
        account_id,
        connector_id,
        
        -- logical info
        date_key,
        
        -- facts
        on_hand_quantity,
        on_hand_amount,
        on_order_quantity,
        on_order_amount
    FROM inventory_source
)

SELECT
    inventory_id,
    store_id,
    product_id,
    account_id,
    connector_id,
    date_key,
    on_hand_quantity,
    on_hand_amount,
    on_order_quantity,
    on_order_amount
FROM final 
