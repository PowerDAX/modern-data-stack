-- Source layer deduplication for grocery connector location inventory

{{
    config(
        materialized='incremental',
        unique_key=['account_id', 'connector_id', 'location_key', 'item_key', 'report_date'],
        incremental_strategy='merge',
        merge_exclude_columns=['ingestion_timestamp']
    )
}}

WITH direct AS (
    SELECT * FROM {{ ref('direct_grocery_connector_location_inventory') }}
    
    {% if is_incremental() %}
        -- Only process records newer than the latest ingestion timestamp in the target table
        WHERE ingestion_timestamp > (
            SELECT COALESCE(MAX(ingestion_timestamp), '1900-01-01'::timestamp) 
            FROM {{ this }}
        )
    {% endif %}
),

deduplicated AS (
    SELECT *
    FROM direct
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY 
            account_id,
            connector_id,
            location_key,
            item_key,
            report_date
        ORDER BY 
            report_date DESC,
            ingestion_timestamp DESC
    ) = 1
)

SELECT
    account_id,
    connector_id,
    ingestion_timestamp,
    report_date,
    location_key,
    item_key,
    upc,
    on_hand_units,
    on_hand_dollars,
    on_order_units,
    on_order_dollars,
    allocated_units,
    available_units
FROM deduplicated
