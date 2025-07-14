-- Source layer deduplication for retail connector store sales

{{
    config(
        materialized='incremental',
        unique_key=['account_id', 'connector_id', 'store_key', 'product_key', 'end_date'],
        incremental_strategy='merge',
        merge_exclude_columns=['ingestion_timestamp']
    )
}}

WITH direct AS (
    SELECT * FROM {{ ref('direct_retail_connector_store_sales') }}
    
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
            store_key,
            product_key,
            end_date
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
    store_key,
    product_key,
    upc,
    begin_date,
    end_date,
    sales_amount,
    sales_quantity
FROM deduplicated 
