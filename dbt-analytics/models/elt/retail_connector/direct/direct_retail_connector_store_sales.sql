-- Direct layer extraction for retail connector store sales

WITH raw_source AS (
    SELECT * FROM {{ source('connector_data', 'retail_connector_store_sales_raw') }}
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
    CAST(sales_amount AS NUMERIC) AS sales_amount,
    CAST(sales_quantity AS INTEGER) AS sales_quantity
FROM raw_source
WHERE account_id IS NOT NULL
    AND connector_id IS NOT NULL
    AND store_key IS NOT NULL
    AND product_key IS NOT NULL 
