-- Harmonized staging for grocery connector retailer dimension
-- Standardizes retailer metadata for cross-connector integration

WITH retailer_metadata AS (
    SELECT
        'grocery_connector' AS retailer,
        'grocery_connector' AS retailer_key,
        'Grocery Connector' AS retailer_name,
        'Grocery Retail' AS retailer_type,
        'Grocery retail operations with food and consumer goods' AS retailer_description,
        'Active' AS retailer_status,
        'Physical' AS channel_type,
        'B2C' AS business_model,
        'North America' AS primary_region,
        'USD' AS primary_currency
),

harmonized AS (
    SELECT
        retailer,
        retailer_key,
        retailer_name,
        retailer_type,
        retailer_description,
        retailer_status,
        channel_type,
        business_model,
        primary_region,
        primary_currency
    FROM retailer_metadata
)

SELECT * FROM harmonized 