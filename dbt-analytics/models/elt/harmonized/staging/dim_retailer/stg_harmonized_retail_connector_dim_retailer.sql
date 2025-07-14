-- Harmonized staging for retail connector retailer dimension
-- Standardizes retailer metadata for cross-connector integration

WITH retailer_metadata AS (
    SELECT
        'retail_connector' AS retailer,
        'retail_connector' AS retailer_key,
        'Retail Connector' AS retailer_name,
        'Traditional Retail' AS retailer_type,
        'Standard retail operations with physical stores' AS retailer_description,
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