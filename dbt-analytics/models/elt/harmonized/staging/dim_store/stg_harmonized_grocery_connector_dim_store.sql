-- Harmonized staging for grocery connector store dimension
-- Standardizes schema for cross-connector integration

WITH source AS (
    SELECT * FROM {{ ref('denormalized_grocery_connector_dim_location') }}
),

harmonized AS (
    SELECT
        'grocery_connector' AS retailer,
        location_id AS store_id,
        location_key AS store_key,
        location_name AS store,
        location_number AS store_number,
        location_type AS store_type,
        location_address AS store_address,
        location_city AS store_city,
        location_state AS store_state,
        location_zip AS store_zip,
        location_country AS store_country,
        location_latitude AS store_latitude,
        location_longitude AS store_longitude,
        location_manager AS store_manager,
        location_phone AS store_phone,
        location_status AS store_status,
        account_id,
        connector_id,
        
        -- Chain attributes
        location_chain_key AS store_chain_key,
        location_chain AS store_chain,
        
        -- District attributes
        location_district_key AS store_district_key,
        location_district AS store_district,
        
        -- Region attributes
        location_region_key AS store_region_key,
        location_region AS store_region,
        
        -- Harmonized attributes
        COALESCE(location_chain, 'Unknown Chain') AS standardized_chain,
        COALESCE(location_district, 'Unknown District') AS standardized_district,
        COALESCE(location_region, 'Unknown Region') AS standardized_region,
        COALESCE(location_type, 'Unknown Type') AS standardized_store_type,
        COALESCE(location_state, 'Unknown State') AS standardized_state,
        COALESCE(location_country, 'Unknown Country') AS standardized_country
    FROM source
)

SELECT * FROM harmonized 