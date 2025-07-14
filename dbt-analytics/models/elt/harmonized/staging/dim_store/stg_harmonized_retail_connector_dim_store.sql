-- Harmonized staging for retail connector store dimension
-- Standardizes schema for cross-connector integration

WITH source AS (
    SELECT * FROM {{ ref('denormalized_retail_connector_dim_store') }}
),

harmonized AS (
    SELECT
        'retail_connector' AS retailer,
        store_id,
        store_key,
        store,
        store_number,
        store_type,
        store_address,
        store_city,
        store_state,
        store_zip,
        store_country,
        store_latitude,
        store_longitude,
        store_manager,
        store_phone,
        store_status,
        account_id,
        connector_id,
        
        -- Chain attributes
        store_chain_key,
        store_chain,
        
        -- District attributes
        store_district_key,
        store_district,
        
        -- Region attributes
        store_region_key,
        store_region,
        
        -- Harmonized attributes
        COALESCE(store_chain, 'Unknown Chain') AS standardized_chain,
        COALESCE(store_district, 'Unknown District') AS standardized_district,
        COALESCE(store_region, 'Unknown Region') AS standardized_region,
        COALESCE(store_type, 'Unknown Type') AS standardized_store_type,
        COALESCE(store_state, 'Unknown State') AS standardized_state,
        COALESCE(store_country, 'Unknown Country') AS standardized_country
    FROM source
)

SELECT * FROM harmonized 