-- Denormalized inventory fact table
-- Joins normalized fact_inventory_location with denormalized dimensions

WITH fact_inventory AS (
    SELECT * FROM {{ ref('normalized_grocery_connector_fact_inventory_location') }}
),

dim_item AS (
    SELECT * FROM {{ ref('denormalized_grocery_connector_dim_item') }}
),

dim_location AS (
    SELECT * FROM {{ ref('denormalized_grocery_connector_dim_location') }}
),

denormalized AS (
    SELECT
        -- primary key
        fact_inventory.inventory_id,
        
        -- account/config
        fact_inventory.account_id,
        fact_inventory.connector_id,
        
        -- item attributes
        dim_item.item_id,
        dim_item.item_key,
        dim_item.item,
        dim_item.item_brand_id,
        dim_item.item_brand_key,
        dim_item.item_brand,
        dim_item.item_subcategory_id,
        dim_item.item_subcategory_key,
        dim_item.item_subcategory,
        dim_item.item_category_id,
        dim_item.item_category_key,
        dim_item.item_category,
        dim_item.item_supplier_id,
        dim_item.item_supplier_key,
        dim_item.item_supplier,
        dim_item.upc,
        dim_item.pack_size,
        dim_item.unit_of_measure,
        dim_item.item_description,
        dim_item.item_status,
        
        -- location attributes
        dim_location.location_id,
        dim_location.location,
        dim_location.location_number,
        dim_location.location_type,
        dim_location.location_address,
        dim_location.location_city,
        dim_location.location_state,
        dim_location.location_zip,
        dim_location.location_country,
        dim_location.location_latitude,
        dim_location.location_longitude,
        dim_location.location_manager,
        dim_location.location_phone,
        dim_location.location_status,
        dim_location.location_chain_id,
        dim_location.location_chain_key,
        dim_location.location_chain,
        dim_location.location_district_id,
        dim_location.location_district_key,
        dim_location.location_district,
        dim_location.location_region_id,
        dim_location.location_region_key,
        dim_location.location_region,
        
        -- logical info
        fact_inventory.date_key,

        -- facts
        fact_inventory.on_hand_units,
        fact_inventory.on_hand_dollars,
        fact_inventory.on_order_units,
        fact_inventory.on_order_dollars,
        fact_inventory.allocated_units,
        fact_inventory.available_units
    FROM fact_inventory
    INNER JOIN dim_item USING (item_id)
    INNER JOIN dim_location USING (location_id)
)

SELECT * FROM denormalized 