{{
    config(
        materialized='table'
    )
}}

WITH location_attributes AS (
    SELECT * FROM {{ ref('stg_grocery_connector_location_attributes') }}
),

deduped AS (
    {{ deduplicate_using_prioritization(
        table_name='location_attributes',
        key_columns=['account_id', 'connector_id', 'location_district_key'],
        attribute_columns=['location_district', 'location_region_key'],
        sorting=['date_key DESC NULLS LAST'],
        ignore_nulls=true
    ) }}
)

SELECT
    account_id,
    connector_id,
    location_district_key,
    location_district,
    location_region_key
FROM deduped 
