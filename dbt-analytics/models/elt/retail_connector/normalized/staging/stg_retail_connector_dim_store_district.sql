{{
    config(
        materialized='table'
    )
}}

WITH store_attributes AS (
    SELECT * FROM {{ ref('stg_retail_connector_store_attributes') }}
),

deduped AS (
    {{ deduplicate_using_prioritization(
        table_name='store_attributes',
        key_columns=['account_id', 'connector_id', 'store_district_key'],
        attribute_columns=['store_district', 'store_region_key'],
        sorting=['date_key DESC NULLS LAST'],
        ignore_nulls=true
    ) }}
)

SELECT
    account_id,
    connector_id,
    store_district_key,
    store_district,
    store_region_key
FROM deduped 
