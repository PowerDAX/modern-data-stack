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
        key_columns=['account_id', 'connector_id', 'store_region_key'],
        attribute_columns=['store_region'],
        sorting=['date_key DESC NULLS LAST'],
        ignore_nulls=true
    ) }}
)

SELECT
    account_id,
    connector_id,
    store_region_key,
    store_region
FROM deduped 
