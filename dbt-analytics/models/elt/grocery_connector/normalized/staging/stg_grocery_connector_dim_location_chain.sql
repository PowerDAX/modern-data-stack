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
        key_columns=['account_id', 'connector_id', 'location_chain_key'],
        attribute_columns=['location_chain'],
        sorting=['date_key DESC NULLS LAST'],
        ignore_nulls=true
    ) }}
)

SELECT
    account_id,
    connector_id,
    location_chain_key,
    location_chain
FROM deduped 