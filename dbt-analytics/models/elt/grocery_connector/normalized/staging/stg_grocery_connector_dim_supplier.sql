{{
  config(
    materialized = 'table',
    indexes = [
      {'columns': ['account_id', 'connector_id'], 'type': 'btree'},
      {'columns': ['supplier_key'], 'type': 'btree'}
    ]
  )
}}

-- Staging layer for grocery connectorr supplier dimension
SELECT DISTINCT
  account_id,
  connector_id,
  supplier_key,
  supplier_name
FROM {{ ref('stg_grocery_connector_item_attributes') }}
WHERE supplier_key IS NOT NULL
  AND supplier_name IS NOT NULL
  AND TRIM(supplier_name) != '' 


