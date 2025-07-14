{{
  config(
    materialized = 'table',
    indexes = [
      {'columns': ['account_id', 'connector_id'], 'type': 'btree'},
      {'columns': ['supplier_key'], 'type': 'btree'}
    ]
  )
}}

-- Normalized grocery connectorr supplier dimension
WITH supplier_dedup AS (
  SELECT
    account_id,
    connector_id,
    supplier_key,
    supplier_name,
    ROW_NUMBER() OVER (
      PARTITION BY account_id, connector_id, supplier_key
      ORDER BY supplier_name
    ) AS rn
  FROM {{ ref('stg_grocery_connector_dim_supplier') }}
)

SELECT
  account_id,
  connector_id,
  supplier_key,
  supplier_name,
  'grocery_connector' AS source_system,
  CURRENT_TIMESTAMP AS created_at
FROM supplier_dedup
WHERE rn = 1 


