{% macro source(source_name, table_name) %}
  {% set rel = builtins.source(source_name, table_name) %}
  
  {% if execute %}
    {% set node_id = "source." ~ project_name ~ "." ~ source_name ~ "." ~ table_name %}
    {% if node_id in graph.sources %}
      {% set node = graph.sources[node_id] %}
      
      {% if 'has_mock_data' in node.tags and 'not-has_mock_data' not in node.tags and var('use_mock_data') == 'true' %}
        {% set mock_dataset = target.schema ~ "_mock" %}
        {% set newrel = rel.replace_path(database=target.database, schema=mock_dataset, identifier="seed_" ~ node['name']) %}
        
        {{ log("Using mock data for " ~ node['name'] ~ ": " ~ newrel, info=true) }}
        {% set rel = newrel %}
      {% endif %}
    {% endif %}
  {% endif %}
  
  {{ return(rel) }}
{% endmacro %} 