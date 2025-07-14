{%- macro aggregated_column( column, aggregation ) -%}
        {%- if aggregation == 'sum' -%}
        sum( round( cast( {{ column }} as NUMERIC), 2))
        {%- elif aggregation == 'sumprecise' -%}
        sum( cast( {{ column }} as NUMERIC))
        {%- elif aggregation == 'count' -%}
        cast( count( {{ column }}) as NUMERIC)
        {%- elif aggregation == 'countdistinct' -%}
        cast( count( distinct {{ column }}) as NUMERIC)
        {%- else -%}
        1
        {%- endif -%}
{%- endmacro -%}


{# ---------------------------------------------------------------------------------------------------------------- #}


{% macro test_fact_data_validation( model, column_name, to, field, aggregation, date = none) %}

with aggregation as (
    select
        'from' AS source_table,
        {{ aggregated_column(field, aggregation) }} as value
    from {{ to }}
    union all
    select
        'to' as source_table,
        {{ aggregated_column(column_name, aggregation) }} as value
    from {{ model }}
),
pivoted as (
    select * from
    (
        select
            source_table,
            round( value, 0) as value
            from aggregation
    )
    pivot
    (
        sum(value) as pipeline
        for source_table in ('from', 'to')
    )
)

select * from (
    select pipeline_to - pipeline_from as result from pivoted
) where abs(result) > 1 -- Allow for slight rounding differences

{% endmacro %}


{# ---------------------------------------------------------------------------------------------------------------- #}


{% macro test_fact_data_validation_w_query( model, column_name, to, field, aggregation, date = none, tested_table_filter = none) %}

with aggregation as (
    select
        'source_table' AS source_table,
        '{{ to }}' as source_table_string,
        '{{ aggregated_column(field, aggregation) }}' as aggregation_string,
        {{ aggregated_column(field, aggregation) }} as value
    from {{ to }}
    group by 1,2,3
    union all
    select
        'tested_table' as source_table,
        '{{ model }}' as source_table_string,
        '{{ aggregated_column(column_name, aggregation) }}' as aggregation_string,
        {{ aggregated_column(column_name, aggregation) }} as value
    from {{ model }}
    {% if tested_table_filter %}
    WHERE {{ tested_table_filter }}
    {% endif %}
    group by 1,2,3
),
pivoted as (
    select * from
    (
        select
            source_table,
            source_table_string,
            aggregation_string,
            round( value, 0) as value
            from aggregation
    )
    pivot
    (
        max(source_table_string) as source_table_string,
        max(aggregation_string) as aggregation_string,
        sum(value) as pipeline
        for source_table in ('source_table', 'tested_table')
    )
),
final as (
    select
        concat("select 'source table' as source_table, ", aggregation_string_source_table, " as value ",
               "from ", source_table_string_source_table," ",
               "union all ",
               "select 'tested table' as source_table, ", aggregation_string_tested_table, " as value ",
               "from ", source_table_string_tested_table) as sql_query,
        (pipeline_tested_table - pipeline_source_table) as result
    from pivoted
)

select *
from final
where abs(result) > 1 -- Allow for slight rounding differences

{% endmacro %}
