/*

Query Template

Parameters: $product, $x, $y, $gqa

*/

/*

-- seems that collecting all tiles is not much slower than one tile

create temporary table tiles as
select id,
    (metadata->'grid_spatial'->'projection'->'geo_ref_points'->'ll'->>
        'x')::numeric::int / 100000 as x,
    (metadata->'grid_spatial'->'projection'->'geo_ref_points'->'ll'->>
        'y')::numeric::int / 100000 as y
from agdc.dataset
where archived is null
and dataset_type_ref =
    (select id from agdc.dataset_type where name like 'wofs_albers')
;

*/

with

seed(id) as (
    select id from agdc.dataset
    where
        dataset_type_ref in
            (select id from agdc.dataset_type where name like '$product')
        and archived is null
        and metadata->'grid_spatial'->'projection'->'geo_ref_points'->'ll'
            = jsonb_build_object('x', $x * 100000, 'y', $y * 100000)
),
recursive lineage(child, ancestor) as (

        select seed.id, src.source_dataset_ref
        from seed join agdc.dataset_source src
        on seed.id = src.dataset_ref
    union
        select lineage.child, src.source_dataset_ref
        from lineage join agdc.dataset_source src
        on lineage.ancestor = src.dataset_ref
),
select * from lineage limit 10;

history as (

    select distinct lineage.ancestor, gqa...
    from lineage join agadc.dataset d
    on lineage.ancestor = d.id
    where d.dataset_type_ref in
        (select id from agdc.dataset_type where name like '%level1%scene')

)

select uri_body
