"""

This version retrieves each cell from database independently.
Hence, may take nearly a minute per cell.

Note, most of the DB work of compiling the task list is the extracting of
cell indices from metadata of numerous records. So could speed this up
even further with a database index or a locally cached database.

Also todo for sqlite3 conn:
    .enable_load_extension(True)
    .load_extension("mod_spatialite")

"""


def get(x, y):
    # find list of potential datasets (either from filesystem or database)
    # apply filtering on GQA metadata
    # group as appropriate (aiming to fuse down to one layer per day)
    #TODO: Set up local cache sqlite3/spatialite, transferring n rows at a time..
    import string
    import psycopg2
    c = psycopg2.connect(host='agdcstaging-db', database='wofstest').cursor()
    sql = string.Template(sql_one_cell).substitute(x=x, y=y, product='wofs_albers')
    c.execute(sql)
    return c.fetchall()


sql_one_cell = """

with recursive lineage(child, ancestor) as (
        select seed.id, src.source_dataset_ref
        from seed join agdc.dataset_source src
        on seed.id = src.dataset_ref
    union
        select lineage.child, src.source_dataset_ref
        from lineage join agdc.dataset_source src
        on lineage.ancestor = src.dataset_ref
),
seed(id) as (
    select id from agdc.dataset
    where
        dataset_type_ref in
            (select id from agdc.dataset_type where name like '$product')
        and archived is null
        and metadata->'grid_spatial'->'projection'->'geo_ref_points'->'ll'
            = jsonb_build_object('x', $x * 100000, 'y', $y * 100000)
),
filter(id) as (
    select lineage.child
    from lineage join agdc.dataset d
    on lineage.ancestor = d.id
    where d.dataset_type_ref in
        (select id from agdc.dataset_type where name like '%level1%scene')
    group by lineage.child
    having
        max((d.metadata->'gqa'->'residual'->'iterative_mean'->>'xy')::numeric)
        < 1
),
locate(id, path) as (
    select distinct on (filter.id)
        filter.id, path.uri_body
    from filter join agdc.dataset_location path
    on filter.id = path.dataset_ref
    where path.archived is null
)
select
    (((d.metadata->'extent'->>'center_dt')::timestamp at time zone 'UTC')
        at time zone 'AEST')::date as date,
    array_agg(locate.path)
from locate join agdc.dataset d on d.id=locate.id
group by date
order by date

"""
