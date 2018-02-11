"""

This is an *enabling* performance optimisation.
Pre-aggregation is intended to enable interactive extent selection.

Will operate on one spatial cell (4000x4000 pixels) at a time.
A cell may have of order 2000 temporal layers.
(Some layers could require grouping and fusing.)
Assume layers are 8bit datatype.

The full cell (few tens of GB) should fit in memory.
Otherwise, would write temporarily to local SSD (with appropriate chunking).
Thus, only need to read once from network storage.
Will parallelise reads.

Assume aggregate is 32bit.
Will aggregate at 100x100 spatial and roughly 13000 temporal.
(This is constrained by total storage for roughly 1300 cells.)
Thus, aggregate (80MB) does not consume significant memory, and
should not require parallel output. (Unless combining multiple cells.)

Compute will be chunked.
(Unchunked would require multiples of the large array in memory simultaneously,
some likely needing to be a minimum of uint16 to store counts.)
Smallest natural chunk is single output column.
If 8bit datatype, this is again of order 100MB.
(In practice may use several multiples of this.)
Processing is likely to be memory access intensive.

-------

Note, most of the work of compiling the work list is the extracting of
cell indices from metadata of numerous records. So could speed this up
even further with a database index or a locally cached database.

"""

def cell_input(x, y):
    global everything

    # find list of potential datasets (either from filesystem or database)
    # apply filtering on GQA metadata
    # group as appropriate (aiming to fuse down to one layer per day)
    #TODO: Set up local cache sqlite3/spatialite, transferring n rows at a time..
    import string
    import psycopg2
    c = psycopg2.connect(host='agdcstaging-db', database='wofstest').cursor()
    sql = string.Template(sql_one_cell).substitute(x=x, y=y, product='wofs_albers')
    c.execute(sql)
    results = c.fetchall()

    # declare enormous array
    import numpy as np
    print(len(results))
    everything = np.empty((len(results), 4000, 4000))

    # load data (ideally with parallel IO)
    import rasterio
    def task(row, filenames):
        for name in filenames:
            with rasterio.open(name) as f:
                this = f.read(1)
            # TODO: implement fuser
            everything[row,:,:] = this

    for i,paths in enumerate(results):
        #TODO: multithreading queue
        task(i, paths)

    return everything

def grid_workflow():

    # declare output array

    # get input array

    # for chunk,
    pass

def aggregate_chunk():

    pass


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



