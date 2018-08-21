"""

This version retrieves each cell from database independently.
Hence, may take nearly a minute per cell.

Note, most of the DB work of compiling the task list is the extracting of
cell indices from metadata of numerous records. So could speed this up
even further with a database index or a locally cached database.

Also todo for sqlite3 conn:
    .enable_load_extension(True)
    .load_extension("mod_spatialite")

Limitation: assumes product is not "stacked". Stacking multiple dates into
the same file would alter the strategy for accessing them.

TODO: spatialite rather than working on (and assuming) whole tiles.

"""
import string
import psycopg2
import logging
import sqlite3
import pickle
import datetime

def cache(x, y):
    return old_version(x, y)
    #return new_version(x, y)

def old_version(x, y):
    x = int(x)
    y = int(y)

    filename = 'cache_%i_%i.pkl' % (x, y)

    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        logging.info('Using cache ' + filename)
    except FileNotFoundError:
        logging.info('Re-caching ' + filename)
        results = harvest_one_cell(x, y)
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    return results

def harvest_one_cell(x, y):
    # find list of potential datasets (either from filesystem or database)
    # apply filtering on GQA metadata
    # group as appropriate (aiming to fuse down to one layer per day)
    #TODO: Set up local cache sqlite3/spatialite, transferring n rows at a time..
    c = psycopg2.connect(host='agdcstaging-db', database='wofstest').cursor()
    sql = string.Template(sql_one_cell).substitute(x=x, y=y, product='wofs_albers')
    c.execute(sql)
    return c.fetchall()

def new_version(x, y):
    filename = 'cache.db'
    try:
        return get2(x, y)
    except sqlite3.OperationalError:
        logging.info('Re-caching ' + filename)
        #populate_all_cells(filename)
        return get2(x, y)


def harvest_all_cells(cache_location):
    """
    Harvest everything.

    Takes about 9 minutes (2min CPU + 7min DB) for 2.7 million tiles (wofls).

    Ideally could build local indexes, but query performance may already suffice.

    TODO: Generalise to other ODC products (not just wofs_albers)
    """
    import sqlite3
    import sqlalchemy
    import pandas

    slow = sqlalchemy.create_engine('postgres://agdcstaging-db/wofstest',
                                    execution_options={'stream_results':True})
    fast = sqlite3.connect(cache_location)
    #      or sqlalchemy.create_engine('sqlite:///cache.db')

    fast.execute("drop table if exists wofs_albers") # fresh

    for chunk in pandas.read_sql(sqlalchemy.text(sql_harvest), slow, chunksize=1000):
        chunk['id'] = chunk.id.astype(str) # uuid unsupported by sqlite
        chunk.to_sql(name='wofs_albers',
                     con=fast,
                     if_exists='append', # concatenate chunks
                     index=False) # do not explicitly store a row number

    fast.commit() # probably not necessary?
    fast.close() # appropriate for connection not engine

def get2(x, y):
    """
    Retrieve cached results for any cell.

    Currently takes ~1 second per cell.
    TODO: could probably speed up by building indexes during harvest
    """
    with sqlite3.connect("cache.db") as c:
        results = c.execute("""select date(time, '+10 hours') as date,
                                   group_concat(filename)
                                from wofs_albers
                                where x=? and y=? and gqa<1
                                group by date
                                order by date""", (x,y))
        # parse date-string and comma-separated-list
        return [(datetime.date(*map(int,t.split('-'))), f.split(','))
                for t,f in results.fetchall()]

# takes about 7 minutes to complete:
sql_harvest = """

with recursive lineage(child, ancestor) as (
        select d.id, d.id
        from agdc.dataset d
        where d.dataset_type_ref =
            (select id from agdc.dataset_type where name = 'wofs_albers')
        and d.archived is null
    union
        select h.child, src.source_dataset_ref
        from lineage h join agdc.dataset_source src
        on h.ancestor = src.dataset_ref
)
select j.*,
    ((d.metadata->'extent'->>'center_dt')::timestamp at time zone 'UTC') as time,
    --metadata->'grid_spatial'->>'projection' as spatial
    (metadata->'grid_spatial'->'projection'->'geo_ref_points'->'ll'->>'x')::numeric / 100000 as x,
    (metadata->'grid_spatial'->'projection'->'geo_ref_points'->'ll'->>'y')::numeric / 100000 as y
from (
    select distinct on (hist.id) hist.*, path.uri_body as filename
    from (
        select h.child id,
            max((src.metadata->'gqa'->'residual'->'iterative_mean'->>'xy')::numeric) gqa
        from lineage h
        left join agdc.dataset src on h.ancestor = src.id
        where src.dataset_type_ref in
            (select id from agdc.dataset_type where name like '%level1%scene')
        group by h.child
    ) as hist
    join agdc.dataset_location path on hist.id = path.dataset_ref
) as j
join agdc.dataset d on d.id = j.id
"""


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
