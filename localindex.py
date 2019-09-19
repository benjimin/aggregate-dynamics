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
import logging
import sqlite3
import datetime
import sqlalchemy #import psycopg2
import pandas

cache_location = 'cache.db'

def cache(x, y):
    return get(x, y)
    try:
        return get(x, y)
    except sqlite3.OperationalError:
        harvest()
        return get(x, y)

def harvest():
    """
    Harvest everything (all cells).

    Takes about 9 minutes (2min CPU + 7min DB) for 2.7 million tiles (wofls).

    Ideally could build local indexes, but query performance may already suffice.

    TODO: Generalise to other ODC products (not just wofs_albers)
    """
    logging.info('Re-caching ' + cache_location)

    slow = sqlalchemy.create_engine('postgres://agdc-db/datacube',
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
    fast.execute("create index cellindex on wofs_albers (x,y)") # 7 seconds
    #fast.execute("create index sorted on wofs_albers (x,y,date(time, '+10 hours'))")
    # Issue: sqlite probably can't use index for date function until v3.20.0
    fast.close() # appropriate for connection not engine

def get(x, y):
    """
    Retrieve cached results for any cell.

    Currently takes ~1 second per cell.
    TODO: could probably speed up by building indexes during harvest
    """
    with sqlite3.connect(cache_location) as c:
        results = c.execute("""select date(time, '+10 hours') as date,
                                   group_concat(filename)
                                from wofs_albers
                                where x=? and y=? and gqa<1
                                group by date
                                order by date""", (x,y))
        # parse date-string and comma-separated-list
        return [(datetime.date(*map(int,t.split('-'))), f.split(','))
                for t,f in results.fetchall()]

def get_time(x, y):
    """
    Find exact times of observation
    """
    with sqlite3.connect(cache_location) as c:
        results = c.execute("""select date(time, '+10 hours') as date,
                                   group_concat(time)
                                from wofs_albers
                                where x=? and y=? and gqa<1
                                group by date
                                order by date""", (x,y))
        # parse date-string and comma-separated-list
        return [(datetime.date(*map(int,t.split('-'))), f.split(','))
                for t,f in results.fetchall()]

# takes about 7 minutes to complete on staging-db (however took 1hr23min on prod)
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

if __name__ == '__main__':
    import os.path
    if not os.path.exists(cache_location):
        harvest()