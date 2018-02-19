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

Good example Albers cell: (15,-40) : Lake George, Burley Griffin, etc.

Estimates:
    Cell raw data: 32gb
    Raw chunk: 20mb
    Foliated: 130mb
    Summary pixel: 0.2mb
    Cell summary: <300mb
    Continental summary: <0.5tb
Thus, want of order 40gb memory.

Or, only read a quarter of raw cell (which has 200x200 chunks).

"""

import logging

def cell_input(x, y, window=None):
    global everything

    import localindex
    results = localindex.cache(x, y)[:10]

    window = (0, 2000), (0, 2000)
    rastershape = [4000, 4000] if window is None else [b-a for a,b in window]



    # declare enormous array
    import numpy as np
    logging.info(len(results))
    everything = np.zeros([len(results)] + rastershape, dtype=np.uint8)

    # load data (ideally with parallel IO)
    import rasterio
    def load(task):
        row, filenames = task
        for i, name in enumerate(filenames):
            assert name.endswith('.nc')
            with rasterio.open('NetCDF:' + name + ':water') as f:
                this = f.read(1, window=window)
            if i: # fuser
                hole = (everything[row,:,:] & 1).astype(np.bool)
                overlap = ~(hole | (this & 1).astype(np.bool))
                everything[row][hole] = this[hole]
                everything[row][overlap] |= this[overlap]
            else:
                everything[row,:,:] = this

    #for i,paths in enumerate(results):
    #    load(i, paths)
    import multiprocessing.dummy as multithreading
    pool = multithreading.Pool(4)
    pool.map(load, enumerate(fs for (d,fs) in results), chunksize=1)
    pool.close() # instruct workers to exit when idle
    pool.join() # wait

    return everything

def grid_workflow():
    import dask
    import multiprocessing
    # read input in full-temporal depth 100x100 pillars
    x = dask.array.from_array(everything, chunks=(-1, 100, 100))
    # output chunks are a different shape
    agg = dask.array.map_blocks(aggregate_chunk, x,
                                chunks=(13000,1,1,3), newaxis=3)
    with dask.set_options(pool=multiprocessing.pool.ThreadPool(2)):
        agg.to_hdf5('output.h5')

def aggregate_chunk():

    pass





