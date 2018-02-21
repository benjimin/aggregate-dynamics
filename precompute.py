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
import rasterio
import numpy as np
import logging
import localindex
#import multiprocessing.dummy as multithreading
import dask.array
import multiprocessing

t0 = np.datetime64('1986-01-01', 'D')
tf = np.datetime64('2020-01-01', 'D')
day = np.timedelta64(1, 'D')
epochs = int((tf - t0) / day) + 1 # inclusive count of distinct dates
newdates = t0 + day * np.arange(epochs)
assert epochs == len(newdates)

def cell_input(x, y, window=None, limit=None): # e.g. window = (0, 2000), (0, 2000)
    global everything
    global dates

    results = localindex.cache(x, y)[:limit]
    dates, filenames = zip(*results)

    dates = np.asarray(dates, dtype=np.datetime64)

    rastershape = [4000, 4000] if window is None else [b-a for a,b in window]

    # declare enormous array

    logging.info(len(results))
    everything = np.zeros([len(results)] + rastershape, dtype=np.uint8)

    # load data (ideally with parallel IO)
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
    pool = multiprocessing.pool.ThreadPool(4)
    #pool = multithreading.Pool(4)
    pool.map(load, enumerate(filenames), chunksize=1)
    pool.close() # instruct workers to exit when idle
    pool.join() # wait

    return dates, everything

def grid_workflow():
    # read input in full-temporal depth 100x100 pillars
    x = dask.array.from_array(everything, chunks=(-1, 100, 100))
    # output chunks are a different shape
    agg = dask.array.map_blocks(aggregate_chunk, x, dtype=np.float32,
                                chunks=(epochs,1,1,3), new_axis=3)
    with dask.set_options(pool=multiprocessing.pool.ThreadPool(4)):
        agg.to_hdf5('output.h5', '/data')

def aggregate_chunk(chunk):
    global dates

    assert epochs == len(newdates)

    def wofs_prep(chunk): # map wofs input to masked boolean
        chunk &= ~np.uint8(4) # remove ocean masking

        wet = chunk == 128
        dry = chunk == 0
        clear = wet | dry

        return np.ma.array(wet, mask=~clear)

    def foliate(chunk, chunkdates):
        indices = ((chunkdates - t0) /  day).round().astype(int)

        foliage = np.zeros((epochs,) + chunk.shape[1:], dtype=np.bool)
        foliage = np.ma.array(data=foliage, mask=True)

        foliage[indices] = chunk
        return foliage

    def interpolate(maskedchunk):
        data = maskedchunk.data
        mask = maskedchunk.mask

        past = mask.cumsum(axis=0)
        future = mask[::-1,...].cumsum(axis=0)[::-1,...]
        unbound = (past == 0) | (future == 0)

        # constant-value extrapolation
        epochs = len(data)
        forward = data.copy()
        reverse = data.copy()
        for i in range(1, epochs):
            gaps = mask[i]
            forward[i][gaps] = forward[i-1][gaps]
        for i in range(epochs-1)[::-1]:
            gaps = mask[i]
            reverse[i][gaps] = reverse[i+1][gaps]

        upper = forward | reverse | unbound # might be wet
        lower = forward & reverse & ~unbound # must be wet
        return upper, lower

    def summarise(upper, lower):
        upper = upper.sum(axis=(1,2))
        lower = lower.sum(axis=(1,2))

        estimate = 0.5 * (upper + lower)

        return np.vstack([lower, estimate, upper]).astype(np.float32)

    obs = wofs_prep(chunk)

    foliage = foliate(obs, dates)

    a, b = interpolate(foliage)

    return summarise(a, b).T[:,None,None,:] # -> shape (t,1,1,3)

x = cell_input(15, -40, window=((0,2000),(0,2000)), limit=20)
