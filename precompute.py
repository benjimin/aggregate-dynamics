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

------

Usage:

    #x = cell_input(15, -40, window=((0,2000),(0,2000)), limit=20)
    %time t,x = cell_input(15, -40, window=((0,2000),(0,2000)))
    %time grid_workflow(x, t)

    or

    %time workflow(15, -40, quadrant=0)

"""
import rasterio
import numpy as np
import logging
import dask.array
import multiprocessing
import functools
import operator
import mmap
import zarr

import localindex
import aggregate

window = ((0,2000),(0,2000))

windows = {0: ((0,2000),(0,2000)), # TODO: check handling of boundary pixels..
           1: ((0,2000),(2000,4000)),
           2: ((2000,4000),(0,2000)),
           3: ((2000,4000),(2000,4000))}

def sharedarray(shape, dtype):
    """
    Sharing numpy array between processes

    Usage: assign returned ndarray to a global before forking.

    Note explicit passing fails, presumably due to argument serialisation.
    """
    size = functools.reduce(operator.mul, shape) * np.dtype(dtype).itemsize
    class releasor(mmap.mmap):
        def __del__(self):
            self.close()
    buffer = releasor(-1, size)
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)

def load(row, filenames):
    """IO subtask: load selected slice with fusion of source data"""
    global obsdata
    global rasterwindow
    dest = obsdata[row]
    for i, name in enumerate(filenames):
        assert name.endswith('.nc')
        with rasterio.open('NetCDF:' + name + ':water') as f:
            this = f.read(1, window=rasterwindow, out=(None if i else dest))
        if i: # fuser
            hole = (dest & 1).astype(np.bool)
            overlap = ~(hole | (this & 1).astype(np.bool))
            dest[hole] = this[hole]
            dest[overlap] |= this[overlap]

def cell_input(x, y, window=None, limit=None): # e.g. window = (0, 2000), (0, 2000)
    """Load dense stack of input data, using parallel IO"""
    global obsdata
    global obsdates
    global rasterwindow

    rasterwindow = window

    results = localindex.cache(x, y)[:limit]
    dates, filenames = zip(*results)

    dates = np.asarray(dates, dtype=np.datetime64)

    rastershape = [4000, 4000] if window is None else [b-a for a,b in window]

    # declare enormous array

    pixels = (lambda x,y : x*y)(*rastershape)
    obs = len(results)
    size = pixels * obs

    logging.info(str(obs))

    obsdata = sharedarray([obs]+rastershape, np.uint8)
    obsdates = dates

    with multiprocessing.Pool(16) as pool:
        pool.starmap(load, enumerate(filenames))

    return dates, obsdata

class storage:
    """
    dask.array.store interface

    We are using 25m x 25m resolution data, and aggregating at 100x100 pixels,
    i.e. each output array element spans 2.5km along either axis.

    In Albers projection, the Australian region spans 4200km x 4000km,
    extending east and north from the corner -2000000mE -5000000mN.

    The datacube system uses 100km x 100km (4000^2 pixel) tiles, indexed by
    the south west corner coordinates in Albers (divided by 100000m).

    So the origin tile is (-20,-50), and the output increments by 40 for
    each tile index increment, with the total output dimensions 1680x1600.

    Chunks will have shape (~12k, 1, 1, 3): temporal, 2D spatial, which curve.
    """
    origin_tx = -20
    origin_ty = -50
    def __init__(self, tx, ty, window=((0,),(0,))):
        self.offset_x = int( (tx - self.origin_tx) * 40 + window[0][0] / 100 )
        self.offset_y = int( (ty - self.origin_ty) * 40 + window[1][0] / 100 )
        epochs = len(aggregate.defaultdates)
        self.array = zarr.open('data.zarr', mode='w',
                               shape=(epochs, 1680, 1600, 3),
                               chunks=(epochs, 1, 1, 3))
    def __setitem__(self, key, value):
        #print(key)
        assert len(key) == 4
        def increment(s, offset):
            assert isinstance(s, slice) # otherwise, return s + offset ?
            assert isinstance(offset, int)
            return slice(s.start + offset, s.stop + offset, s.step)
        sx = increment(key[1], self.offset_x)
        sy = increment(key[2], self.offset_y)
        key2 = key[0], sx, sy, key[3]
        #print(key2)
        self.array[key2] = value

def grid_workflow(obsdata, obsdates):
    """
    Orchestrate aggregation (with multithreading) and store result.

    This accesses n_obs x 100 x 100 chunks of the input,
    to generate n_days x 1 x 1 chunks of output.
    """
    def aggregate_chunk(chunk):
        lower, expect, upper = aggregate.aggregate_wofs(chunk, obsdates)
        return np.vstack([lower, expect, upper]).T[:,None,None,:]
    epochs = len(aggregate.defaultdates)

    # read input in full-temporal depth 100x100 pillars
    x = dask.array.from_array(obsdata, chunks=(-1, 100, 100))
    # output chunks are a different shape
    agg = dask.array.map_blocks(aggregate_chunk, x, dtype=np.float32,
                                chunks=(epochs,1,1,3), new_axis=3)
    with dask.set_options(pool=multiprocessing.pool.ThreadPool(8)):
        agg.to_hdf5('output.h5', '/data')

    return agg

def workflow(tx, ty, quadrant=0):
    """
    Orchestrate aggregation (with multithreading) and store result.

    This accesses n_obs x 100 x 100 chunks of the input,
    to generate n_days x 1 x 1 chunks of output.
    """
    window = windows[quadrant]

    output = storage(tx, ty, window)

    obsdates, obsdata = cell_input(tx, ty, window)

    def aggregate_chunk(chunk):
        lower, expect, upper = aggregate.aggregate_wofs(chunk, obsdates)
        return np.vstack([lower, expect, upper]).T[:,None,None,:]
    epochs = len(aggregate.defaultdates)

    # read input in full-temporal depth 100x100 pillars
    x = dask.array.from_array(obsdata, chunks=(-1, 100, 100))
    # output chunks are a different shape
    agg = dask.array.map_blocks(aggregate_chunk, x, dtype=np.float32,
                                chunks=(epochs,1,1,3), new_axis=3)
    with dask.set_options(pool=multiprocessing.pool.ThreadPool(8)):
        agg.store(output, lock=False)

