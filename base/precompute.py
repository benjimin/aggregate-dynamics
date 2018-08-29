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
import numpy as np
import logging
import dask.array
import multiprocessing
import zarr

from . import aggregate
from . import dataload
from . import pyramid


def workflow(tx, ty, quadrant=0):
    """
    Orchestrate aggregation (with multithreading) and store result.

    This accesses n_obs x 100 x 100 chunks of the input,
    to generate n_days x 1 x 1 chunks of output.
    """
    window = dataload.windows[quadrant]

    output = pyramid.storage(tx, ty, window)

    obsdates, obsdata = dataload.cell_input(tx, ty, window)

    print("Collected input..")

    def aggregate_chunk(chunk):
        print("<chunk>")
        lower, expect, upper = aggregate.aggregate_wofs(chunk, obsdates)
        return np.vstack([lower, expect, upper]).T[:,None,None,:]
    epochs = len(aggregate.defaultdates)

    # read input in full-temporal depth 100x100 pillars
    x = dask.array.from_array(obsdata, chunks=(-1, 100, 100))
    # output chunks are a different shape
    agg = dask.array.map_blocks(aggregate_chunk, x, dtype=np.float32,
                                chunks=(epochs,1,1,3), new_axis=3)
    with dask.config.set(pool=multiprocessing.pool.ThreadPool(8)):
        agg.store(output, lock=False)
        #agg.to_hdf5('output.h5', '/data')

if __name__ == '__main__':
    workflow(15,-40)
