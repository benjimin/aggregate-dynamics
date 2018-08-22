"""
Using local index, retrieve large chunks of data.

Also performs fusing.
Uses parallel processing internally. Probably not threadsafe externally.

Current assumptions:
    - WOfS (WOFL bitfield interpretation) data
    - Upstream grouping (index groups results by local date)
    - Retrieve full temporal depth (spatial chunking only)
    - Co-aligned tiles (uniform extent, projection, etc)
    - Spatial window/chunk specified in pixel coordinates
    - Unstacked NetCDF

"""

import rasterio
import numpy as np
import logging
import multiprocessing
import functools
import operator
import mmap

import localindex

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