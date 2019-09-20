import math
import numpy as np
from collections import deque

"""
The intended product is an image pyramid.
That consists of multiple pages (representing sequential levels),
each composed of multiple tiles (each representing a chunk of raster data).

The input is the base (finest) level raster tiles, to be aggregated.

The algorithm is expected to:
  - Limit the maximum number of tiles concurrently held in memory.
  - Read each base input tile into memory only once.
  - Not revisit tiles that have been written out from memory.
  - Also want to implement pre-fetching, i.e. asynchronous IO, so that
    throughput is not limited to cumulative latency of individual reads/writes.
  - Perhaps reuse memory arrays, to avoid memory reallocation (both delays and
    opportunity for leaks).

Assumptions:
  - The reading order among base tiles will not unduly affect performance,
    although a raster-like scan is preferred (as more likely to avoid seeks).
  - The tiles may be written out in any order, without need to group by level.
  - The levels are aligned such that the finer subtiles beneath one tile do not
    overlap any other adjacent tiles. Moreover, assume each pixel in a coarser
    tile depends on a set of pixels in only one subtile. (E.g. the tile size
    is an integer multiple of the zoom factor.)

Depth-first strategy (unavoidably there may be a tile in progress at every
level, but concurrent tiles at an equal level should be avoided, except
explicitly when parallelising).

Once a tile is commenced, must schedule the subtiles. Once any subtile
is ready, its aggregate must be propagated up into the corresponding zone of
the coarser tile, so that the subtile's memory can be immediately reassigned
for the next subtile.

Want to predict base-level tiles to facilitate pre-fetching. Thus, there are
two types of task:
  - load a base-level tile into memory, or
  - dispatch a tile (i.e. from a tile now ready in memory, computer the
    aggregate, while outputting directly to a section of the parent tile.
    Also, prepare/compress the tile for storage. Once both complete,
    release the memory for another tile.)
One approach is to lazily expand a task queue, to determine the order in which
to pre-fetch the base tiles.

"""
"""
def tileBuilder(xres, yres, zoomfactor=2, size=128):
    assert xres > 0 and yres > 0

    assert size // zoomfactor == 0 # alignment

    levels = int(math.log(min([xres, yres])/size, zoomfactor))

    # extra alignment.. not really required
    assert xres // (zoomfactor**levels * 128) == 0
    assert yres // (zoomfactor**levels * 128) == 0

    class Tile:
        data = None
        def __init__(self, level, parent, extent):
            pass
        def subtiles(self):
            yield

def coarsest_level():
    for i in range(1):
        for j in range(2):
            tile = Tile()
            yield tile

def pyramid(toplevel):
    for tile

for tile in queue:
    if tile.data is None:
        pass
    if tile.ready
"""
## Test

def trivial(a, level, method='mean'):
    factor = 2**level
    m, n = a.shape
    a = a.reshape(m//factor, factor, n//factor, factor)
    return getattr(a, method)(axis=(1, -1))

xres = 2**3 * 2
yres = 2**3

raw = np.arange(xres * yres).reshape(yres, xres)

level1 = trivial(raw, 1)
level3 = trivial(raw, 3)

print(raw)
print(trivial(raw, 2))



def shape(level):
    mag = zoomfactor**level
    return yres // mag, xres // mag

size = 2 # let's use 2x2 tiles.
zoomfactor = 2
levels = int(math.log(min([xres, yres])/size, zoomfactor))
print(levels)

class Tile:
    size = 2
    zoom = 2
    def __init__(self, level, y_offset, x_offset, parent=None):
        self.level = level
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.parent = parent
    def subtiles(self):
        level = self.level - 1
        if level < 0:
            return
        i0 = self.y_offset * self.zoom
        j0 = self.x_offset * self.zoom
        increments = range(0, self.size * self.zoom, self.size)
        for i in increments:
            for j in increments:
                yield Tile(level, i + i0, j + j0, self)
    def window(self):
        return (slice(self.y_offset, self.y_offset + self.size),
                slice(self.x_offset, self.x_offset + self.size))
    def parent_window(self):
        y = self.y_offset // self.zoom
        x = self.x_offset // self.zoom
        s = self.size // self.zoom
        return slice(y, y + s), slice(x, x + s)
    def __repr__(self):
        return 'Tile' + str((self.level, self.y_offset, self.x_offset))

def coarse_tiles():
    mag = zoomfactor ** levels
    x = xres // mag
    y = yres // mag
    for i in range(0, y, size):
        for j in range(0, x, size):
            yield Tile(levels, i, j)

def depthfirst(tile=None):
    for t in tile.subtiles() if tile is not None else coarse_tiles():
        yield from depthfirst(t)
        yield t

output = [np.zeros(shape(i)) for i in range(levels + 1)]

#for i, t in enumerate(depthfirst()):
#    output[t.level][t.window()] = i

def condense(source, dest, zoom=Tile.zoom):
    m, n = dest.shape
    source = source.reshape(m, zoom, n, zoom)
    np.mean(source, axis=(1, -1), out=dest)

output[0] = raw

for t in depthfirst():
    if t.parent is not None:
        dest = output[t.level + 1][t.parent_window()]
        src = output[t.level][t.window()]
        condense(src, dest)


print(output[0])
print(output[1])
print(output[2])


"""




outputs = {i + 1: np.zeros(shape(i + 1)) for i in range(levels)}

def coarse_tiles():
    y, x = shape(levels)
    for i in range(0, y, size):
        for j in range(0, x, size):
            yield levels, (slice(i, i + size), slice(j, j + size))

def zoom_in(level, window):
    w1, w2 = window
    z = zoomfactor
    return level - 1, tuple(slice(w.start * z, w.stop * z) for w in window)

def iterate(window):
    w0, w1 = window
    seq = lambda s: range(s.start, s.stop, size)
    for i in seq(w0):
        for j in seq(w1):
            yield slice(i, i + size), slice(j, j + size)

def subtiles(level, window):
    if level < 1:
        return
    for subwindow in iterate(zoom_in(level, window)[1]):
        yield level - 1, subwindow

def depthfirst(top=None):
    branch = coarse_tiles() if top is None else subtiles(*top)
    for i in branch:
        yield from depthfirst(i)
        yield i

for level, window in depthfirst():
    fine = outputs[level][window]
    #fine.reshape

"""