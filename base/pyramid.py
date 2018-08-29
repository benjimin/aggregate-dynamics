"""
Precomputed aggregates are stored in a coarse array.

Here will manage reading and writing.



We are interested in the Australian continent, in Albers (EPSG3577) projection.
Say, the rectangle spanning 4200km x 4000km,
with south-west corner at -2000000mE -5000000mN
(and hence north-east corner at +2200000mE -1000000).

The ODC/DEA convention uses 100km square tiles, referenced by the lower left
corner (the x, y coordinates there divided by 100km).
However, within each tile, the origin is instead the upper left coordinate
(i.e. the y value incremented by one 100km unit; each pixel extends 25m,-25m).

Note this implies the extent spans from tile (-20,-11) to (21,-50) inclusive.

We aggregate to a 2.5km grid (each unit covering 100x100 pixels).
This means 40x40 units per tile; the total extent will be 1680x1600.

Let the aggregate array origin be at the top left (consistent with tiles).
Note tile index decreases southward; other variables increase south and east.

"""
import zarr
import numpy as np

from . import aggregate

location = '../data.zarr'

# in projection units (m)
coordinate_bounds = -2000000,-1000000,2200000,-5000000
pixel_width = 25
tile_width = 10**5
res = 2500


assert not any([res % pixel_width, tile_width % res]) # insist integer factors

origin_tx = coordinate_bounds[0] // tile_width
origin_ty = coordinate_bounds[1] // tile_width - 1 # note offset (ll vs ul)
assert (origin_tx, origin_ty) == (-20, -11)

tile_inc = tile_width // res
assert tile_inc == 40

pixel_to_grid = res // pixel_width
assert pixel_to_grid == 100

assert tile_width // pixel_width == 4000

wx = coordinate_bounds[2] - coordinate_bounds[0]
wy = coordinate_bounds[1] - coordinate_bounds[3]
assert wx >= 0 and wy >= 0
assert not any([wx % tile_width, wy % tile_width])
size_x, size_y = wx // res, wy // res
assert size_x == 1680 and size_y == 1600

epochs = len(aggregate.defaultdates)

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

    def __init__(self, tx=origin_tx, ty=origin_ty, window=((0,),(0,))):
        px,py = window[0][0],window[1][0]
        # test window start is congruent with boundary
        assert not any([px % pixel_to_grid, py % pixel_to_grid])

        dtx = tx - origin_tx
        dty = -(ty - origin_ty) # note ty increases in opposite direction
        assert dty >= 0 and dtx >= 0

        self.offset_x = int( dtx * tile_inc + px // pixel_to_grid )
        self.offset_y = int( dty * tile_inc + py // pixel_to_grid )

        self.array = zarr.open(location, mode='w',
                               shape=(epochs, size_x, size_y, 3),
                               chunks=(epochs, 1, 1, 3))
    def __setitem__(self, key, value):
        assert len(key) == 4
        def increment(s, offset):
            assert isinstance(s, slice) # otherwise, return s + offset ?
            assert isinstance(offset, int)
            return slice(s.start + offset, s.stop + offset, s.step)
        sx = increment(key[1], self.offset_x)
        sy = increment(key[2], self.offset_y)
        key2 = key[0], sx, sy, key[3]
        self.array[key2] = value




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    array = zarr.open(location, mode='r')
    _,x,y,_ = array.shape
    img = np.zeros((x,y))
    print(img.shape)
    for i in range(x):
        print('.', end='', flush=True)
        for j in range(y):
            img[i,j] = array[:,i,j,1].mean()
    print('_')

    plt.figure()
    plt.imshow(img, origin='upper')
    plt.show()

