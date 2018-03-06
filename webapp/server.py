import flask
import shapely.geometry
import numpy as np
import math
import rasterio.features
import h5py
import datetime

app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.redirect(flask.url_for('static', filename='client.html'))

@app.route('/plot', methods=['POST'])
def receive():
    data = flask.request.get_json()
    snap = data['snap']
    geom = shapely.geometry.shape(data['geometry'])

    g = grid(geom, snap=snap)

    #response = flask.jsonify(dict(geometry=g.geom, output='<h1>Hello</h1>'))
    response = flask.jsonify(dict(geometry=g.geom,
                                  dates=newdates.astype(str).tolist(), # aiming for ISO8601 dates
                                  data=g.data.tolist()
                                  ))

    #response = flask.jsonify(geom_to_grid(geom, snap=snap, res=2500))
    response.status_code = 200
    return response


t0 = np.datetime64('1986-01-01', 'D')
tf = np.datetime64('2020-01-01', 'D')
day = np.timedelta64(1, 'D')
epochs = int((tf - t0) / day) + 1 # inclusive count of distinct dates
newdates = t0 + day * np.arange(epochs)


class grid:
    file = h5py.File('../output.h5', 'r')

    def __init__(self, geom, snap=True, res=2500):
        x0,y0,x1,y1 = np.asarray(geom.bounds) / res
        x0,y0,x1,y1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)

        self.affine = rasterio.Affine(res, 0, x0*res, 0, -res, y1*res)
        self.shape = (y1-y0, x1-x0)
        self.x0 = x0
        self.y1 = y1

        self.process(geom, snap)

    def rasterise(self, geom, all_touched=True):
        return rasterio.features.rasterize([geom],
                                           out_shape=self.shape,
                                           transform=self.affine,
                                           all_touched=all_touched)
    def indices(self, array):
        j, i = array.nonzero()
        i += self.x0 - (600)
        j -= self.y1 - (-1560)
        return zip(i, j)

    def vectorise(self, array):
        shapes = rasterio.features.shapes(array, mask=array,
                                          transform=self.affine)
        return list(geom for (geom, value) in shapes)

    def objectify_shapes(self, polygons):
        if len(polygons) == 1:
            self.geom = polygons[0]
        else:
            self.geom = dict(type='GeometryCollection', geometries=polygons)

    def process(self, geom, snap):
        # Mask to precomputed extent
        bounds = shapely.geometry.geo.box(1500000,-3950000,1550000,-3900000)
        mask = self.rasterise(bounds, False)

        array = self.rasterise(geom, (not snap))
        array &= mask

        if snap:
            polygons = self.vectorise(array)
            self.objectify_shapes(polygons)
        else:
            partial = self.rasterise(geom.exterior)
            partial &= mask

            array[partial != 0] = 0

            clipped = geom.intersection(bounds)
            self.geom = shapely.geometry.mapping(clipped)
            #polygons += self.vectorise(partial)

        ds = self.file['data']

        # TODO: handle if array or partial are entirely empty
        summary = sum(ds[:,y,x,:] for (x,y) in self.indices(array))
        if not snap:
            possible = sum(ds[:,y,x,1:] for (x,y) in self.indices(partial))
            summary[:,-2] += 0.5 * possible[:,-2] # best guess half inside poly
            summary[:,-1] += possible[:,-1] # upper envelope grows, not lower

        self.data = summary * (25**2) / (100**2)



