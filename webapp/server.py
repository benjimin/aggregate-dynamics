import flask
import shapely.geometry
import numpy as np
import math
import rasterio.features
import h5py
import matplotlib.pyplot as plt

app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.redirect(flask.url_for('static', filename='client.html'))

@app.route('/plot', methods=['POST'])
def receive():
    data = flask.request.get_json()
    snap = data['snap']
    geom = shapely.geometry.shape(data['geometry'])

    g = grid(geom, snap)

    #response = flask.jsonify(dict(geometry=g.geom, output='<h1>Hello</h1>'))
    response = flask.jsonify(dict(geometry=g.geom, output=g.image))

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

        if snap:
            array &= mask

        polygons = self.vectorise(array)

        if not snap:
            array2 = self.rasterise(geom.exterior)
            array[array2 != 0] = 0
            polygons += self.vectorise(array)

        self.objectify_shapes(polygons)

        j, i = array.nonzero()
        i += self.x0 - (600)
        j -= self.y1 - (-1560)

        ds = self.file['data']



        summary = sum(ds[:,y,x,:] for (x,y) in zip (i,j))

        import datetime
        x = newdates.astype(datetime.datetime)
        ymin, y, ymax = summary.T * (25**2) / (100**2) # pixels -> hectares

        plt.plot(x, y)
        plt.ylabel('water (hectares)')
        plt.gca().fill_between(x, ymin, ymax, alpha=0.5)
        import io
        f = io.BytesIO()
        plt.savefig(f, format='svg')
        self.image = str(f.getvalue(), 'utf-8')
        plt.close()


