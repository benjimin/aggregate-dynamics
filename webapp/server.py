import flask
import shapely.geometry
import numpy as np
import math
import rasterio.features

app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.redirect(flask.url_for('static', filename='client.html'))

@app.route('/plot', methods=['POST'])
def receive():
    data = flask.request.get_json()
    snap = data['snap']
    poly = shapely.geometry.shape(data['geometry'])

    response = flask.jsonify(geom_to_grid(poly, snap=snap))
    response.status_code = 200
    return response

def geom_to_grid(geom, snap=False, res=100000):
    x0,y0,x1,y1 = np.asarray(geom.bounds) / res
    x0,y0,x1,y1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
    affine = rasterio.Affine(res, 0, x0*res, 0, -res, y1*res)

    def rasterise(geom, all_touched=True):
        return rasterio.features.rasterize([geom],
                                           out_shape=(y1-y0, x1-x0),
                                           transform=affine,
                                           all_touched=all_touched)
    def vectorise(array):
        shapes = rasterio.features.shapes(array,
                                          mask=array,
                                          transform=affine)
        return list(geom for (geom, value) in shapes)

    array = rasterise(geom, not snap)
    polygons = vectorise(array)

    if not snap:
        array2 = rasterise(geom.exterior)
        array[array2 != 0] = 0
        polygons += vectorise(array)

    # TODO: if len(..) == 0 i.e. tiny input region -> grab one grid square?
    if len(polygons) == 1:
        return polygons[0]
    else:
        return dict(type='GeometryCollection', geometries=polygons)
