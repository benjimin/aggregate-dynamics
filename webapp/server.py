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
    assert data['type'] == 'Polygon'
    poly = shapely.geometry.shape(data)

    res = 100000
    x0,y0,x1,y1 = np.asarray(poly.bounds) / res
    x0,y0,x1,y1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)

    affine = rasterio.Affine(res, 0, x0*res, 0, -res, y1*res)
    array = rasterio.features.rasterize(
                [poly],
                out_shape=(y1-y0, x1-x0),
                transform=affine,
                all_touched=True)
    shapes = list(rasterio.features.shapes(array, mask=array, transform=affine))
    assert len(shapes) == 1
    exterior = shapes[0]

    array2 = rasterio.features.rasterize(
                [poly.exterior],
                out_shape=(y1-y0, x1-x0),
                transform=affine,
                all_touched=True)
    array[array2 != 0] = 0
    shapes = list(rasterio.features.shapes(array, mask=array, transform=affine))
    interiors = shapes

    z = flask.jsonify(interiors[0][0])
    #z = flask.jsonify(exterior[0])
    z.status_code = 200
    return z