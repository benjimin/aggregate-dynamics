import flask
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.redirect(flask.url_for('static', filename='client.html'))

@app.route('/plot', methods=['POST'])
def receive():
    data = flask.request.get_json()
    assert data['type'] == 'Polygon'
    print(data)
    return repr(data['coordinates'])