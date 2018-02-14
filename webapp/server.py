import flask
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.redirect(flask.url_for('static', filename='client.html'))

