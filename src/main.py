from flask import Flask
from api.run_api import run_api


app: Flask = Flask(__name__)
app.register_blueprint(run_api, url_prefix='/v1')
