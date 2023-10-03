from flask import Flask
from api.train_api import train_api
from api.predict_api import predict_api


app: Flask = Flask(__name__)
app.register_blueprint(train_api, url_prefix='/v1')
app.register_blueprint(predict_api, url_prefix='/v1')
