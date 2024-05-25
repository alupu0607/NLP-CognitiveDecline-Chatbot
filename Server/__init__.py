'''
This is how to run the server:
python -m flask --app Server run --port 8000 --debug
'''
from flask import Flask
from flask_cors import CORS
from Server import pages
import os
def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(pages.bp)
    return app