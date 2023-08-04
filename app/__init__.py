from app.routes import main
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.config["SECRET_KEY"] = "qwertyuiop1234"
    app.register_blueprint(main)
    return app
