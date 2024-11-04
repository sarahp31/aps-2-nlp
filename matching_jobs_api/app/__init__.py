from flask import Flask
from app.logging_config import setup_logging


def create_app():
    app = Flask(__name__)

    with app.app_context():
        from app.routes import register_routes
        register_routes(app)

    setup_logging()


    return app
