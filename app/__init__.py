# app/__init__.py

from flask import Flask
from dotenv import load_dotenv
import os

def create_app():
    # Load environment variables from .env file
    load_dotenv()

    app = Flask(__name__)

    # Secret key for sessions (you can generate a random one)
    app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "defaultsecret")

    # Import and register routes
    from .routes import main
    app.register_blueprint(main)

    return app
