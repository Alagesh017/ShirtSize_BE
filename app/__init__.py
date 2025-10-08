from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:alagesh17@localhost/shirt_app'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    CORS(app)
    db.init_app(app)
    migrate.init_app(app, db)

    from app.model import MeasurementRecord

    from app.routes.measurement_route import measurement_bp
    from app.routes.upload_record import upload_bp

    app.register_blueprint(measurement_bp)
    app.register_blueprint(upload_bp)

    return app
