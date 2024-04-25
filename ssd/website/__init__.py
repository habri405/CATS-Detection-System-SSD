from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    UPLOAD_FOLDER = r'C:\Users\Habzizi\Desktop\CatsTrafficSign\website\static\uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    DETECTION_FOLDER = r'C:\Users\Habzizi\Desktop\CatsTrafficSign\website\static\detections'
    app.config['DETECTION_FOLDER'] = DETECTION_FOLDER


    db.init_app(app)

    from .auth import auth

    from .ssd import ssd

    app.register_blueprint(auth, url_prefix='/')

    app.register_blueprint(ssd, url_prefix='/')

    from .models import User
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
