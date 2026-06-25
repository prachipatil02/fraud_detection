# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base configuration"""
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
    
    # Model paths
    MODEL_PATH = os.path.join(BASE_DIR, 'data/models/xgb_model.pkl')
    SCALER_PATH = os.path.join(BASE_DIR, 'data/models/scaler.pkl')
    FEATURES_PATH = os.path.join(BASE_DIR, 'data/models/feature_names.pkl')

class DevelopmentConfig(Config):
    """Development configuration"""
    SQLALCHEMY_DATABASE_URI = 'sqlite:///fraud_detection.db'
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://user:password@localhost/fraud_detection'
    )
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}