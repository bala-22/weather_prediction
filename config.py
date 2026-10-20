"""
Configuration module for Weather Prediction application.
Loads environment variables and defines application settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # Flask settings
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    # Database settings
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "weather_prediction")

    # Weather API
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
    DATA_MODE = os.getenv("DATA_MODE", "openmeteo")  # "openmeteo", "api", or "sample"
    DATA_SOURCE = os.getenv("DATA_SOURCE", "openmeteo")  # "openmeteo" = free, no key needed

    # Location: Tiruchirappalli, Tamil Nadu, India
    DEFAULT_LATITUDE = float(os.getenv("DEFAULT_LATITUDE", 10.7905))
    DEFAULT_LONGITUDE = float(os.getenv("DEFAULT_LONGITUDE", 78.7047))
    DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Tiruchirappalli")

    # Rain classification
    RAIN_THRESHOLD = float(os.getenv("RAIN_THRESHOLD", 0.1))  # mm

    # ML Model paths
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "weather_model.pkl")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

    # Data paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")

    # Analysis period
    ANALYSIS_MONTHS = 6

    # Tiruchirappalli sub-areas with approximate coordinates
    SUB_AREAS = {
        "Srirangam": {"lat": 10.8625, "lon": 78.6850},
        "Thillai Nagar": {"lat": 10.7930, "lon": 78.6980},
        "Cantonment": {"lat": 10.7850, "lon": 78.7100},
        "Woraiyur": {"lat": 10.8200, "lon": 78.6950},
        "KK Nagar": {"lat": 10.7880, "lon": 78.7170},
        "Tennur": {"lat": 10.7910, "lon": 78.7210},
        "Thiruverumbur": {"lat": 10.7650, "lon": 78.7350},
        "Golden Rock": {"lat": 10.7856, "lon": 78.7233},
        "Abhishekapuram": {"lat": 10.8030, "lon": 78.6970},
        "Ariyamangalam": {"lat": 10.8050, "lon": 78.7150},
        "Karumandapam": {"lat": 10.7867, "lon": 78.6708},
        "Thiruvanaikoil": {"lat": 10.8540, "lon": 78.7090},
        "Vayalur": {"lat": 10.8510, "lon": 78.6420},
        "Puthur": {"lat": 10.7750, "lon": 78.7050},
        "Palakarai": {"lat": 10.8100, "lon": 78.6880},
        "Melapudur": {"lat": 10.7980, "lon": 78.7030},
        "Kajamalai": {"lat": 10.7950, "lon": 78.6950},
        "Edamalaipattipudur": {"lat": 10.7700, "lon": 78.7200},
        "Chatram": {"lat": 10.8000, "lon": 78.6850},
        "Panjapur": {"lat": 10.7420, "lon": 78.7100},
        "Subramaniapuram": {"lat": 10.7810, "lon": 78.7240},
        "Mannarpuram": {"lat": 10.7900, "lon": 78.7000},
        "Rockfort": {"lat": 10.8280, "lon": 78.6980},
        "Ponmalai": {"lat": 10.7730, "lon": 78.7060},
        "Thuvakudi": {"lat": 10.7620, "lon": 78.7500},
        "Manachanallur": {"lat": 10.8350, "lon": 78.6800},
        "Samayapuram": {"lat": 10.8500, "lon": 78.6700},
        "Lalgudi": {"lat": 10.8730, "lon": 78.6190},
        "Musiri": {"lat": 10.9350, "lon": 78.4470},
        "Manapparai": {"lat": 10.6010, "lon": 78.4280},
        "Thuraiyur": {"lat": 10.9100, "lon": 78.6300},
        "Thottiyam": {"lat": 10.8700, "lon": 78.4500},
        "Pullambadi": {"lat": 10.8170, "lon": 78.5580},
        "Uppiliyapuram": {"lat": 10.8980, "lon": 78.5100},
        "Kallakudi": {"lat": 10.8770, "lon": 78.5960},
        "Sirugamani": {"lat": 10.8440, "lon": 78.8370},
        "Koothappar": {"lat": 10.7970, "lon": 78.7140},
        "Pappakurichi": {"lat": 10.8450, "lon": 78.6260},
        "Navalpattu": {"lat": 10.8360, "lon": 78.7550},
        "Pirattiyur": {"lat": 10.8300, "lon": 78.6530},
    }

    @classmethod
    def get_db_config(cls):
        """Return database configuration as a dictionary."""
        return {
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "database": cls.DB_NAME,
        }
