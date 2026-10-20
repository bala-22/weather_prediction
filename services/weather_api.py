"""
Weather API service for fetching data from Open-Meteo.
Open-Meteo provides free historical weather data without API keys.
Focus: Tiruchirappalli, Tamil Nadu, India
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import numpy as np
from datetime import datetime, timedelta
from config import Config


# Open-Meteo base URLs
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_current_weather(lat=None, lon=None):
    """
    Fetch current weather.
    Uses OpenWeatherMap if API key is available, otherwise Open-Meteo.
    Returns dict with current weather data or None on failure.
    """
    lat = lat or Config.DEFAULT_LATITUDE
    lon = lon or Config.DEFAULT_LONGITUDE

    # Try OpenWeatherMap first (user's API key)
    if Config.WEATHER_API_KEY:
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": Config.WEATHER_API_KEY,
                "units": "metric",
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "temperature": data["main"]["temp"],
                "min_temperature": data["main"]["temp_min"],
                "max_temperature": data["main"]["temp_max"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "cloud_cover": data.get("clouds", {}).get("all", 0),
                "weather_condition": data["weather"][0]["description"].title(),
                "rainfall": data.get("rain", {}).get("1h", 0),
                "location": data.get("name", Config.DEFAULT_LOCATION),
                "source": "OpenWeatherMap",
            }
        except (requests.RequestException, KeyError, ValueError):
            pass

    # Fallback to Open-Meteo
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,weather_code,cloud_cover,pressure_msl,"
            "wind_speed_10m,wind_direction_10m"
        ),
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(CURRENT_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        weather_code = current.get("weather_code", 0)

        return {
            "temperature": current.get("temperature_2m", 0),
            "min_temperature": current.get("temperature_2m", 0),
            "max_temperature": current.get("temperature_2m", 0),
            "humidity": current.get("relative_humidity_2m", 0),
            "pressure": current.get("pressure_msl", 1010),
            "wind_speed": current.get("wind_speed_10m", 0),
            "cloud_cover": current.get("cloud_cover", 0),
            "weather_condition": _weather_code_to_text(weather_code),
            "rainfall": current.get("precipitation", 0),
            "location": Config.DEFAULT_LOCATION,
            "source": "Open-Meteo",
        }
    except (requests.RequestException, KeyError, ValueError):
        return None


def fetch_historical_weather(dt, lat=None, lon=None):
    """
    Fetch historical weather for a specific date from Open-Meteo archive.
    Returns dict with weather data or None.
    """
    lat = lat or Config.DEFAULT_LATITUDE
    lon = lon or Config.DEFAULT_LONGITUDE

    date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": (
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,wind_speed_10m_max,pressure_msl_mean,"
            "cloud_cover_mean"
        ),
        "hourly": "relative_humidity_2m",
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        hourly = data.get("hourly", {})

        # Compute daily average humidity from hourly data
        humidity_values = hourly.get("relative_humidity_2m", [])
        avg_humidity = (
            np.mean([h for h in humidity_values if h is not None])
            if humidity_values else 50.0
        )

        return {
            "date": date_str,
            "temperature": daily.get("temperature_2m_mean", [0])[0] or 0,
            "min_temperature": daily.get("temperature_2m_min", [0])[0] or 0,
            "max_temperature": daily.get("temperature_2m_max", [0])[0] or 0,
            "humidity": round(float(avg_humidity), 1),
            "rainfall": daily.get("precipitation_sum", [0])[0] or 0,
            "pressure": daily.get("pressure_msl_mean", [1010])[0] or 1010,
            "wind_speed": daily.get("wind_speed_10m_max", [0])[0] or 0,
            "cloud_cover": daily.get("cloud_cover_mean", [0])[0] or 0,
            "weather_condition": _classify_from_rainfall(
                daily.get("precipitation_sum", [0])[0] or 0
            ),
        }
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return None


def fetch_bulk_historical(start_date, end_date, lat=None, lon=None):
    """
    Fetch bulk historical data for a date range using Open-Meteo archive.
    Downloads all days in one API call (efficient).
    Returns a pandas DataFrame.
    """
    lat = lat or Config.DEFAULT_LATITUDE
    lon = lon or Config.DEFAULT_LONGITUDE

    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
    end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)

    # Check cache first
    cache_path = os.path.join(
        Config.RAW_DATA_DIR,
        f"trichy_{start_str}_to_{end_str}.csv",
    )
    if os.path.exists(cache_path):
        print(f"  Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    print(f"  Fetching real weather data from Open-Meteo...")
    print(f"  Period: {start_str} to {end_str}")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        "daily": (
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,wind_speed_10m_max,pressure_msl_mean,"
            "cloud_cover_mean"
        ),
        "hourly": "relative_humidity_2m",
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"  API request failed: {e}")
        return pd.DataFrame()

    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    dates = daily.get("time", [])
    if not dates:
        print("  No data returned from API")
        return pd.DataFrame()

    # Compute daily average humidity from hourly data
    humidity_values = hourly.get("relative_humidity_2m", [])
    hourly_times = hourly.get("time", [])

    # Group hourly humidity by date
    humidity_by_date = {}
    for i, ht in enumerate(hourly_times):
        date_key = ht[:10]
        val = humidity_values[i] if i < len(humidity_values) else None
        if val is not None:
            humidity_by_date.setdefault(date_key, []).append(val)

    daily_humidity = {}
    for date_key, vals in humidity_by_date.items():
        daily_humidity[date_key] = round(np.mean(vals), 1)

    records = []
    for i, date_str in enumerate(dates):
        temp_mean = daily.get("temperature_2m_mean", [0])[i] or 0
        temp_min = daily.get("temperature_2m_min", [0])[i] or 0
        temp_max = daily.get("temperature_2m_max", [0])[i] or 0
        rainfall = daily.get("precipitation_sum", [0])[i] or 0
        wind_speed = daily.get("wind_speed_10m_max", [0])[i] or 0
        pressure = daily.get("pressure_msl_mean", [1010])[i] or 1010
        cloud_cover = daily.get("cloud_cover_mean", [0])[i] or 0
        humidity = daily_humidity.get(date_str, 50.0)

        records.append({
            "date": date_str,
            "location": "Tiruchirappalli",
            "temperature": round(float(temp_mean), 1),
            "min_temperature": round(float(temp_min), 1),
            "max_temperature": round(float(temp_max), 1),
            "humidity": round(float(humidity), 1),
            "rainfall": round(float(rainfall), 1),
            "pressure": round(float(pressure), 1),
            "wind_speed": round(float(wind_speed), 1),
            "cloud_cover": round(float(cloud_cover), 1),
            "weather_condition": _classify_from_rainfall(rainfall),
        })

    df = pd.DataFrame(records)

    # Save to cache
    os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} days of real data to {cache_path}")

    return df


def fetch_7day_forecast(lat=None, lon=None):
    """
    Fetch 7-day forecast from Open-Meteo for Tiruchirappalli.
    Returns list of daily forecast dicts or None on failure.
    """
    lat = lat or Config.DEFAULT_LATITUDE
    lon = lon or Config.DEFAULT_LONGITUDE

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": (
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,precipitation_probability_max,"
            "wind_speed_10m_max,cloud_cover_mean,"
            "weather_code"
        ),
        "hourly": "relative_humidity_2m",
        "timezone": "Asia/Kolkata",
        "forecast_days": 8,
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        hourly = data.get("hourly", {})
        dates = daily.get("time", [])

        # Compute daily average humidity from hourly data
        humidity_values = hourly.get("relative_humidity_2m", [])
        hourly_times = hourly.get("time", [])
        humidity_by_date = {}
        for i, ht in enumerate(hourly_times):
            date_key = ht[:10]
            val = humidity_values[i] if i < len(humidity_values) else None
            if val is not None:
                humidity_by_date.setdefault(date_key, []).append(val)
        daily_humidity = {}
        for dk, vals in humidity_by_date.items():
            daily_humidity[dk] = round(np.mean(vals), 1)

        forecasts = []
        for i, date_str in enumerate(dates):
            if i == 0:
                continue
            weather_code = daily.get("weather_code", [0])[i] or 0
            forecasts.append({
                "date": date_str,
                "temperature": round(daily.get("temperature_2m_mean", [30])[i] or 30, 1),
                "min_temperature": round(daily.get("temperature_2m_min", [25])[i] or 25, 1),
                "max_temperature": round(daily.get("temperature_2m_max", [35])[i] or 35, 1),
                "humidity": daily_humidity.get(date_str, 50.0),
                "rainfall": round(daily.get("precipitation_sum", [0])[i] or 0, 1),
                "wind_speed": round(daily.get("wind_speed_10m_max", [0])[i] or 0, 1),
                "cloud_cover": round(daily.get("cloud_cover_mean", [0])[i] or 0, 1),
                "weather_condition": _weather_code_to_text(weather_code),
                "rain_probability": daily.get("precipitation_probability_max", [0])[i] or 0,
            })

        return forecasts

    except (requests.RequestException, KeyError, ValueError):
        return None


def _weather_code_to_text(code):
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "Clear Sky",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing Rime Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        71: "Slight Snow",
        73: "Moderate Snow",
        75: "Heavy Snow",
        80: "Slight Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",
        85: "Slight Snow Showers",
        86: "Heavy Snow Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with Slight Hail",
        99: "Thunderstorm with Heavy Hail",
    }
    return codes.get(code, "Unknown")


def _classify_from_rainfall(rainfall):
    """Classify weather condition based on rainfall amount."""
    if rainfall >= 7.5:
        return "Heavy Rain"
    elif rainfall >= 2.5:
        return "Moderate Rain"
    elif rainfall >= Config.RAIN_THRESHOLD:
        return "Light Rain"
    else:
        return "Clear"
