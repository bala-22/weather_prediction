# Weather Prediction System.

An intelligent weather forecasting system for **Tiruchirappalli** that uses machine learning algorithms to predict rainfall and analyze weather patterns.

## Features

- **6-Month Weather Analysis** - Analyze historical weather data with ML model evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- **1-Week Sub-Area Prediction** - Get 7-day weather forecasts for 12 Tiruchirappalli sub-areas using trained ML models
- **Interactive Charts** - Visualize weather trends with Chart.js (temperature, humidity, rainfall, rain probability)
- **ML Model Training** - Compare Random Forest, Logistic Regression, and XGBoost algorithms
- **Responsive Dashboard** - Works on desktop, tablet, and mobile

## Technologies

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| Machine Learning | scikit-learn, XGBoost, pandas, NumPy |
| Database | PostgreSQL |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Chart.js |
| Weather Data | OpenWeatherMap API |
| Model Persistence | Joblib |

## Project Structure

```
weather_prediction/
├── app.py                     # Flask application (routes & API)
├── config.py                  # Configuration (env vars, settings)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── database_schema.sql        # PostgreSQL database schema
├── README.md                  # This file
│
├── ml/                        # Machine Learning pipeline
│   ├── train_model.py         # Model training script
│   ├── predict.py             # Prediction module
│   └── evaluate.py            # Evaluation metrics
│
├── services/                  # Business logic services
│   ├── weather_api.py         # OpenWeatherMap API integration
│   └── data_processor.py      # Data cleaning & feature engineering
│
├── templates/                 # HTML templates
│   ├── base.html              # Base layout
│   ├── index.html             # Home page
│   ├── analysis.html          # 6-Month Analysis page
│   ├── prediction.html        # 1-Week Prediction page
│   └── about.html             # About page
│
├── static/                    # Static assets
│   ├── css/style.css          # Main stylesheet
│   └── js/
│       ├── analysis.js        # Analysis page charts & logic
│       └── prediction.js      # Prediction page charts & logic
│
├── data/                      # Weather data storage
│   ├── raw/                   # Raw downloaded data
│   ├── processed/             # Processed & cached results
│   └── sample/                # Sample dataset
│
└── models/                    # Trained ML model files
    ├── weather_model.pkl
    └── scaler.pkl
```

## Requirements

- Python 3.8+
- PostgreSQL 13+
- pip (Python package manager)

## Installation & Setup

### 1. Clone / Copy the project

```bash
cd weather_prediction
```

### 2. Create Python virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. PostgreSQL Database Setup

Open PostgreSQL and run the schema:

```bash
psql -U postgres -f database_schema.sql
```

Or manually execute the SQL commands in `database_schema.sql` using pgAdmin or any PostgreSQL client.

### 5. Configure Environment Variables

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
```

Edit `.env` and fill in your values:

```env
# For sample data mode (no API key needed):
DATA_MODE=sample

# For real weather data:
DATA_MODE=api
WEATHER_API_KEY=your_openweathermap_api_key_here

# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=weather_prediction
```

### 6. Get a Weather API Key (Optional)

1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get a free API key
3. Add it to your `.env` file
4. Set `DATA_MODE=api`

**Note:** The application works without an API key using sample data mode.

### 7. Train the ML Model

```bash
python ml/train_model.py
```

This will:
- Load weather data (sample or API-based)
- Clean and preprocess the data
- Engineer features
- Train Random Forest, Logistic Regression, and XGBoost models
- Evaluate on test set
- Save the best model to `models/weather_model.pkl`

### 8. Run the Application

```bash
python app.py
```

The application will start at `http://localhost:5000`

### 9. Open the Website

Open your browser and navigate to:
```
http://localhost:5000
```

## Quick Start (Sample Data Mode)

If you just want to run the project quickly without any API key or database:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ml/train_model.py
python app.py
```

Open `http://localhost:5000` in your browser.

## How the 6-Month Analysis Works

1. **Data Collection**: Historical weather data for the past 6 months is loaded (from sample dataset or API)
2. **Preprocessing**: Data is cleaned, missing values handled, and features are engineered
3. **Feature Engineering**: Creates 22 features including temporal, lag, and rolling average features
4. **Chronological Split**: Data is split 80/20 maintaining time order (no data leakage)
5. **Model Training**: Random Forest, Logistic Regression, and XGBoost are trained
6. **Evaluation**: Best model is evaluated on the test set:
   - **TP** (True Positive): Model predicted rain, actual was rain
   - **TN** (True Negative): Model predicted no rain, actual was no rain
   - **FP** (False Positive): Model predicted rain, actual was no rain
   - **FN** (False Negative): Model predicted no rain, actual was rain
7. **Metrics**: Accuracy, Precision, Recall, F1-Score are calculated from the confusion matrix

**Rain Classification**: A day is "Rain" if rainfall >= 0.1 mm (configurable in config.py)


<img width="400" alt="Screenshot (60)" src="https://github.com/user-attachments/assets/d2e78b23-ae48-4b20-9e5c-083117bd2dc7" />


## How the 1-Week Prediction Works

1. User selects a Tiruchirappalli sub-area from the dropdown
2. The system loads the trained ML model
3. For each of the next 7 days:
   - Base weather data is estimated using seasonal climate averages for the sub-area
   - 22 features are constructed matching the training format
   - The ML model predicts rain/no-rain with probability
4. Results are displayed in a table with temperature, humidity, rainfall, wind, condition, and rain probability
5. Charts show rain probability and temperature forecast

**Note**: When the API is configured, real forecast data from OpenWeatherMap is used as input to the ML model, providing more accurate predictions.

## Sub-Areas

The system supports predictions for these Tiruchirappalli sub-areas:

- Srirangam, Thillai Nagar, Cantonment, Woraiyur
- KK Nagar, Tennur, Thiruverumbur, Manachanallur
- Samayapuram, Lalgudi, Musiri, Manapparai

New areas can be added by updating the `SUB_AREAS` dictionary in `config.py`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure virtual environment is activated and packages are installed |
| `Connection refused` for PostgreSQL | Check PostgreSQL is running and credentials in `.env` are correct |
| `No weather data available` | Run `python ml/train_model.py` first to generate sample data |
| Charts not loading | Check internet connection (Chart.js is loaded from CDN) |
| Port already in use | Change port in `app.py` or stop other applications on port 5000 |
| `Model not found` | Run `python ml/train_model.py` to train and save the model |

## License

## 📄 License

This project is licensed under the MIT License.

