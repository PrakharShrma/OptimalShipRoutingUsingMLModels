# Optimal Ship Routing System 🚢🌊

## Project Overview
The **Optimal Ship Routing System** is a backend application that computes the safest and most efficient sea route between two geographic locations. It combines **geospatial analysis, weather data, and machine learning–based ship performance prediction** to support intelligent maritime navigation while avoiding land areas and unfavorable ocean conditions.

This system is designed for academic, research, and prototype-level maritime decision support applications.

---

## Key Capabilities
- Computes optimal sea routes using **A\* pathfinding**
- Avoids land using **GeoJSON country polygons + Shapely**
- Considers **weather and ocean conditions** during routing
- Predicts ship speed and fuel consumption using **ML models**
- Automatically updates routes when conditions change
- Exposes a **REST API** for frontend integration

---

## Tech Stack
- **Python**
- **Flask & Flask-CORS**
- **Shapely (Geospatial Processing)**
- **NumPy**
- **OpenWeather API**
- **Machine Learning (LSTM-based predictor)**
- **GeoJSON Datasets**

---

## How It Works (High Level)
1. User provides start and end coordinates  
2. System generates a spatial grid over the sea region  
3. Land areas are excluded using polygon masking  
4. Weather and ocean data influence route cost  
5. A\* algorithm computes the optimal path  
6. ML model estimates performance metrics  
7. Route and analytics are returned via API  

---

## API Highlights
- `POST /route` – Compute optimal ship route
- `GET /route/<id>/status` – Check route updates
- `GET /weather` – Fetch weather data
- `POST /predict` – Predict ship performance
- `GET /health` – System status

---

## Setup & Run
```bash
pip install flask flask-cors shapely numpy requests
python app.py
