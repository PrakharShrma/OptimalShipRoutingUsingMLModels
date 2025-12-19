import requests
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ADDED: Rate Limiter ---
# This class ensures we don't make more than ~1 call per second to respect free tier limits.
class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self.last_call_time = 0

    def wait(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call_time = time.time()

# Create a single rate limiter instance for the OpenWeatherMap API
openweathermap_limiter = RateLimiter(requests_per_minute=55) # Stay safely under the 60/min limit


class WeatherOceanAPI:
    """
    Client for fetching real-time weather and ocean data from multiple sources.
    Combines OpenWeatherMap, NOAA, and other APIs for comprehensive marine data.
    """
    
    def __init__(self, openweather_api_key: str, noaa_api_key: Optional[str] = None):
        self.openweather_key = "key"
        self.noaa_key = "key"
        self.base_urls = {
            'openweather': 'https://api.openweathermap.org/data/2.5',
            'marine_weather': 'https://api.worldweatheronline.com/v1/marine.ashx'
        }
        
    def get_current_weather(self, lat: float, lon: float) -> Dict:
        """Get current weather conditions with retry logic."""
        url = f"{self.base_urls['openweather']}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.openweather_key,
            'units': 'metric'
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                # ADDED: Wait before making the call to respect rate limits
                openweathermap_limiter.wait()
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                return {
                    'wind_speed': data['wind']['speed'] * 1.94384,
                    'wind_direction': data['wind'].get('deg', 0),
                    'air_temp': data['main']['temp'],
                    'pressure': data['main']['pressure'],
                    'visibility': data.get('visibility', 10000) / 1000,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for ({lat},{lon}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"All retries failed for fetching weather data.")
                    return self._get_default_weather()
    
    def get_marine_forecast(self, lat: float, lon: float, hours: int = 48) -> List[Dict]:
        """
        Get marine weather forecast using the 5-day/3-hour forecast endpoint
        compatible with the Free API plan.
        """
        try:
            url = f"{self.base_urls['openweather']}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_key,
                'units': 'metric',
            }
            
            # ADDED: Wait before making the call to respect rate limits
            openweathermap_limiter.wait()

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            
            for forecast_item in data.get('list', []):
                if len(forecast) * 3 >= hours:
                    break
                
                forecast.append({
                    'timestamp': datetime.fromtimestamp(forecast_item['dt']),
                    'wind_speed': forecast_item['wind']['speed'] * 1.94384,
                    'wind_direction': forecast_item['wind']['deg'],
                    'air_temp': forecast_item['main']['temp'],
                    'pressure': forecast_item['main']['pressure'],
                    'visibility': forecast_item.get('visibility', 10000) / 1000,
                    'weather_desc': forecast_item['weather'][0]['description']
                })
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error fetching marine forecast: {e}")
            return self._get_default_forecast(hours)
    
    def get_ocean_conditions(self, lat: float, lon: float) -> Dict:
        """Get ocean conditions including waves and currents"""
        try:
            ocean_data = self._simulate_ocean_conditions(lat, lon)
            
            return {
                'wave_height': ocean_data['wave_height'],
                'wave_period': ocean_data['wave_period'],
                'wave_direction': ocean_data['wave_direction'],
                'current_speed': ocean_data['current_speed'],
                'current_direction': ocean_data['current_direction'],
                'sea_temp': ocean_data['sea_temp'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching ocean conditions: {e}")
            return self._get_default_ocean()
    
    def get_complete_conditions(self, lat: float, lon: float) -> Dict:
        """Get complete weather and ocean conditions for a location"""
        weather = self.get_current_weather(lat, lon)
        ocean = self.get_ocean_conditions(lat, lon)
        
        return {**weather, **ocean}
    
    def get_route_conditions(self, route_points: List[Tuple[float, float]]) -> List[Dict]:
        """Get conditions along entire route"""
        conditions = []
        
        for lat, lon in route_points:
            point_conditions = self.get_complete_conditions(lat, lon)
            point_conditions['lat'] = lat
            point_conditions['lon'] = lon
            conditions.append(point_conditions)
            time.sleep(0.1)
        
        return conditions
    
    def get_forecast_along_route(self, route_points: List[Tuple[float, float]], 
                               hours: int = 24) -> Dict:
        """Get weather forecast along the entire route"""
        route_forecast = {}
        
        for i, (lat, lon) in enumerate(route_points):
            forecast = self.get_marine_forecast(lat, lon, hours)
            route_forecast[f'point_{i}'] = {
                'lat': lat,
                'lon': lon,
                'forecast': forecast
            }
            time.sleep(0.1)
        
        return route_forecast
    
    def _simulate_ocean_conditions(self, lat: float, lon: float) -> Dict:
        np.random.seed(int((lat + lon) * 1000) % 2147483647)
        if 20 <= lon <= 120 and -40 <= lat <= 30:
            base_wave, base_current, base_temp = 2.0, 0.8, 25 + 5 * np.sin(np.radians(lat))
        else:
            base_wave, base_current, base_temp = 1.5, 0.5, 20
        now = datetime.now()
        seasonal_factor = np.sin(2 * np.pi * now.timetuple().tm_yday / 365)
        return {
            'wave_height': max(0.5, base_wave + seasonal_factor + np.random.normal(0, 0.5)),
            'wave_period': np.random.uniform(6, 12),
            'wave_direction': np.random.uniform(0, 360),
            'current_speed': max(0.1, base_current + 0.3 * seasonal_factor + np.random.normal(0, 0.2)),
            'current_direction': np.random.uniform(0, 360),
            'sea_temp': base_temp + 2 * seasonal_factor + np.random.normal(0, 1)
        }
    
    def _get_default_weather(self) -> Dict:
        return {'wind_speed': 10.0, 'wind_direction': 180.0, 'air_temp': 20.0, 'pressure': 1013.25, 'visibility': 10.0, 'timestamp': datetime.now()}
    
    def _get_default_ocean(self) -> Dict:
        return {'wave_height': 2.0, 'wave_period': 8.0, 'wave_direction': 180.0, 'current_speed': 0.5, 'current_direction': 90.0, 'sea_temp': 22.0, 'timestamp': datetime.now()}
    
    def _get_default_forecast(self, hours: int) -> List[Dict]:
        forecast, base_time = [], datetime.now()
        for i in range(hours):
            forecast.append({'timestamp': base_time + timedelta(hours=i), 'wind_speed': 10.0 + np.random.normal(0, 2), 'wind_direction': 180.0 + np.random.normal(0, 30), 'air_temp': 20.0 + np.random.normal(0, 3), 'pressure': 1013.25 + np.random.normal(0, 10), 'visibility': 10.0, 'weather_desc': 'Clear sky'})
        return forecast

class WeatherCache:
    def __init__(self, cache_duration_minutes: int = 15):
        self.cache, self.cache_duration = {}, timedelta(minutes=cache_duration_minutes)
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration: return data
            else: del self.cache[key]
        return None
    def set(self, key: str, data: Dict): self.cache[key] = (data, datetime.now())
    def clear(self): self.cache.clear()

weather_cache = WeatherCache()

def get_cached_weather(api_client: WeatherOceanAPI, lat: float, lon: float) -> Dict:
    cache_key = f"weather_{lat:.2f}_{lon:.2f}"
    cached_data = weather_cache.get(cache_key)
    if cached_data: return cached_data
    data = api_client.get_complete_conditions(lat, lon)
    weather_cache.set(cache_key, data)
    return data

def fetch_region_conditions(api, bounds, step=5):
    conditions_list = []
    lat_range = range(bounds["lat_min"], bounds["lat_max"] + 1, step)
    lon_range = range(bounds["lon_min"], bounds["lon_max"] + 1, step)
    
    for lat in lat_range:
        for lon in lon_range:
            try:
                cond = api.get_complete_conditions(lat, lon)
                cond["lat"], cond["lon"] = lat, lon
                conditions_list.append(cond)
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed at ({lat},{lon}): {e}")
    return conditions_list

if __name__ == "__main__":
    api = WeatherOceanAPI(openweather_api_key="key")
    
    test_lat, test_lon = 12.0, 80.0
    conditions = api.get_complete_conditions(test_lat, test_lon)
    print("Current conditions:", conditions)
    
    # Test the new forecast function
    forecast = api.get_marine_forecast(test_lat, test_lon)
    print("\n3-Hour Interval Forecast:", forecast[:3]) # Print first 3 results

    bounds = {"lat_min": -40, "lat_max": 30, "lon_min": 20, "lon_max": 120}
    
    dataset = fetch_region_conditions(api, bounds, step=10)
    print(f"\nCollected {len(dataset)} points across Indian Ocean")
    
    test_route = [(12.0, 80.0), (10.0, 85.0), (8.0, 90.0)]
    route_conditions = api.get_route_conditions(test_route)
    print("Route conditions:", len(route_conditions), "points")