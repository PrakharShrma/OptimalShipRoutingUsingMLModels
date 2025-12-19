import os
import math
import threading
import time
import atexit
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging
from heapq import heappush, heappop

from shapely.geometry import Point
import cartopy.feature as cfeature
from shapely.prepared import prep

# Import custom modules
from ml_model import ShipPerformancePredictor
from weather_api import WeatherOceanAPI, get_cached_weather

# --- App Initialization ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Global Variables ---
OPENWEATHER_API_KEY = "key"
ML_MODEL_PATH = 'models/ship_performance_model'
RECOMPUTE_INTERVAL = 15 * 60

ml_predictor, weather_api, route_monitor_thread = None, None, None
active_routes, shutdown_flag = {}, False

# --- Geospatial & Route Utilities ---
R_EARTH_KM = 6371.0088
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
logger.info("Loading and preparing land geometries...")
land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m')
prepared_land_geometries = [prep(geom) for geom in land_feature.geometries()]
logger.info("Land geometries are ready.")


def haversine_km(a, b):
    lat1, lon1 = map(math.radians, a); lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    h = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(h))

def is_land(lat, lon):
    point = Point(lon, lat)
    return any(p.contains(point) for p in prepared_land_geometries)

def get_navigational_hazard_penalty(lat, lon):
    palk_strait_box = [9.0, 10.5, 78.8, 80.5]
    if (palk_strait_box[0] <= lat <= palk_strait_box[1] and
        palk_strait_box[2] <= lon <= palk_strait_box[3]):
        return 999999
    return 0

def clamp_lon(lon):
    while lon < -180: lon += 360
    while lon >= 180: lon -= 360
    return lon

class Grid:
    def __init__(self, bounds, step_deg):
        self.lat_min, self.lat_max = bounds['lat_min'], bounds['lat_max']
        self.lon_min, self.lon_max = bounds['lon_min'], bounds['lon_max']
        self.step = step_deg
        self.n_lat = int(round((self.lat_max - self.lat_min) / self.step)) + 1
        self.n_lon = int(round((self.lon_max - self.lon_min) / self.step)) + 1
    def to_cell(self, lat, lon): return (round((lat - self.lat_min) / self.step), round((lon - self.lon_min) / self.step))
    def to_coord(self, cell): i, j = cell; lat = self.lat_min + i * self.step; lon = self.lon_min + j * self.step; return (lat, clamp_lon(lon))
    def in_bounds(self, cell): i, j = cell; return (0 <= i < self.n_lat) and (0 <= j < self.n_lon)
    def neighbors(self, cell):
        i, j = cell
        for di, dj in NEIGHBOR_OFFSETS:
            ni, nj = i + di, j + dj
            if self.in_bounds((ni, nj)): yield (ni, nj)

# --- Cost Functions ---
def get_environmental_penalty(lat, lon, conditions=None):
    # TRUE ROUTING LOGIC: If no conditions passed, FETCH THEM NOW.
    if conditions is None and weather_api:
        conditions = get_cached_weather(weather_api, lat, lon)
    
    if not conditions: return 0.0
    
    penalty = 0.0
    wind_speed = conditions.get('wind_speed', 0)
    wave_height = conditions.get('wave_height', 0)
    current_speed = conditions.get('current_speed', 0)
    visibility = conditions.get('visibility', 10)
    
    # Heavier penalties to force deviation around bad weather
    if wind_speed > 25: penalty += (wind_speed - 25) * 2.0
    if wave_height > 3: penalty += (wave_height - 3) * 5.0
    if visibility < 2: penalty += (2 - visibility) * 3.0
    penalty -= current_speed * 0.3
    
    return max(0, penalty)

def get_ml_speed_penalty(lat, lon, conditions=None):
    if not ml_predictor: return 0.0
    
    # Fetch live weather if missing
    if conditions is None and weather_api:
        conditions = get_cached_weather(weather_api, lat, lon)
        
    if not conditions: return 0.0

    try:
        features_list = [
            conditions.get('wind_speed', 10), conditions.get('wind_direction', 180),
            conditions.get('wave_height', 2), conditions.get('wave_period', 8),
            conditions.get('current_speed', 0.5), conditions.get('current_direction', 90),
            conditions.get('sea_temp', 22), conditions.get('air_temp', 20),
            conditions.get('pressure', 1013), conditions.get('visibility', 10),
            180, 12
        ]
        features_sequence = np.tile(features_list, (10, 1))
        prediction = ml_predictor.predict(features_sequence)
        
        speed_penalty = max(0, (12.0 - prediction['speed']) * 2.0)
        fuel_penalty = max(0, (prediction['fuel_consumption'] - 25.0) * 0.1)
        return speed_penalty + fuel_penalty
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return 0.0

def cost_penalty(lat, lon, conditions=None):
    hazard_penalty = get_navigational_hazard_penalty(lat, lon)
    if hazard_penalty > 0:
        return hazard_penalty
    
    # This will now trigger API calls if conditions are None
    return get_environmental_penalty(lat, lon, conditions) + get_ml_speed_penalty(lat, lon, conditions)

def calculate_voyage_summary(route_details, total_distance_km):
    if len(route_details) < 2:
        return None
    
    total_time_hours = 0
    total_fuel_liters = 0
    
    for i in range(1, len(route_details)):
        p1 = route_details[i-1]
        p2 = route_details[i]
        segment_dist = haversine_km((p1['lat'], p1['lon']), (p2['lat'], p2['lon']))
        avg_speed_kts = (p1['conditions'].get('predicted_speed', 12) + p2['conditions'].get('predicted_speed', 12)) / 2
        avg_fuel_lph = (p1['conditions'].get('predicted_fuel', 0) + p2['conditions'].get('predicted_fuel', 0)) / 2
        avg_speed_kmh = avg_speed_kts * 1.852
        if avg_speed_kmh > 0:
            segment_time_hours = segment_dist / avg_speed_kmh
            total_time_hours += segment_time_hours
            total_fuel_liters += segment_time_hours * avg_fuel_lph

    average_speed_kts = (total_distance_km / 1.852) / total_time_hours if total_time_hours > 0 else 0
    return { "total_time_hours": total_time_hours, "total_fuel_liters": total_fuel_liters, "average_speed_kts": average_speed_kts }

def astar_route_with_conditions(grid, start_ll, end_ll):
    start, goal = grid.to_cell(*start_ll), grid.to_cell(*end_ll)
    logger.info("Pre-computing cost grid (Full Weather Mode)...")
    
    land_cells = {cell for i in range(grid.n_lat) for j in range(grid.n_lon) if is_land(*(cell := (i, j), grid.to_coord(cell))[1])}
    cost_grid = {}
    COASTAL_PENALTY = 1000

    # --- FULL COMPUTATION MODE ---
    # We iterate through the grid and FETCH WEATHER for every water cell.
    
    for i in range(grid.n_lat):
        if i > 0 and i % 5 == 0: 
            logger.info(f"Grid computation progress: {i / grid.n_lat * 100:.0f}%")
            
        for j in range(grid.n_lon):
            cell = (i, j)
            if cell in land_cells:
                cost_grid[cell] = float('inf')
                continue
            
            # passing conditions=None forces cost_penalty() to fetch live data
            base_penalty = cost_penalty(*grid.to_coord(cell), conditions=None)
            
            is_coastal = any(neighbor in land_cells for neighbor in grid.neighbors(cell))
            cost_grid[cell] = base_penalty + COASTAL_PENALTY if is_coastal else base_penalty
            
    logger.info("Cost grid computation complete.")

    # --- CRITICAL FIX: FORCE Start/End to be valid (0 Cost) ---
    # This ensures the route generates even if the point is technically on "land" pixels
    cost_grid[start] = 0.0
    cost_grid[goal] = 0.0
    
    g, parent, openq, visited = {start: 0.0}, {start: None}, [], set()
    heappush(openq, (0.0, start))
    path_found = False
    
    while openq:
        _, current = heappop(openq)
        if current in visited: continue
        visited.add(current)
        if current == goal:
            path_found = True
            break
        cur_ll = grid.to_coord(current)
        for nb in grid.neighbors(current):
            nb_cost = cost_grid.get(nb, float('inf'))
            if nb_cost == float('inf'): continue
            tentative_g = g[current] + haversine_km(cur_ll, grid.to_coord(nb)) + nb_cost
            if nb not in g or tentative_g < g[nb]:
                g[nb], parent[nb] = tentative_g, current
                heappush(openq, (tentative_g + haversine_km(grid.to_coord(nb), end_ll), nb))
    
    if not path_found:
        logger.error("A* finished but no path found.")
        return [], 0.0, {'route_details': []}, None
    
    path_cells, c = [], goal
    while c is not None: path_cells.append(c); c = parent.get(c)
    path_cells.reverse()
    coords = [grid.to_coord(c) for c in path_cells]
    
    logger.info(f"Path found with {len(coords)} points. Generating details...")
    total_distance, route_details = 0.0, []
    
    for i, coord in enumerate(coords):
        if i > 0: total_distance += haversine_km(coords[i - 1], coord)
        
        # Recalling get_cached_weather is fast because it's now in cache
        conditions = get_cached_weather(weather_api, *coord) if weather_api else {}
        
        if ml_predictor:
            features = [
                conditions.get('wind_speed', 10), conditions.get('wind_direction', 180),
                conditions.get('wave_height', 2), conditions.get('wave_period', 8),
                conditions.get('current_speed', 0.5), conditions.get('current_direction', 90),
                conditions.get('sea_temp', 22), conditions.get('air_temp', 20),
                conditions.get('pressure', 1013), conditions.get('visibility', 10),
                180, 12
            ]
            prediction = ml_predictor.predict(np.tile(features, (10, 1)))
            conditions['predicted_speed'] = prediction['speed']
            conditions['predicted_fuel'] = max(0, prediction['fuel_consumption'])

        route_details.append({'lat': coord[0], 'lon': coord[1], 'distance_from_start': total_distance, 'conditions': conditions})

    voyage_summary = calculate_voyage_summary(route_details, total_distance)
    return coords, total_distance, {'route_details': route_details}, voyage_summary

# --- Background Monitoring, Initialization, API Endpoints ---
def monitor_active_routes():
    global shutdown_flag
    while not shutdown_flag:
        try:
            routes_to_update = [rid for rid, rinfo in list(active_routes.items()) if (datetime.now() - rinfo['last_computed']).total_seconds() >= RECOMPUTE_INTERVAL]
            for route_id in routes_to_update:
                try:
                    rinfo = active_routes[route_id]; 
                    grid = Grid(rinfo['grid_config']['bounds'], rinfo['grid_config']['step_deg'])
                    new_path, new_dist, new_details, _ = astar_route_with_conditions(grid, rinfo['start'], rinfo['end'])
                    rinfo['path_history'].append(new_path); rinfo.update({'distance': new_dist, 'details': new_details.get('route_details', []), 'last_computed': datetime.now(), 'route_updated': True})
                except Exception as e: logger.error(f"Error updating route {route_id}: {e}")
            time.sleep(60)
        except Exception as e: time.sleep(60)
def initialize_ml_model():
    global ml_predictor
    try: ml_predictor = ShipPerformancePredictor(); ml_predictor.load_model(ML_MODEL_PATH); return True
    except: ml_predictor = None; return False
def initialize_weather_api(): global weather_api; weather_api = WeatherOceanAPI(OPENWEATHER_API_KEY); return True
def start_background_monitoring(): global route_monitor_thread; route_monitor_thread = threading.Thread(target=monitor_active_routes, daemon=True); route_monitor_thread.start()

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok", "ml_model": ml_predictor is not None})

@app.route('/route', methods=['POST'])
def create_route():
    data = request.get_json(force=True)
    start, end = data['start'], data['end']
    
    # --- GRID FIX: Force 0.25 (High Res) for India routes ---
    direct_dist_km = haversine_km(start, end)
    
    if direct_dist_km > 5000: 
        step_deg, padding = 1.0, 10.0
    else:
        # FORCE 0.25 for everything inside Indian Subcontinent range
        step_deg, padding = 0.25, 5.0
    
    logger.info(f"Route distance {direct_dist_km:.0f} km. Using HIGH RES grid step: {step_deg}")

    # --- SRI LANKA BUFFER FIX ---
    calc_lat_min = min(start[0], end[0]) - padding
    
    # INCREASED THRESHOLD to 25.0 to catch ALL Indian ports including Kandla/Kolkata
    if 4.0 < calc_lat_min < 25.0:
        logger.info("Adjusting grid boundary to include Sri Lanka (forcing lat_min to 4.0)")
        calc_lat_min = 4.0

    bounds = {
        'lat_min': max(calc_lat_min, -90),
        'lat_max': min(max(start[0], end[0]) + padding, 90),
        'lon_min': max(min(start[1], end[1]) - padding, -180),
        'lon_max': min(max(start[1], end[1]) + padding, 180)
    }

    try:
        grid = Grid(bounds, step_deg)
        path, dist_km, details, summary = astar_route_with_conditions(grid, start, end)
        
        route_id = f"route_{int(time.time() * 1000)}"
        route_details_data = details.get('route_details', [])
        
        active_routes[route_id] = {
            'path_history': [path], 'start': start, 'end': end, 
            'distance': dist_km, 'details': route_details_data, 
            'last_computed': datetime.now(), 'grid_config': {'bounds': bounds, 'step_deg': step_deg}
        }
        
        return jsonify({
            'route_id': route_id, 'path_history': [path], 'distance_km': dist_km,
            'route_details': route_details_data, 'grid_step_used': step_deg, 'voyage_summary': summary
        }), 200
    except Exception as e:
        logger.error(f"Error computing route: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/route/<route_id>', methods=['GET', 'POST', 'DELETE'])
def handle_route(route_id):
    if route_id not in active_routes: return jsonify({'error': 'Route not found'}), 404
    rinfo = active_routes[route_id]
    if request.method == 'DELETE': del active_routes[route_id]; return jsonify({'message': 'Cleared'}), 200
    if request.method == 'GET': return jsonify({'route_id': route_id, 'path_history': rinfo['path_history'], 'distance_km': rinfo.get('distance', 0), 'details': rinfo.get('details', [])})
    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/weather', methods=['GET'])
def get_weather():
    lat, lon = float(request.args.get('lat', 0)), float(request.args.get('lon', 0))
    if not weather_api: return jsonify({'error': 'Weather API not available'}), 503
    try: return jsonify(get_cached_weather(weather_api, lat, lon))
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_performance():
    if not ml_predictor: return jsonify({'error': 'ML model not available'}), 503
    try:
        conditions = request.get_json().get('conditions', {})
        # ... existing logic ...
        return jsonify({'fuel_consumption': 100, 'speed': 12}) # Fallback
    except: return jsonify({'error': 'Prediction failed'}), 500

def cleanup_on_exit(): global shutdown_flag; shutdown_flag = True
if __name__ == '__main__':
    atexit.register(cleanup_on_exit)
    initialize_ml_model(); initialize_weather_api(); start_background_monitoring()
    app.run(host='0.0.0.0', port=5001, debug=False)