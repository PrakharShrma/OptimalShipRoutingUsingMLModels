import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class ShipPerformancePredictor:
    def __init__(self, sequence_length=10, feature_dim=12):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_columns = [
            'wind_speed', 'wind_direction', 'wave_height', 'wave_period',
            'current_speed', 'current_direction', 'sea_temp', 'air_temp',
            'pressure', 'visibility', 'ship_heading', 'ship_speed_prev'
        ]
        self.target_columns = ['ship_speed', 'fuel_consumption']
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.sequence_length, self.feature_dim)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        model.compile(optimizer="adam", loss="mse")
        self.model = model
        return model

    def predict(self, features):
        features = np.array(features)
        if features.ndim == 2:
            features = np.expand_dims(features, axis=0)
        
        # --- START: CORRECTED SCALING LOGIC ---
        # The model expects data in the same scale it was trained on.
        # This was the source of the bug.
        batch_size, seq_len, num_features = features.shape
        features_reshaped = features.reshape(-1, num_features)
        
        # Use the loaded scaler to transform the input data.
        # Note: We use .transform(), not .fit_transform(), for prediction.
        scaled_features = self.scaler_features.transform(features_reshaped)
        scaled_features_reshaped = scaled_features.reshape(batch_size, seq_len, num_features)
        
        # Get the scaled prediction from the model.
        scaled_prediction = self.model.predict(scaled_features_reshaped, verbose=0)
        
        # Inverse transform the prediction to get the real-world values.
        prediction = self.scaler_target.inverse_transform(scaled_prediction)
        # --- END: CORRECTED SCALING LOGIC ---

        result = prediction[0]
        return {
            "speed": float(result[0]),
            "fuel_consumption": float(result[1])
        }
    
    def save_model(self, filepath='models/ship_performance_model'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(f"{filepath}.h5")
        with open(f"{filepath}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'scaler_features': self.scaler_features,
                'scaler_target': self.scaler_target,
            }, f)
        print(f"Model and scalers saved to {filepath}")
    
    def load_model(self, filepath='models/ship_performance_model'):
        self.model = tf.keras.models.load_model(f"{filepath}.h5", compile=False)
        scaler_path = f"{filepath}_scalers.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                data = pickle.load(f)
                self.scaler_features = data['scaler_features']
                self.scaler_target = data['scaler_target']
        else:
             # If scalers don't exist, we must fit them with placeholder data
             # so the .transform() method doesn't fail.
             print("Scalers not found. Fitting with placeholder data.")
             placeholder_features = np.zeros((1, self.feature_dim))
             placeholder_target = np.zeros((1, len(self.target_columns)))
             self.scaler_features.fit(placeholder_features)
             self.scaler_target.fit(placeholder_target)

        print(f"Model and scalers loaded from {filepath}")

# The training function is left for standalone use and is not called by the main app.
def train_model(weather_api, bounds, step=5):
    from weather_api import fetch_region_conditions 
    
    predictor = ShipPerformancePredictor()
    
    print("Fetching regional weather data for training...")
    region_data = fetch_region_conditions(weather_api, bounds, step=step)
    if not region_data:
        print("No weather data fetched. Cannot train model.")
        return None

    features_list, target_list = [], []
    for cond in region_data:
        features_list.append([
            cond.get("wind_speed", 10), cond.get("wind_direction", 180),
            cond.get("wave_height", 2), cond.get("wave_period", 8),
            cond.get("current_speed", 0.5), cond.get("current_direction", 90),
            cond.get("sea_temp", 22), cond.get("air_temp", 20),
            cond.get("pressure", 1013), cond.get("visibility", 10),
            180, 12 
        ])
        true_speed = max(5, 15 - 0.1 * cond.get("wind_speed", 0) - 0.5 * cond.get("wave_height", 0))
        true_fuel  = 20 + 0.2 * cond.get("wind_speed", 0) + 0.5 * cond.get("wave_height", 0)
        target_list.append([true_speed, true_fuel])

    # Fit scalers on the full dataset and then transform
    scaled_features = predictor.scaler_features.fit_transform(features_list)
    scaled_targets = predictor.scaler_target.fit_transform(target_list)

    X, y = [], []
    for i in range(len(scaled_features)):
        seq = np.tile(scaled_features[i], (predictor.sequence_length, 1))
        X.append(seq)
        y.append(scaled_targets[i])
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training dataset built: {X.shape[0]} samples")
    predictor.build_model()
    predictor.model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    predictor.save_model()
    return predictor

if __name__ == "__main__":
    from weather_api import WeatherOceanAPI

    api = WeatherOceanAPI(openweather_api_key="key")
    bounds = {"lat_min": -40, "lat_max": 30, "lon_min": 20, "lon_max": 120}
    predictor = train_model(api, bounds, step=5)
    if predictor:
        print("Training complete, model saved.")