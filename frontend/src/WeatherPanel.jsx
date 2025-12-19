import React from 'react'

export default function WeatherPanel({ weatherData, mlPrediction, systemStatus }) {
  const weatherPoints = Object.values(weatherData);

  const formatCondition = (value, unit, decimals = 1) => value != null ? `${value.toFixed(decimals)}${unit}` : 'N/A';

  const getConditionColor = (type, value) => {
    if (value == null) return '#6c757d';
    switch (type) {
      case 'wind': return value > 25 ? '#dc3545' : value > 15 ? '#ffc107' : '#28a745';
      case 'waves': return value > 4 ? '#dc3545' : value > 2 ? '#ffc107' : '#28a745';
      case 'visibility': return value < 2 ? '#dc3545' : value < 5 ? '#ffc107' : '#28a745';
      default: return '#6c757d';
    }
  };

  const ConditionBadge = ({ value, unit, label, type }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
      <div style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: getConditionColor(type, value) }} />
      <span>{label}: <strong>{formatCondition(value, unit)}</strong></span>
    </div>
  );

  return (
    <div style={{ backgroundColor: 'white', borderBottom: '1px solid #dee2e6' }}>
      {mlPrediction && systemStatus.ml_model && (
        <div style={{ padding: 16, borderBottom: '1px solid #e9ecef' }}>
          <h6 style={{ margin: '0 0 12px 0', fontSize: 15, fontWeight: 600, color: '#343a40' }}>
            🤖 AI Performance Forecast
          </h6>
          <div style={{ marginBottom: 12, fontSize: 12, padding: '8px 12px', backgroundColor: '#f8f9fa', borderRadius: 8 }}>
            For point: <strong>{mlPrediction.lat.toFixed(3)}, {mlPrediction.lon.toFixed(3)}</strong>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div style={{ padding: 12, backgroundColor: '#e3f2fd', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#1976d2', marginBottom: 4 }}>Predicted Speed</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#1976d2' }}>{mlPrediction.speed?.toFixed(1)} kts</div>
            </div>
            <div style={{ padding: 12, backgroundColor: '#fff3e0', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#f57c00', marginBottom: 4 }}>Fuel Consumption</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#f57c00' }}>{mlPrediction.fuel_consumption?.toFixed(1)} t/day</div>
            </div>
          </div>
          {mlPrediction.weather && (
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#495057' }}>Based on Conditions:</div>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                <ConditionBadge type="wind" value={mlPrediction.weather.wind_speed} unit=" kts" label="Wind" />
                <ConditionBadge type="waves" value={mlPrediction.weather.wave_height} unit="m" label="Waves" />
                <ConditionBadge type="visibility" value={mlPrediction.weather.visibility} unit="km" label="Visibility" />
              </div>
            </div>
          )}
        </div>
      )}

      {weatherPoints.length > 0 && (
        <div style={{ padding: 16 }}>
          <h6 style={{ margin: '0 0 12px 0', fontSize: 15, fontWeight: 600, color: '#343a40' }}>
            🌊 Route Weather Summary
          </h6>
          <div style={{ maxHeight: 250, overflowY: 'auto', paddingRight: 8 }}>
            {weatherPoints.map((point, i) => (
              <div key={i} style={{ marginBottom: 8, padding: '10px 12px', backgroundColor: '#f8f9fa', borderRadius: 8, fontSize: 12, border: '1px solid #e9ecef' }}>
                <div style={{ fontWeight: 600, color: '#007bff', marginBottom: 8 }}>
                  📍 {point.lat?.toFixed(3)}, {point.lon?.toFixed(3)}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <ConditionBadge type="wind" value={point.wind_speed} unit=" kts" label="Wind" />
                  <ConditionBadge type="waves" value={point.wave_height} unit="m" label="Waves" />
                  <ConditionBadge type="temp" value={point.sea_temp} unit="°C" label="Sea Temp" />
                  <ConditionBadge type="visibility" value={point.visibility} unit="km" label="Visibility" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}