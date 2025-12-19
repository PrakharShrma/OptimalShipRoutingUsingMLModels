import React, { useState, useEffect, useCallback } from 'react';
import MapView from './MapView';
import { deleteRoute, fetchRoute, fetchRouteStatus, forceRouteUpdate, fetchWeather, fetchPrediction, fetchHealthStatus } from './api';
import LandingPage from './LandingPage';
import RouteMonitor from './RouteMonitor';
import WeatherPanel from './WeatherPanel';

// --- ADDED: Voyage Summary Component ---
const VoyageSummaryBar = ({ summary }) => {
  if (!summary) return null;

  const summaryStyle = {
    position: 'absolute',
    top: '20px',
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 1001,
    background: 'rgba(255, 255, 255, 0.9)',
    padding: '8px 16px',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
    display: 'flex',
    gap: '24px',
    fontSize: '14px',
    fontWeight: 600,
    color: '#343a40'
  };

  const itemStyle = { textAlign: 'center' };
  const valueStyle = { color: '#007bff', display: 'block', fontSize: '16px' };
  const labelStyle = { fontSize: '11px', textTransform: 'uppercase', color: '#6c757d' };

  return (
    <div style={summaryStyle}>
      <div style={itemStyle}>
        <span style={valueStyle}>{summary.total_time_hours.toFixed(1)}</span>
        <span style={labelStyle}>Total Hours</span>
      </div>
      <div style={itemStyle}>
        <span style={valueStyle}>{summary.total_fuel_liters.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
        <span style={labelStyle}>Total Fuel (L)</span>
      </div>
      <div style={itemStyle}>
        <span style={valueStyle}>{summary.average_speed_kts.toFixed(1)}</span>
        <span style={labelStyle}>Avg. Speed (kts)</span>
      </div>
    </div>
  );
};


const SidebarPlaceholder = () => {
    const Feature = ({ icon, title, description }) => (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', marginBottom: '16px' }}>
            <div style={{ fontSize: '20px', color: '#007bff', marginTop: '2px' }}>{icon}</div>
            <div>
                <h5 style={{ margin: '0 0 4px 0', fontSize: '14px', color: '#343a40' }}>{title}</h5>
                <p style={{ margin: 0, fontSize: '12px', color: '#6c757d', lineHeight: 1.5 }}>{description}</p>
            </div>
        </div>
    );
    return (
        <div style={{ padding: 16, borderBottom: '1px solid #dee2e6', backgroundColor: '#fff' }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: 18, fontWeight: 600, color: '#007bff' }}>Optimal Ship Routing</h4>
            <p style={{ fontSize: 13, marginTop: 0, marginBottom: '24px', color: '#6c757d' }}>Welcome! Click on the map to select a start and end point for your voyage.</p>
            <div>
                <Feature icon="🌦️" title="Real-Time Weather" description="Routes are optimized using live wind, wave, and current data." />
                <Feature icon="🤖" title="AI Performance Model" description="Our ML model predicts speed and fuel consumption for maximum efficiency." />
                <Feature icon="📡" title="Dynamic Monitoring" description="Active routes are automatically updated as weather conditions change." />
            </div>
            <div style={{ textAlign: 'center', marginTop: 12, padding: '12px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
                <div style={{ width: 32, height: 32, border: '4px solid #dee2e6', borderTop: '4px solid #007bff', borderRadius: '50%', margin: '0 auto', animation: 'spin 1s linear infinite' }} />
                <p style={{ marginTop: 8, fontSize: 12, color: '#6c757d', margin: 0 }}>Waiting for route selection...</p>
            </div>
            <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        </div>
    );
};

const SidebarLoader = () => (
    <div style={{ padding: 16, textAlign: 'center' }}>
        <div style={{ width: 32, height: 32, border: '4px solid #dee2e6', borderTop: '4px solid #007bff', borderRadius: '50%', margin: '0 auto', animation: 'spin 1s linear infinite' }} />
        <p style={{ marginTop: 8, fontSize: 13, color: '#495057' }}>Computing route...</p>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
    </div>
);

export default function App() {
    const [start, setStart] = useState(null);
    const [end, setEnd] = useState(null);
    const [pathHistory, setPathHistory] = useState([]);
    const [distance, setDistance] = useState(null);
    const [gridStep, setGridStep] = useState(0.25);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentRouteId, setCurrentRouteId] = useState(null);
    const [routeDetails, setRouteDetails] = useState([]);
    const [weatherData, setWeatherData] = useState({});
    const [mlPrediction, setMlPrediction] = useState(null);
    const [monitoringEnabled, setMonitoringEnabled] = useState(false);
    const [routeUpdates, setRouteUpdates] = useState([]);
    const [systemStatus, setSystemStatus] = useState({ ml_model: false, weather_api: false });
    const [isStarted, setIsStarted] = useState(false);
    const [isPanelOpen, setIsPanelOpen] = useState(true);
    // ADDED: New state for the voyage summary
    const [voyageSummary, setVoyageSummary] = useState(null);

    const checkSystemHealth = useCallback(async () => {
        try {
            const health = await fetchHealthStatus();
            setSystemStatus({ ml_model: health.ml_model, weather_api: health.weather_api });
        } catch (e) { console.error('Health check failed:', e) }
    }, []);

    useEffect(() => {
        checkSystemHealth();
    }, [checkSystemHealth]);

    const checkRouteStatus = useCallback(async () => {
        if (!currentRouteId) return;
        try {
            const status = await fetchRouteStatus(currentRouteId);
            if (status.route_updated) {
                setPathHistory(status.path_history || []);
                setDistance(status.distance_km?.toFixed(1));
                setRouteDetails(status.details || []);
                setRouteUpdates(prev => [{ timestamp: new Date(), message: 'Route updated due to changing conditions', type: 'info' }, ...prev.slice(0, 4)]);
            }
        } catch (e) { console.error('Route status check failed:', e) }
    }, [currentRouteId]);
    
    useEffect(() => {
        let interval = null;
        if (currentRouteId && monitoringEnabled) {
            interval = setInterval(checkRouteStatus, 30000);
        }
        return () => { if (interval) clearInterval(interval) };
    }, [currentRouteId, monitoringEnabled, checkRouteStatus]);

    const fetchRouteWeather = useCallback(async (routePath) => {
        try {
            const samplePoints = routePath.filter((_, i) => i % Math.ceil(routePath.length / 10) === 0);
            const weatherPromises = samplePoints.map(([lat, lon]) => fetchWeather(lat, lon));
            const weatherResults = await Promise.all(weatherPromises);
            const weatherMap = {};
            weatherResults.forEach((w, i) => {
                const [lat, lon] = samplePoints[i];
                weatherMap[`${lat.toFixed(2)}_${lon.toFixed(2)}`] = { lat, lon, ...w };
            });
            setWeatherData(weatherMap);
        } catch (e) { console.error('Weather fetch failed:', e) }
    }, []);

    const computeRoute = useCallback(async () => {
        if (!start || !end) return;
        setLoading(true); 
        setError(null);
        // ADDED: Clear previous summary
        setVoyageSummary(null);
        try {
            const res = await fetchRoute(start, end, { step_deg: gridStep });
            setCurrentRouteId(res.route_id);
            setPathHistory(res.path_history || []);
            setDistance(res.distance_km.toFixed(1));
            setRouteDetails(res.route_details || []);
            setMonitoringEnabled(true);
            // ADDED: Set new summary data from API response
            setVoyageSummary(res.voyage_summary || null);
            
            if (res.grid_step_used) {
                setGridStep(res.grid_step_used);
            }

            const currentPath = res.path_history[res.path_history.length - 1];
            if (systemStatus.weather_api && currentPath?.length > 0) {
                fetchRouteWeather(currentPath);
            }
        } catch (e) { setError(e.message) } finally { setLoading(false) }
    }, [start, end, gridStep, systemStatus.weather_api, fetchRouteWeather]);
    
    const forceUpdate = useCallback(async () => {
        if (!currentRouteId) return;
        setLoading(true);
        try {
            const result = await forceRouteUpdate(currentRouteId);
            setPathHistory(result.path_history || []);
            setDistance(result.distance_km.toFixed(1));
            const currentPath = result.path_history[result.path_history.length - 1];
             if (systemStatus.weather_api && currentPath?.length > 0) {
                fetchRouteWeather(currentPath);
            }
            setRouteUpdates(prev => [{ timestamp: new Date(), message: 'Route manually updated', type: 'success' }, ...prev.slice(0, 4)]);
        } catch (e) { setError(e.message) } finally { setLoading(false) }
    }, [currentRouteId, systemStatus.weather_api, fetchRouteWeather]);
    
    const clearRoute = async () => {
        if (currentRouteId) {
            try { await deleteRoute(currentRouteId); } 
            catch (e) { console.error("Failed to clear route on backend:", e.message); }
        }
        setStart(null); setEnd(null); setPathHistory([]); setDistance(null);
        setCurrentRouteId(null); setRouteDetails([]); setMonitoringEnabled(false);
        setRouteUpdates([]); setMlPrediction(null); setWeatherData({}); setError(null);
        // ADDED: Clear summary data
        setVoyageSummary(null);
    };
    
    const getPredictionForPoint = useCallback(async (lat, lon) => {
        if (!systemStatus.ml_model) return;
        try {
            const weather = await fetchWeather(lat, lon);
            const prediction = await fetchPrediction({
                conditions: {
                    wind_speed: weather.wind_speed, wind_direction: weather.wind_direction, wave_height: weather.wave_height, wave_period: weather.wave_period,
                    current_speed: weather.current_speed, current_direction: weather.current_direction, sea_temp: weather.sea_temp, air_temp: weather.air_temp,
                    pressure: weather.pressure, visibility: weather.visibility, ship_heading: 180, ship_speed_prev: 12
                }
            });
            setMlPrediction({ lat, lon, ...prediction, weather });
        } catch (e) { console.error('Prediction failed:', e) }
    }, [systemStatus.ml_model]);

    return (
        <div style={{ height: '100vh', width: '100vw', position: 'relative' }}>
            <div style={{ transform: isStarted ? 'translateY(-100%)' : 'translateY(0%)', transition: 'transform 0.8s ease-in-out', pointerEvents: isStarted ? 'none' : 'auto', width: '100%', height: '100%', position: 'absolute', zIndex: 20000 }}>
                <LandingPage onGetStarted={() => setIsStarted(true)} />
            </div>

            <div style={{ display: 'flex', width: '100%', height: '100%' }}>
                <div style={{ flex: 1, position: 'relative' }}>
                    <MapView
                        isPanelOpen={isPanelOpen}
                        start={start} end={end}
                        setStart={setStart} setEnd={setEnd}
                        pathHistory={pathHistory}
                        gridStep={gridStep} setGridStep={setGridStep}
                    />
                    
                    {/* ADDED: Render the summary bar */}
                    <VoyageSummaryBar summary={voyageSummary} />

                    <div style={{ position: 'absolute', zIndex: 1000, bottom: 20, left: '50%', transform: 'translateX(-50%)', background: 'white', padding: 16, borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.15)', display: 'flex', flexDirection: 'column', gap: 10, minWidth: 450 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: 14 }}>
                            <div><strong>Start:</strong> {start ? start.map(n => n.toFixed(3)).join(', ') : 'Click on map'}</div>
                            <div><strong>End:</strong> {end ? end.map(n => n.toFixed(3)).join(', ') : 'Click on map'}</div>
                        </div>
                        <div style={{ display: 'flex', gap: 8 }}>
                            <button onClick={computeRoute} disabled={!start || !end || loading} style={{ flex: 1, padding: '10px 16px', backgroundColor: loading ? '#ccc' : '#007bff', color: 'white', border: 'none', borderRadius: 8, cursor: loading ? 'not-allowed' : 'pointer', fontSize: 14, fontWeight: 600 }}>{loading ? 'Computing...' : 'Compute Route'}</button>
                            {currentRouteId && (<button onClick={forceUpdate} disabled={loading} style={{ padding: '10px 16px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: 8, cursor: loading ? 'not-allowed' : 'pointer', fontSize: 14 }}>Update</button>)}
                            <button onClick={clearRoute} style={{ padding: '10px 16px', backgroundColor: '#6c757d', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14 }}>Clear</button>
                        </div>
                        {distance && (<div style={{ marginTop: 4, textAlign: 'center', fontWeight: 600, fontSize: 15, color: '#343a40' }}>Optimized Distance: <span style={{ color: '#007bff' }}>{distance} km</span></div>)}
                        {error && <div style={{ color: '#dc3545', marginTop: 4, fontSize: 14, textAlign: 'center' }}>{error}</div>}
                    </div>
                </div>

                <button onClick={() => setIsPanelOpen(!isPanelOpen)} style={{ position: 'absolute', top: '50%', right: isPanelOpen ? '400px' : '0px', transform: 'translateY(-50%)', zIndex: 15000, width: '25px', height: '80px', backgroundColor: 'rgba(0, 123, 255, 0.8)', color: 'white', border: 'none', borderTopLeftRadius: '8px', borderBottomLeftRadius: '8px', cursor: 'pointer', writingMode: 'vertical-rl', textOrientation: 'mixed', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px', transition: 'right 0.5s ease' }}>
                    {isPanelOpen ? '◀' : '▶'}
                </button>

                <div style={{ width: isPanelOpen ? 400 : 0, backgroundColor: '#f8f9fa', borderLeft: '1px solid #dee2e6', display: 'flex', flexDirection: 'column', height: '100vh', overflowY: 'auto', overflowX: 'hidden', transition: 'width 0.5s ease', flexShrink: 0 }}>
                    <div style={{ minWidth: 400 }}>
                        {loading && <SidebarLoader />}
                        {!currentRouteId && !loading && <SidebarPlaceholder />}
                        {currentRouteId && (<RouteMonitor routeId={currentRouteId} updates={routeUpdates} monitoringEnabled={monitoringEnabled} onToggleMonitoring={setMonitoringEnabled} />)}
                        {(Object.keys(weatherData).length > 0 || mlPrediction) && (<WeatherPanel weatherData={weatherData} mlPrediction={mlPrediction} systemStatus={systemStatus} />)}
                        
                        {routeDetails.length > 0 && (
                            <div style={{ padding: '16px', borderTop: '1px solid #dee2e6', backgroundColor: '#fff', flexGrow: 1 }}>
                                <h6 style={{ margin: '0 0 12px 0', fontSize: 15, fontWeight: 600, color: '#343a40' }}>
                                    📍 Route Waypoints ({routeDetails.length})
                                </h6>
                                <div style={{ maxHeight: 'calc(100vh - 500px)', overflowY: 'auto', paddingRight: 8 }}>
                                    {routeDetails.map((point, i) => {
                                        const { conditions } = point;
                                        const formatStat = (value, unit, decimals = 1) => 
                                            (value == null || isNaN(value)) ? 'N/A' : `${value.toFixed(decimals)}${unit}`;

                                        return (
                                            <div key={i} style={{ fontSize: 13, marginBottom: 8, padding: '10px 12px', backgroundColor: '#f8f9fa', borderRadius: 8, border: '1px solid #e9ecef' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                                                    <strong style={{ color: '#007bff' }}>Waypoint {i + 1}</strong>
                                                    <span style={{ fontSize: 12, color: '#6c757d' }}>{point.lat.toFixed(3)}, {point.lon.toFixed(3)}</span>
                                                </div>
                                                <div style={{ marginBottom: 8, fontSize: 12 }}>
                                                    <span>Distance: <strong>{point.distance_from_start?.toFixed(1)} km</strong></span>
                                                </div>
                                                {conditions && (
                                                    <div style={{
                                                        paddingTop: 8, borderTop: '1px solid #e9ecef', display: 'grid',
                                                        gridTemplateColumns: '1fr 1fr', gap: '6px 12px', fontSize: 12, color: '#495057'
                                                    }}>
                                                        <span>💨 Wind: <strong>{formatStat(conditions.wind_speed, ' kts')}</strong></span>
                                                        <span>🌊 Waves: <strong>{formatStat(conditions.wave_height, 'm')}</strong></span>
                                                        <span>👁️ Vis: <strong>{formatStat(conditions.visibility, ' km')}</strong></span>
                                                        <span>🚢 Speed: <strong>{formatStat(conditions.predicted_speed, ' kts')}</strong></span>
                                                        <span style={{ gridColumn: 'span 2' }}>⛽ Fuel: <strong>{formatStat(conditions.predicted_fuel, ' L/hr')}</strong></span>
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}