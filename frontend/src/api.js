// api.js

const BASE_URL = 'http://localhost:5001';

// Enhanced error handling
class APIError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

const handleResponse = async (response) => {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Unknown API error' }));
    throw new APIError(errorData.error || `HTTP error status: ${response.status}`, response.status);
  }
  // Handle empty responses from DELETE requests, which are successful but have no body
  if (response.status === 200 && response.headers.get('content-length') === '0') {
    return { success: true };
  }
  return response.json();
};

// Create a new route
export async function fetchRoute(start, end, config = {}) {
  try {
    const response = await fetch(`${BASE_URL}/route`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start, end, grid: config }),
    });
    return await handleResponse(response);
  } catch (error) {
    console.error('Route fetch error:', error);
    throw error;
  }
}

// Get the status of an existing route
export async function fetchRouteStatus(routeId) {
  try {
    // MODIFIED: Endpoint points to the base route URL with GET method
    const response = await fetch(`${BASE_URL}/route/${routeId}`);
    return await handleResponse(response);
  } catch (error) {
    console.error('Route status fetch error:', error);
    throw error;
  }
}

// Force an update on an existing route
export async function forceRouteUpdate(routeId) {
  try {
    // MODIFIED: Endpoint points to the base route URL with POST method
    const response = await fetch(`${BASE_URL}/route/${routeId}`, {
      method: 'POST',
    });
    return await handleResponse(response);
  } catch (error) {
    console.error('Route update error:', error);
    throw error;
  }
}

// NEW: Delete a route from the backend
export async function deleteRoute(routeId) {
  try {
    const response = await fetch(`${BASE_URL}/route/${routeId}`, {
      method: 'DELETE',
    });
    return await handleResponse(response);
  } catch (error) {
    console.error('Route delete error:', error);
    throw error;
  }
}


// --- Other API functions remain the same ---

// Weather API functions
export async function fetchWeather(lat, lon) {
  try {
    const response = await fetch(`${BASE_URL}/weather?lat=${lat}&lon=${lon}`);
    return await handleResponse(response);
  } catch (error) {
    console.error('Weather fetch error:', error);
    throw error;
  }
}

// ML prediction function
export async function fetchPrediction(conditions) {
  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(conditions),
    });
    return await handleResponse(response);
  } catch (error) {
    console.error('Prediction fetch error:', error);
    throw error;
  }
}

// System health check
export async function fetchHealthStatus() {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    return await handleResponse(response);
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
}