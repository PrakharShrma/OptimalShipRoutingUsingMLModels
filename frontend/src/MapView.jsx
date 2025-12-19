import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Tooltip, useMap, useMapEvents, LayersControl } from 'react-leaflet';
import L from 'leaflet';
import { indianPorts } from './ports';

// Destructure LayersControl components
const { BaseLayer, Overlay } = LayersControl;

// --- Helper component to resize the map ---
function MapResizer({ isPanelOpen }) {
  const map = useMap();
  useEffect(() => {
    const timer = setTimeout(() => map.invalidateSize(), 500);
    return () => clearTimeout(timer);
  }, [isPanelOpen, map]);
  return null;
}

// --- Component to handle general map clicks ---
function ClickToSet({ onSet }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onSet([lat, lng]);
    }
  });
  return null;
}

// --- Custom Icons ---
const defaultMarkerIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

const portIcon = new L.Icon({
    iconUrl: 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iIzAwNTZhYSIgd2lkdGg9IjM2cHgiIGhlaWdodD0iMzZweCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMCA0YzAtMi4yMSAyLjU5LTQgNC00czQgMS43OSA0IDRjMCAyLjQyLTIuNzIgMy41MS00IDQuODEtMS4yOC0xLjMtNC0yLjM5LTQtNC44MXpNMjAgNmgtMnYySDRWM0gydjJoMlYzaDR2MmgyVjNoMnY0aDRWM2gyVjJoLTJWNWgyVjJoLTJWNWgyVjNoLTJWMmgtMnYxaC0yVjNoLTJ2MmgtMnY0aDRjMSAwaDIgMCAyLTF2LTR6TTE4IDEyYy0xLjY4IDAtMy4wNCAxLjEyLTMuNzkgMi42Ny0uMTYuMzEtLjIuNjQtLjIxLjk3SDl2Mmg4djJoLTJWMTZoLTR2LTJoLTJ2MmgtMnYyaDR2MmgydjJoMnYtMmMzLjM0IDAgNi0yLjY2IDYtNnMtMi42Ni02LTYtNnptMCA4aC0ycy0uNTYtMy40NS0yLTMuNDVTMTEuNDYgMjAgMTEuNDYgMjBIMTAuNWMtLjQ4LTIuNzEtMi4yNS01LTUuNS01IDAgMCAxLjM0LTYgNS41LTZTMTggMTIgMTggMTJ6Ii8+PC9zdmc+',
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
});

const waypointIcon = new L.divIcon({
  className: 'waypoint-marker',
  iconSize: [10, 10]
});

// --- MapView Component ---
export default function MapView({ start, end, setStart, setEnd, pathHistory, isPanelOpen }) {
  const apiKey = "key";
  const windLayer = `https://tile.openweathermap.org/map/wind_new/{z}/{x}/{y}.png?appid=${apiKey}`;
  const cloudLayer = `https://tile.openweathermap.org/map/clouds_new/{z}/{x}/{y}.png?appid=${apiKey}`;

  const currentPath = pathHistory.length > 0 ? pathHistory[pathHistory.length - 1] : [];
  
  const handlePointSelect = (point) => {
    if (!start) setStart(point);
    else if (!end) setEnd(point);
    else { setStart(point); setEnd(null); }
  };

  return (
    <MapContainer
      center={[20.5937, 78.9629]}
      zoom={5}
      style={{ height: "100vh", width: "100%" }}
      scrollWheelZoom={true}
    >
      <MapResizer isPanelOpen={isPanelOpen} />

      <LayersControl position="topright">
        <BaseLayer checked name="OpenStreetMap">
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
        </BaseLayer>
        <Overlay checked name="Wind">
          <TileLayer url={windLayer} opacity={0.4} />
        </Overlay>
        <Overlay name="Clouds">
          <TileLayer url={cloudLayer} opacity={0.7} />
        </Overlay>
      </LayersControl>

      <ClickToSet onSet={handlePointSelect} />

      {/* Render selectable port markers */}
      {indianPorts.map(port => (
        <Marker 
          key={port.name} 
          position={[port.lat, port.lng]} 
          icon={portIcon}
          eventHandlers={{ click: (e) => {
            L.DomEvent.stopPropagation(e);
            handlePointSelect([port.lat, port.lng]);
          }}}
        >
          <Tooltip>{port.name}</Tooltip>
        </Marker>
      ))}

      {/* Render start and end markers */}
      {start && <Marker position={start} icon={defaultMarkerIcon} />}
      {end && <Marker position={end} icon={defaultMarkerIcon} />}

      {start && end && (
        <>
          <Polyline positions={[start, end]} pathOptions={{ color: "blue", weight: 2, dashArray: "5, 10" }} />
          
          {currentPath.length > 0 && (
            <Polyline 
              positions={[start, ...currentPath, end]}
              pathOptions={{ color: "#28a745", weight: 5 }} 
            />
          )}

          {currentPath.map((point, index) => (
            <Marker key={`wp-${index}`} position={point} icon={waypointIcon}>
              <Tooltip>Waypoint {index + 1}</Tooltip>
            </Marker>
          ))}
        </>
      )}
      
      <div className="leaflet-bottom leaflet-left" style={{ padding: 8, pointerEvents: 'auto' }}>
        <div style={{ background: "white", padding: 6, borderRadius: 6, fontSize: 12, boxShadow: "0 2px 6px rgba(0,0,0,0.3)" }}>
          <div><span style={{color:"blue"}}>▬</span> GPS Route</div>
          {currentPath.length > 0 && (
            <div><span style={{color:"#28a745"}}>▬</span> Current Optimized Route</div>
          )}
        </div>
      </div>
    </MapContainer>
  );
}