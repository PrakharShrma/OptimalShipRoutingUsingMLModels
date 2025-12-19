import React from 'react'

export default function RouteMonitor({ routeId, updates, monitoringEnabled, onToggleMonitoring }) {
  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    })
  }

  const getUpdateIcon = (type) => {
    switch (type) {
      case 'success': return '✅'; case 'warning': return '⚠️'; case 'error': return '❌';
      default: return 'ℹ️';
    }
  }

  const getUpdateColor = (type) => {
    switch (type) {
      case 'success': return '#28a745'; case 'warning': return '#ffc107'; case 'error': return '#dc3545';
      default: return '#17a2b8';
    }
  }

  return (
    <div style={{ padding: 16, backgroundColor: 'white', borderBottom: '1px solid #dee2e6' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h6 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: '#343a40' }}>
          🗺️ Route Monitor
        </h6>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#6c757d' }}>{monitoringEnabled ? 'Active' : 'Inactive'}</span>
          <button
            onClick={() => onToggleMonitoring(!monitoringEnabled)}
            style={{
              padding: '4px 12px', fontSize: 11, border: 'none', borderRadius: 16,
              backgroundColor: monitoringEnabled ? '#28a745' : '#6c757d', color: 'white', cursor: 'pointer',
              transition: 'background-color 0.3s'
            }}
          >
            {monitoringEnabled ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* Monitoring Status */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16, padding: '10px 12px',
        backgroundColor: monitoringEnabled ? '#eaf7ec' : '#fbebee', borderRadius: 8, fontSize: 13
      }}>
        <div style={{ 
          width: 10, height: 10, borderRadius: '50%',
          backgroundColor: monitoringEnabled ? '#28a745' : '#dc3545',
          animation: monitoringEnabled ? 'pulse 2s infinite' : 'none'
        }} />
        <span style={{ color: monitoringEnabled ? '#155724' : '#721c24', fontWeight: 500 }}>
          {monitoringEnabled ? 'Real-time monitoring is active.' : 'Monitoring is disabled.'}
        </span>
      </div>

      {/* Updates List */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#495057' }}>
          Recent Updates
        </div>
        <div style={{ maxHeight: 200, overflowY: 'auto' }}>
          {updates.length > 0 ? (
            updates.map((update, i) => (
              <div key={i} style={{
                padding: '10px 0', display: 'flex', alignItems: 'flex-start', gap: 10,
                borderBottom: i < updates.length - 1 ? '1px solid #f1f3f4' : 'none'
              }}>
                <span style={{ fontSize: 16, marginTop: 2 }}>{getUpdateIcon(update.type)}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ color: '#343a40', fontWeight: 500, marginBottom: 2, fontSize: 13 }}>
                    {update.message}
                  </div>
                  <div style={{ color: '#6c757d', fontSize: 11 }}>
                    {formatTime(update.timestamp)}
                  </div>
                </div>
                <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: getUpdateColor(update.type), alignSelf: 'center' }} />
              </div>
            ))
          ) : (
            <div style={{ padding: '20px 0', textAlign: 'center', color: '#6c757d', fontSize: 12, backgroundColor: '#f8f9fa', borderRadius: 8 }}>
              {monitoringEnabled ? 'Awaiting updates on route conditions...' : 'Enable monitoring to see live updates.'}
            </div>
          )}
        </div>
      </div>

      {/* CSS for pulse animation */}
      <style>{`
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
      `}</style>
    </div>
  )
}