// LandingPage.jsx

import React from 'react';

export default function LandingPage({ onGetStarted }) {
  const styles = {
    container: {
      position: 'absolute',
      top: 0, left: 0,
      width: '100%', height: '100%',
      color: 'white',
      display: 'flex', flexDirection: 'column',
      justifyContent: 'center', alignItems: 'center',
      zIndex: 20000,
      textAlign: 'center',
      padding: '20px',
      // Animated gradient background
      background: 'linear-gradient(45deg, #001f3f, #0074D9, #7FDBFF, #39CCCC)',
      backgroundSize: '400% 400%',
      animation: 'gradientAnimation 15s ease infinite',
    },
    title: {
      fontFamily: `'Georgia', 'Times New Roman', serif`,
      fontSize: 'clamp(3rem, 6vw, 4.5rem)',
      fontWeight: 700,
      marginBottom: '16px',
      letterSpacing: '1px',
      textShadow: '0 4px 15px rgba(0,0,0,0.4)',
    },
    subtitle: {
      fontFamily: `sans-serif`,
      fontSize: 'clamp(1rem, 2vw, 1.25rem)',
      maxWidth: '650px',
      lineHeight: 1.6,
      marginBottom: '40px',
      opacity: 0.9,
    },
    button: {
      fontFamily: `sans-serif`,
      fontSize: '1.1rem',
      fontWeight: 600,
      padding: '14px 32px',
      backgroundColor: 'white',
      color: '#007bff',
      border: 'none',
      borderRadius: '50px',
      cursor: 'pointer',
      transition: 'transform 0.2s ease, box-shadow 0.2s ease',
      boxShadow: '0 4px 20px rgba(0, 123, 255, 0.5)',
    },
    shipIcon: {
        fontSize: '48px',
        marginBottom: '24px',
        opacity: 0.8,
    }
  };

  return (
    <div style={styles.container}>
      {/* Add a style tag for the keyframes animation */}
      <style>{`
        @keyframes gradientAnimation {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
      <div style={styles.shipIcon}>🚢</div>
      <h1 style={styles.title}>Optimal Ship Routing</h1>
      <p style={styles.subtitle}>
        Navigate smarter. Plan efficient sea routes with real-time weather data,
        AI-powered monitoring, and advanced fuel optimization.
      </p>
      <button 
        style={styles.button}
        onClick={onGetStarted}
        onMouseOver={e => e.currentTarget.style.transform = 'scale(1.05)'}
        onMouseOut={e => e.currentTarget.style.transform = 'scale(1)'}
      >
        Get Started
      </button>
    </div>
  );
}