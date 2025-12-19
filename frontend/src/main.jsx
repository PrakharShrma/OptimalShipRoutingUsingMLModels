// main.jsx

import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './global.css'; // NEW: Import a global CSS file

createRoot(document.getElementById('root')).render(<App />);