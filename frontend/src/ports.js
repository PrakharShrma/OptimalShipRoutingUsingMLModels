// // A static list of major Indian ports with coordinates moved slightly offshore
// // to guarantee they fall in a 'water' grid cell for the pathfinding algorithm.
// export const indianPorts = [
//   { name: 'Mumbai Port', lat: 18.925, lng: 72.835 },
//   { name: 'Chennai Port', lat: 13.100, lng: 80.305 },
//    { name: 'Kochi Port (Cochin)', lat: 9.968, lng: 76.240 },
//   { name: 'Visakhapatnam Port', lat: 17.688, lng: 83.305 },
//   { name: 'Kandla Port', lat: 22.975, lng: 70.180 },
//   { name: 'Jawaharlal Nehru Port (Nhava Sheva)', lat: 18.940, lng: 72.945 },
//   { name: 'Mormugao Port', lat: 15.410, lng: 73.785 },
//   { name: 'Paradip Port', lat: 20.265, lng: 86.695 },
//   { name: 'Haldia Port', lat: 22.025, lng: 88.075 },
//   { name: 'Tuticorin Port', lat: 8.810, lng: 78.185 },
//   { name: 'Mangalore Port', lat: 12.890, lng: 74.805 }
// ];
// src/ports.js (or wherever your indianPorts list is)

export const indianPorts = [
  // SAFE OFFSHORE COORDINATES (Pilot Stations)
  
  // Moved West into the sea (approx 15km)
  { name: 'Mumbai Port', lat: 18.910, lng: 72.700 }, 

  // Moved East
  { name: 'Chennai Port', lat: 13.100, lng: 80.400 },

  // Moved West
  { name: 'Kochi Port (Cochin)', lat: 9.960, lng: 76.100 },

  // Moved East
  { name: 'Visakhapatnam Port', lat: 17.680, lng: 83.400 },

  // --- KANDLA FIX ---
  // OLD (Bad): 22.975, 70.180 (This is what is in your screenshot!)
  // NEW (Good): 22.500, 70.000 (This is in the open water)
  { name: 'Kandla Port', lat: 22.600, lng: 69.900 },

  // Moved West
  { name: 'Jawaharlal Nehru Port (Nhava Sheva)', lat: 18.930, lng: 72.750 },
  { name: 'Mormugao Port', lat: 15.410, lng: 73.700 },
  { name: 'Paradip Port', lat: 20.260, lng: 86.800 },

  // --- HALDIA FIX ---
  // OLD (Bad): 22.025, 88.075 (River Bank)
  // NEW (Good): 21.600, 88.000 (Open Sea)
  { name: 'Haldia Port', lat: 21.600, lng: 88.000 }, 

  { name: 'Tuticorin Port', lat: 8.750, lng: 78.300 },
  { name: 'Mangalore Port', lat: 12.890, lng: 74.700 }
];