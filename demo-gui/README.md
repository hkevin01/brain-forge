# Brain-Forge Demo GUI - README

## Overview
A comprehensive demonstration GUI for the Brain-Forge brain scanning and simulation platform that showcases all major capabilities of a multi-modal brain-computer interface system.

## 🧠 Features

### Real-Time Brain Visualization
- Interactive 3D brain model with neural activity overlays
- Dynamic connectivity network visualization
- Multiple viewing modes (3D, sagittal, coronal, axial)
- Activity-based region scaling and color coding

### Multi-Modal Data Acquisition
- **OPM Helmet**: 306-channel MEG-like sensor simulation
- **Kernel Optical**: Flow/Flux NIRS data with hemodynamic modeling
- **Accelerometer**: 6-axis head motion tracking
- Real-time signal quality monitoring

### Signal Processing
- Live frequency spectrum analysis (Delta, Theta, Alpha, Beta, Gamma)
- Signal-to-noise ratio monitoring
- Artifact detection and quality assessment
- Data compression and throughput metrics

### System Controls
- Start/stop acquisition with visual feedback
- Real-time parameter adjustment
- Device calibration interfaces
- Emergency stop protocols
- Data export simulation

### Professional Interface
- Dark neuroscience-themed design with glassmorphism effects
- Responsive grid layout for multiple screen sizes
- Smooth animations and transitions
- Color-coded status indicators

## 🚀 Tech Stack
- **Frontend**: React 18 + TypeScript
- **Build System**: Vite
- **Styling**: TailwindCSS
- **3D Graphics**: Three.js + React Three Fiber
- **Charts**: Recharts
- **Animations**: Framer Motion
- **Icons**: Lucide React

## 📦 Installation & Quick Start

```bash
# Option 1: Full setup and run (from Brain-Forge root directory)
bash run.sh

# Option 2: Manual setup
cd demo-gui
npm install
npm run dev

# Option 3: Quick run (after dependencies installed)
bash run.sh

# Option 4: Individual commands
npm install     # Install dependencies
npm run dev     # Start development server
npm run build   # Build for production
```

**🌐 Access the application at: http://localhost:3000**

## 🎯 Usage

1. **Start the application**: `npm run dev`
2. **Open browser**: Navigate to `http://localhost:3000`
3. **Begin acquisition**: Click "Start Acquisition" to begin real-time simulation
4. **Explore visualizations**: Interact with 3D brain model and monitoring panels
5. **Monitor systems**: View real-time metrics and alerts

## 📁 Project Structure

```
demo-gui/
├── src/
│   ├── components/
│   │   ├── layout/           # Header and layout components
│   │   ├── panels/           # Dashboard panels (Device, Control, etc.)
│   │   └── visualization/    # 3D brain visualization
│   ├── contexts/            # React context for state management
│   ├── types/               # TypeScript type definitions
│   └── utils/               # Data simulation utilities
├── public/                  # Static assets
└── docs/                   # Documentation
```

## 🔧 Development

### Key Components
- `Dashboard.tsx`: Main layout coordinator
- `BrainVisualization.tsx`: 3D brain rendering with Three.js
- `NeuralDataContext.tsx`: Global state management
- `NeuralDataSimulator.ts`: Realistic brain signal simulation

### Data Simulation
The application includes comprehensive simulation of:
- Neural signal patterns across brain regions
- Device connectivity and health status
- System performance metrics
- Alert generation and management

## 🎨 Design System

### Colors
- **Neural Blue**: `#00d4ff` - Primary data streams
- **Neural Green**: `#00ff88` - Healthy status indicators
- **Neural Purple**: `#b347ff` - Secondary data types
- **Neural Orange**: `#ff6b35` - Warnings and alerts
- **Neural Red**: `#ff3366` - Errors and critical alerts

### Typography
- **Headers**: Inter font family
- **Data/Code**: JetBrains Mono for technical readability

## 📊 Simulated Data

### Brain Regions
- Frontal, Parietal, Temporal, Occipital Cortex
- Motor, Sensory, Auditory, Visual Cortex
- Hippocampus, Amygdala, Thalamus, Cerebellum

### Signal Processing
- Realistic frequency band power distribution
- Hemodynamic response modeling for NIRS
- Motion artifact simulation
- Signal quality degradation patterns

## 🚨 Alerts & Monitoring
- Real-time anomaly detection
- Device health monitoring
- Signal quality thresholds
- System performance alerts

## 📈 Performance
- Real-time data updates at 10-50 Hz depending on modality
- Efficient 3D rendering with optimized geometries
- Responsive UI with smooth 60fps animations
- Memory-efficient data streaming

## 🎯 Target Audience
This demonstration is designed to impress:
- **Neuroscientists**: Realistic data patterns and processing
- **Investors**: Professional interface and advanced capabilities
- **Technical Teams**: Modern architecture and implementation quality

## 📝 License
Part of the Brain-Forge BCI Platform project.

---

**Status**: Demo implementation 85% complete - Ready for final integration and testing.