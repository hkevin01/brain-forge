import { motion } from 'framer-motion'
import { Activity, Brain, Settings, Wifi } from 'lucide-react'
import { useState } from 'react'
import { useNeuralData } from '../../contexts/NeuralDataContext'

export default function Header() {
  const { isAcquiring, devices } = useNeuralData()
  const [currentTime, setCurrentTime] = useState(new Date())

  // Update time every second
  useState(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  })

  const connectedDevices = devices.filter(d => d.status === 'connected').length
  const totalSignalQuality = devices.reduce((sum, d) => sum + d.signalQuality, 0) / devices.length || 0

  return (
    <header className="h-16 glass-panel border-b border-white/10 flex items-center justify-between px-6">
      {/* Logo and Title */}
      <div className="flex items-center space-x-4">
        <motion.div
          animate={{ rotate: isAcquiring ? 360 : 0 }}
          transition={{ duration: 2, repeat: isAcquiring ? Infinity : 0, ease: "linear" }}
        >
          <Brain className="w-8 h-8 text-neural-blue neural-glow" />
        </motion.div>
        <div>
          <h1 className="text-xl font-bold text-white">Brain-Forge BCI Platform</h1>
          <p className="text-sm text-gray-400">Real-time Neural Interface System</p>
        </div>
      </div>

      {/* System Status */}
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <Wifi className="w-4 h-4 text-neural-green" />
          <span className="text-sm font-mono">
            {connectedDevices}/{devices.length} Devices
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-neural-purple" />
          <span className="text-sm font-mono">
            Signal: {totalSignalQuality.toFixed(1)}%
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${
            isAcquiring ? 'bg-neural-green animate-pulse' : 'bg-gray-500'
          }`} />
          <span className="text-sm font-mono">
            {isAcquiring ? 'ACQUIRING' : 'STANDBY'}
          </span>
        </div>

        <div className="text-sm font-mono text-gray-400">
          {currentTime.toLocaleTimeString()}
        </div>

        <button className="neural-button p-2">
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}
