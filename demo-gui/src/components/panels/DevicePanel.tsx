import { motion } from 'framer-motion'
import { AlertTriangle, Battery, Monitor, Wifi, WifiOff } from 'lucide-react'
import { useNeuralData } from '../../contexts/NeuralDataContext'
import { DeviceStatus } from '../../types/neural'

export default function DevicePanel() {
  const { devices } = useNeuralData()

  const getStatusColor = (status: DeviceStatus['status']) => {
    switch (status) {
      case 'connected': return 'text-neural-green'
      case 'disconnected': return 'text-gray-500'
      case 'error': return 'text-neural-red'
      case 'calibrating': return 'text-neural-orange'
      default: return 'text-gray-500'
    }
  }

  const getStatusIcon = (status: DeviceStatus['status']) => {
    switch (status) {
      case 'connected': return <Wifi className="w-4 h-4" />
      case 'disconnected': return <WifiOff className="w-4 h-4" />
      case 'error': return <AlertTriangle className="w-4 h-4" />
      case 'calibrating': return <Monitor className="w-4 h-4 animate-pulse" />
      default: return <WifiOff className="w-4 h-4" />
    }
  }

  return (
    <div className="glass-panel p-4 h-full">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
        <Monitor className="w-5 h-5 mr-2 text-neural-blue" />
        Device Status
      </h2>
      
      <div className="space-y-3">
        {devices.map((device, index) => (
          <motion.div
            key={device.id}
            initial={{ x: -50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: index * 0.1 }}
            className="glass-panel p-3 border border-white/5"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <span className={getStatusColor(device.status)}>
                  {getStatusIcon(device.status)}
                </span>
                <h3 className="font-medium text-sm text-white">{device.name}</h3>
              </div>
              {device.batteryLevel && (
                <div className="flex items-center space-x-1">
                  <Battery className="w-3 h-3 text-gray-400" />
                  <span className="text-xs font-mono text-gray-400">
                    {device.batteryLevel.toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
            
            <div className="space-y-1 text-xs font-mono text-gray-400">
              <div className="flex justify-between">
                <span>Channels:</span>
                <span className="text-neural-blue">{device.channels}</span>
              </div>
              <div className="flex justify-between">
                <span>Quality:</span>
                <span className={`${device.signalQuality > 90 ? 'text-neural-green' : 
                  device.signalQuality > 70 ? 'text-neural-orange' : 'text-neural-red'}`}>
                  {device.signalQuality.toFixed(1)}%
                </span>
              </div>
              {device.sampleRate && (
                <div className="flex justify-between">
                  <span>Rate:</span>
                  <span className="text-neutral-300">{device.sampleRate} Hz</span>
                </div>
              )}
            </div>

            {/* Signal quality bar */}
            <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
              <motion.div
                className={`h-1 rounded-full ${device.signalQuality > 90 ? 'bg-neural-green' : 
                  device.signalQuality > 70 ? 'bg-neural-orange' : 'bg-neural-red'}`}
                initial={{ width: 0 }}
                animate={{ width: `${device.signalQuality}%` }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
              />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
