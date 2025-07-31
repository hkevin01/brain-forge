import { motion } from 'framer-motion'
import { Download, Pause, Play, Settings, Square } from 'lucide-react'
import { useNeuralData } from '../../contexts/NeuralDataContext'

export default function ControlPanel() {
  const { isAcquiring, startAcquisition, stopAcquisition } = useNeuralData()

  const handleStartStop = () => {
    if (isAcquiring) {
      stopAcquisition()
    } else {
      startAcquisition()
    }
  }

  return (
    <div className="glass-panel p-4 h-full">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
        <Settings className="w-5 h-5 mr-2 text-neural-purple" />
        Control Panel
      </h2>

      <div className="space-y-4">
        {/* Main Controls */}
        <div className="space-y-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleStartStop}
            className={`w-full neural-button p-3 flex items-center justify-center space-x-2 ${
              isAcquiring 
                ? 'bg-neural-red/20 border-neural-red/30 hover:bg-neural-red/30' 
                : 'bg-neural-green/20 border-neural-green/30 hover:bg-neural-green/30'
            }`}
          >
            {isAcquiring ? (
              <>
                <Pause className="w-4 h-4" />
                <span>Stop Acquisition</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Start Acquisition</span>
              </>
            )}
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full neural-button p-2 flex items-center justify-center space-x-2"
            disabled={!isAcquiring}
          >
            <Square className="w-4 h-4" />
            <span>Emergency Stop</span>
          </motion.button>
        </div>

        {/* Parameter Controls */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-300">Signal Parameters</h3>
          
          <div className="space-y-2">
            <label className="block text-xs text-gray-400">Sample Rate</label>
            <select className="w-full bg-dark-800 border border-white/10 rounded-lg p-2 text-sm text-white">
              <option value="1000">1000 Hz</option>
              <option value="2000">2000 Hz</option>
              <option value="4000">4000 Hz</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="block text-xs text-gray-400">Filter Range</label>
            <div className="flex space-x-2">
              <input 
                type="number" 
                placeholder="0.1" 
                className="flex-1 bg-dark-800 border border-white/10 rounded-lg p-2 text-sm text-white"
              />
              <span className="text-gray-400 self-center">-</span>
              <input 
                type="number" 
                placeholder="100" 
                className="flex-1 bg-dark-800 border border-white/10 rounded-lg p-2 text-sm text-white"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-xs text-gray-400">Compression</label>
            <input 
              type="range" 
              min="0" 
              max="100" 
              defaultValue="73"
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400">
              <span>None</span>
              <span>73%</span>
              <span>Max</span>
            </div>
          </div>
        </div>

        {/* Export Controls */}
        <div className="pt-2 border-t border-white/10">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full neural-button p-2 flex items-center justify-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>Export Data</span>
          </motion.button>
        </div>

        {/* Calibration Status */}
        <div className="pt-2 border-t border-white/10">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Calibration</span>
            <span className="text-neural-green font-mono">COMPLETE</span>
          </div>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full mt-2 neural-button p-2 text-xs"
          >
            Recalibrate Devices
          </motion.button>
        </div>
      </div>
    </div>
  )
}
