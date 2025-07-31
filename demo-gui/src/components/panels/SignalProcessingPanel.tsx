import { motion } from 'framer-motion'
import { BarChart3, HardDrive, TrendingUp, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNeuralData } from '../../contexts/NeuralDataContext'

export default function SignalProcessingPanel() {
  const { isAcquiring, simulator } = useNeuralData()
  const [spectralData, setSpectralData] = useState<{frequencies: number[], power: number[]}>({
    frequencies: [],
    power: []
  })

  useEffect(() => {
    if (isAcquiring) {
      const interval = setInterval(() => {
        const newSpectralData = simulator.generateSpectralData()
        setSpectralData(newSpectralData)
      }, 2000) // Update every 2 seconds

      return () => clearInterval(interval)
    }
  }, [isAcquiring, simulator])

  const getFrequencyBand = (freq: number) => {
    if (freq < 4) return { name: 'Delta', color: 'text-neural-red' }
    if (freq < 8) return { name: 'Theta', color: 'text-neural-orange' }
    if (freq < 13) return { name: 'Alpha', color: 'text-neural-blue' }
    if (freq < 30) return { name: 'Beta', color: 'text-neural-green' }
    return { name: 'Gamma', color: 'text-neural-purple' }
  }

  const averagePowerByBand = {
    delta: spectralData.frequencies.length > 0 ? 
      spectralData.power.slice(0, 8).reduce((a, b) => a + b, 0) / 8 : 0,
    theta: spectralData.frequencies.length > 0 ? 
      spectralData.power.slice(8, 16).reduce((a, b) => a + b, 0) / 8 : 0,
    alpha: spectralData.frequencies.length > 0 ? 
      spectralData.power.slice(16, 26).reduce((a, b) => a + b, 0) / 10 : 0,
    beta: spectralData.frequencies.length > 0 ? 
      spectralData.power.slice(26, 60).reduce((a, b) => a + b, 0) / 34 : 0,
    gamma: spectralData.frequencies.length > 0 ? 
      spectralData.power.slice(60, 100).reduce((a, b) => a + b, 0) / 40 : 0,
  }

  return (
    <div className="glass-panel p-4 h-full">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
        <BarChart3 className="w-5 h-5 mr-2 text-neural-blue" />
        Signal Processing
      </h2>

      <div className="space-y-4">
        {/* Frequency Band Analysis */}
        <div className="glass-panel p-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center">
            <Zap className="w-4 h-4 mr-1" />
            Frequency Bands
          </h3>
          
          <div className="space-y-2">
            {Object.entries(averagePowerByBand).map(([band, power]) => {
              const bandInfo = getFrequencyBand(
                band === 'delta' ? 2 : band === 'theta' ? 6 : 
                band === 'alpha' ? 10 : band === 'beta' ? 20 : 40
              )
              const maxPower = Math.max(...Object.values(averagePowerByBand))
              const width = maxPower > 0 ? (power / maxPower) * 100 : 0

              return (
                <div key={band} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 w-20">
                    <span className={`text-xs font-medium ${bandInfo.color}`}>
                      {band.charAt(0).toUpperCase() + band.slice(1)}
                    </span>
                  </div>
                  <div className="flex-1 mx-2">
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <motion.div
                        className={`h-2 rounded-full ${
                          bandInfo.color.includes('red') ? 'bg-neural-red' :
                          bandInfo.color.includes('orange') ? 'bg-neural-orange' :
                          bandInfo.color.includes('blue') ? 'bg-neural-blue' :
                          bandInfo.color.includes('green') ? 'bg-neural-green' :
                          'bg-neural-purple'
                        }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${width}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  </div>
                  <span className="text-xs font-mono text-gray-400 w-12 text-right">
                    {power.toFixed(1)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Signal Quality Metrics */}
        <div className="glass-panel p-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center">
            <TrendingUp className="w-4 h-4 mr-1" />
            Signal Quality
          </h3>
          
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">SNR:</span>
              <span className="font-mono text-neural-green">
                {(15 + Math.random() * 5).toFixed(1)} dB
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Artifacts:</span>
              <span className="font-mono text-neural-blue">
                {(Math.random() * 3).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Coherence:</span>
              <span className="font-mono text-neural-purple">
                {(0.7 + Math.random() * 0.2).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Latency:</span>
              <span className="font-mono text-neural-orange">
                {(8 + Math.random() * 4).toFixed(0)} ms
              </span>
            </div>
          </div>
        </div>

        {/* Data Processing Stats */}
        <div className="glass-panel p-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center">
            <HardDrive className="w-4 h-4 mr-1" />
            Processing
          </h3>
          
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Throughput:</span>
              <span className="font-mono text-neural-blue">
                {(125 + Math.random() * 25).toFixed(1)} MB/s
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Compression:</span>
              <span className="font-mono text-neural-green">
                {(0.7 + Math.random() * 0.1).toFixed(1)}x
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Buffer:</span>
              <span className="font-mono text-neural-orange">
                {(45 + Math.random() * 20).toFixed(0)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Processed:</span>
              <span className="font-mono text-neural-purple">
                {(Math.random() * 1000000).toFixed(0)} samples
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
