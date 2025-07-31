import { Activity, Waves, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import { useNeuralData } from '../../contexts/NeuralDataContext'

export default function DataAcquisitionPanel() {
  const { isAcquiring, simulator } = useNeuralData()
  const [signalData, setSignalData] = useState<Array<{time: number, omp: number, kernel: number, accel: number}>>([])
  const [currentTime, setCurrentTime] = useState(0)

  useEffect(() => {
    if (isAcquiring) {
      const interval = setInterval(() => {
        const ompData = simulator.generateOPMData()
        const kernelData = simulator.generateKernelData()
        const accelData = simulator.generateAccelerometerData()
        
        const newPoint = {
          time: currentTime,
          omp: ompData.channels[0] * 1000, // Scale for visualization
          kernel: kernelData.channels[0] * 1000,
          accel: accelData.magnitude * 10000
        }

        setSignalData(prev => {
          const newData = [...prev.slice(-99), newPoint] // Keep last 100 points
          return newData
        })
        
        setCurrentTime(prev => prev + 1)
      }, 100) // 10 Hz update rate

      return () => clearInterval(interval)
    }
  }, [isAcquiring, simulator, currentTime])

  return (
    <div className="glass-panel p-4 h-full">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
        <Activity className="w-5 h-5 mr-2 text-neural-green" />
        Live Data Acquisition
      </h2>

      <div className="grid grid-cols-3 gap-4 h-[calc(100%-3rem)]">
        {/* OPM Helmet Data Stream */}
        <div className="glass-panel p-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-neural-blue flex items-center">
              <Zap className="w-4 h-4 mr-1" />
              OPM Helmet (306ch)
            </h3>
            <span className="text-xs font-mono text-gray-400">1000 Hz</span>
          </div>
          
          <div className="h-24 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={signalData}>
                <XAxis dataKey="time" hide />
                <YAxis hide />
                <Line 
                  type="monotone" 
                  dataKey="omp" 
                  stroke="#00d4ff" 
                  strokeWidth={1}
                  dot={false}
                  animationDuration={0}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Amplitude:</span>
              <span className="font-mono text-neural-blue">
                {signalData.length > 0 ? signalData[signalData.length - 1]?.omp.toFixed(2) : '0.00'} µV
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Quality:</span>
              <span className="font-mono text-neural-green">96.3%</span>
            </div>
          </div>
        </div>

        {/* Kernel Optical Data Stream */}
        <div className="glass-panel p-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-neural-purple flex items-center">
              <Waves className="w-4 h-4 mr-1" />
              Kernel Flow/Flux (52ch)
            </h3>
            <span className="text-xs font-mono text-gray-400">50 Hz</span>
          </div>
          
          <div className="h-24 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={signalData}>
                <XAxis dataKey="time" hide />
                <YAxis hide />
                <Line 
                  type="monotone" 
                  dataKey="kernel" 
                  stroke="#b347ff" 
                  strokeWidth={1}
                  dot={false}
                  animationDuration={0}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Flow Rate:</span>
              <span className="font-mono text-neural-purple">
                {signalData.length > 0 ? (signalData[signalData.length - 1]?.kernel * 0.1).toFixed(1) : '0.0'} ml/s
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Oxygenation:</span>
              <span className="font-mono text-neural-green">98.2%</span>
            </div>
          </div>
        </div>

        {/* Accelerometer Data Stream */}
        <div className="glass-panel p-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-neural-orange flex items-center">
              <Activity className="w-4 h-4 mr-1" />
              Motion Tracker (6ch)
            </h3>
            <span className="text-xs font-mono text-gray-400">200 Hz</span>
          </div>
          
          <div className="h-24 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={signalData}>
                <XAxis dataKey="time" hide />
                <YAxis hide />
                <Line 
                  type="monotone" 
                  dataKey="accel" 
                  stroke="#ff6b35" 
                  strokeWidth={1}
                  dot={false}
                  animationDuration={0}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Magnitude:</span>
              <span className="font-mono text-neural-orange">
                {signalData.length > 0 ? (signalData[signalData.length - 1]?.accel * 0.001).toFixed(3) : '0.000'} g
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Motion:</span>
              <span className="font-mono text-neural-green">STABLE</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
