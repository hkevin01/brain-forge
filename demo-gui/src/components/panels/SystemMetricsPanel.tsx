import { motion } from 'framer-motion'
import { Cpu, Gauge, HardDrive, MemoryStick, Wifi } from 'lucide-react'
import { useNeuralData } from '../../contexts/NeuralDataContext'

export default function SystemMetricsPanel() {
  const { systemMetrics } = useNeuralData()

  const getStatusColor = (value: number, thresholds: {warning: number, critical: number}) => {
    if (value >= thresholds.critical) return 'text-neural-red'
    if (value >= thresholds.warning) return 'text-neural-orange'
    return 'text-neural-green'
  }

  const getStatusBackground = (value: number, thresholds: {warning: number, critical: number}) => {
    if (value >= thresholds.critical) return 'bg-neural-red'
    if (value >= thresholds.warning) return 'bg-neural-orange'
    return 'bg-neural-green'
  }

  const metrics = [
    {
      name: 'CPU Usage',
      value: systemMetrics.cpuUsage,
      unit: '%',
      icon: Cpu,
      thresholds: { warning: 70, critical: 90 }
    },
    {
      name: 'Memory',
      value: systemMetrics.memoryUsage,
      unit: '%',
      icon: MemoryStick,
      thresholds: { warning: 80, critical: 95 }
    },
    {
      name: 'Disk Usage',
      value: systemMetrics.diskUsage,
      unit: '%',
      icon: HardDrive,
      thresholds: { warning: 70, critical: 85 }
    },
    {
      name: 'Network Latency',
      value: systemMetrics.networkLatency,
      unit: 'ms',
      icon: Wifi,
      thresholds: { warning: 30, critical: 50 }
    }
  ]

  return (
    <div className="glass-panel p-4 h-full">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
        <Gauge className="w-5 h-5 mr-2 text-neural-green" />
        System Metrics
      </h2>

      <div className="space-y-4">
        {/* System Performance Metrics */}
        <div className="space-y-3">
          {metrics.map((metric, index) => {
            const IconComponent = metric.icon
            return (
              <motion.div
                key={metric.name}
                initial={{ x: 50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
                className="glass-panel p-3"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <IconComponent className="w-4 h-4 text-gray-400" />
                    <span className="text-sm font-medium text-white">{metric.name}</span>
                  </div>
                  <span className={`text-sm font-mono ${getStatusColor(metric.value, metric.thresholds)}`}>
                    {metric.value.toFixed(1)}{metric.unit}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <motion.div
                    className={`h-2 rounded-full ${getStatusBackground(metric.value, metric.thresholds)}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(metric.value, 100)}%` }}
                    transition={{ duration: 0.8, delay: index * 0.1 }}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Performance Statistics */}
        <div className="glass-panel p-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Performance</h3>
          
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Data Rate:</span>
              <span className="font-mono text-neural-blue">
                {systemMetrics.dataRate.toFixed(1)} MB/s
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Compression:</span>
              <span className="font-mono text-neural-green">
                {(systemMetrics.compressionRatio * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Processing Delay:</span>
              <span className="font-mono text-neural-purple">
                {systemMetrics.processingDelay.toFixed(0)} ms
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Uptime:</span>
              <span className="font-mono text-neural-orange">
                {Math.floor(Date.now() / 1000 / 3600)}h {Math.floor((Date.now() / 1000 % 3600) / 60)}m
              </span>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div className="glass-panel p-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Status</h3>
          
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Overall Health:</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-neural-green animate-pulse" />
                <span className="font-mono text-neural-green">OPTIMAL</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Temperature:</span>
              <span className="font-mono text-neural-blue">
                {(35 + Math.random() * 10).toFixed(1)}°C
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Power Draw:</span>
              <span className="font-mono text-neural-orange">
                {(250 + Math.random() * 50).toFixed(0)}W
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Cooling:</span>
              <span className="font-mono text-neural-green">ACTIVE</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
