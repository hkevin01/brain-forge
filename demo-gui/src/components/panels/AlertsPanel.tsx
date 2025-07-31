import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, AlertTriangle, CheckCircle, Info, X } from 'lucide-react'
import { useState } from 'react'
import { useNeuralData } from '../../contexts/NeuralDataContext'
import { AlertData } from '../../types/neural'

export default function AlertsPanel() {
  const { alerts, clearAlerts } = useNeuralData()
  const [expandedAlert, setExpandedAlert] = useState<string | null>(null)

  const getAlertIcon = (type: AlertData['type']) => {
    switch (type) {
      case 'error': return AlertTriangle
      case 'warning': return AlertCircle
      case 'anomaly': return AlertTriangle
      case 'info': return Info
      default: return Info
    }
  }

  const getAlertColor = (severity: AlertData['severity']) => {
    switch (severity) {
      case 'critical': return 'text-neural-red border-neural-red/30 bg-neural-red/10'
      case 'high': return 'text-neural-orange border-neural-orange/30 bg-neural-orange/10'
      case 'medium': return 'text-neural-blue border-neural-blue/30 bg-neural-blue/10'
      case 'low': return 'text-neural-green border-neural-green/30 bg-neural-green/10'
      default: return 'text-gray-400 border-gray-400/30 bg-gray-400/10'
    }
  }

  const getIconColor = (type: AlertData['type'], severity: AlertData['severity']) => {
    if (type === 'error' || severity === 'critical') return 'text-neural-red'
    if (type === 'warning' || severity === 'high') return 'text-neural-orange'
    if (type === 'anomaly' || severity === 'medium') return 'text-neural-blue'
    return 'text-neural-green'
  }

  return (
    <div className="glass-panel p-4 h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white flex items-center">
          <AlertTriangle className="w-5 h-5 mr-2 text-neural-orange" />
          System Alerts
        </h2>
        {alerts.length > 0 && (
          <button
            onClick={clearAlerts}
            className="text-xs text-gray-400 hover:text-white neural-button px-2 py-1"
          >
            Clear All
          </button>
        )}
      </div>

      <div className="space-y-2 h-[calc(100%-4rem)] scrollbar-hide overflow-y-auto">
        <AnimatePresence>
          {alerts.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-8"
            >
              <CheckCircle className="w-12 h-12 text-neural-green mx-auto mb-2 opacity-50" />
              <p className="text-sm text-gray-400">No active alerts</p>
              <p className="text-xs text-gray-500">System operating normally</p>
            </motion.div>
          ) : (
            alerts.map((alert, index) => {
              const IconComponent = getAlertIcon(alert.type)
              const isExpanded = expandedAlert === alert.id
              
              return (
                <motion.div
                  key={alert.id}
                  initial={{ x: -300, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: 300, opacity: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`border rounded-lg p-3 cursor-pointer ${getAlertColor(alert.severity)}`}
                  onClick={() => setExpandedAlert(isExpanded ? null : alert.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-2 flex-1">
                      <IconComponent 
                        className={`w-4 h-4 mt-0.5 ${getIconColor(alert.type, alert.severity)}`} 
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="text-xs font-medium uppercase tracking-wide">
                            {alert.type}
                          </span>
                          <span className={`text-xs px-1.5 py-0.5 rounded-full border ${
                            alert.severity === 'critical' ? 'border-neural-red/50 bg-neural-red/20' :
                            alert.severity === 'high' ? 'border-neural-orange/50 bg-neural-orange/20' :
                            alert.severity === 'medium' ? 'border-neural-blue/50 bg-neural-blue/20' :
                            'border-neural-green/50 bg-neural-green/20'
                          }`}>
                            {alert.severity}
                          </span>
                        </div>
                        <p className="text-sm text-white font-medium truncate">
                          {alert.message}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          {alert.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        // Mark as resolved logic could go here
                      }}
                      className="ml-2 text-gray-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>

                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="mt-3 pt-3 border-t border-white/10"
                      >
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Alert ID:</span>
                            <span className="font-mono text-gray-300">{alert.id}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Timestamp:</span>
                            <span className="font-mono text-gray-300">
                              {alert.timestamp.toLocaleString()}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Status:</span>
                            <span className={`font-mono ${alert.resolved ? 'text-neural-green' : 'text-neural-orange'}`}>
                              {alert.resolved ? 'RESOLVED' : 'ACTIVE'}
                            </span>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )
            })
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
