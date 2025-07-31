import { createContext, ReactNode, useContext, useEffect, useState } from 'react'
import { AlertData, DeviceStatus, SystemMetrics } from '../types/neural'
import { NeuralDataSimulator } from '../utils/dataSimulator'

interface NeuralDataContextType {
  devices: DeviceStatus[]
  isAcquiring: boolean
  systemMetrics: SystemMetrics
  alerts: AlertData[]
  simulator: NeuralDataSimulator
  startAcquisition: () => void
  stopAcquisition: () => void
  clearAlerts: () => void
}

const NeuralDataContext = createContext<NeuralDataContextType | undefined>(undefined)

interface NeuralDataProviderProps {
  children: ReactNode
}

export function NeuralDataProvider({ children }: NeuralDataProviderProps) {
  const [simulator] = useState(new NeuralDataSimulator())
  const [devices, setDevices] = useState<DeviceStatus[]>([])
  const [isAcquiring, setIsAcquiring] = useState(false)
  const [alerts, setAlerts] = useState<AlertData[]>([])
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpuUsage: 45,
    memoryUsage: 67,
    diskUsage: 23,
    networkLatency: 12,
    dataRate: 125.6,
    compressionRatio: 0.73,
    processingDelay: 8
  })

  useEffect(() => {
    // Initialize devices
    setDevices(simulator.getDeviceStatus())

    // Update system metrics periodically
    const metricsInterval = setInterval(() => {
      setSystemMetrics(prev => ({
        cpuUsage: Math.max(20, Math.min(95, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(30, Math.min(90, prev.memoryUsage + (Math.random() - 0.5) * 5)),
        diskUsage: Math.max(10, Math.min(80, prev.diskUsage + (Math.random() - 0.5) * 2)),
        networkLatency: Math.max(5, Math.min(50, prev.networkLatency + (Math.random() - 0.5) * 8)),
        dataRate: Math.max(50, Math.min(200, prev.dataRate + (Math.random() - 0.5) * 20)),
        compressionRatio: Math.max(0.5, Math.min(0.9, prev.compressionRatio + (Math.random() - 0.5) * 0.1)),
        processingDelay: Math.max(3, Math.min(25, prev.processingDelay + (Math.random() - 0.5) * 4))
      }))
    }, 2000)

    // Generate occasional alerts
    const alertInterval = setInterval(() => {
      if (Math.random() < 0.1) { // 10% chance per interval
        const alertTypes = ['info', 'warning', 'anomaly'] as const
        const severities = ['low', 'medium', 'high'] as const
        const messages = [
          'Signal quality degradation detected in temporal region',
          'Anomalous spike detected in frontal cortex',
          'Device calibration recommended',
          'High motion artifact detected',
          'Network latency spike detected',
          'Neural pattern recognition successful'
        ]

        const newAlert: AlertData = {
          id: `alert-${Date.now()}`,
          type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
          message: messages[Math.floor(Math.random() * messages.length)],
          timestamp: new Date(),
          severity: severities[Math.floor(Math.random() * severities.length)],
          resolved: false
        }

        setAlerts(prev => [newAlert, ...prev.slice(0, 9)]) // Keep last 10 alerts
      }
    }, 8000)

    return () => {
      clearInterval(metricsInterval)
      clearInterval(alertInterval)
    }
  }, [simulator])

  useEffect(() => {
    if (isAcquiring) {
      const deviceUpdateInterval = setInterval(() => {
        setDevices(simulator.getDeviceStatus())
      }, 1000)

      return () => clearInterval(deviceUpdateInterval)
    }
  }, [isAcquiring, simulator])

  const startAcquisition = () => {
    setIsAcquiring(true)
    const alert: AlertData = {
      id: `alert-${Date.now()}`,
      type: 'info',
      message: 'Data acquisition started successfully',
      timestamp: new Date(),
      severity: 'low',
      resolved: false
    }
    setAlerts(prev => [alert, ...prev.slice(0, 9)])
  }

  const stopAcquisition = () => {
    setIsAcquiring(false)
    const alert: AlertData = {
      id: `alert-${Date.now()}`,
      type: 'info',
      message: 'Data acquisition stopped',
      timestamp: new Date(),
      severity: 'low',
      resolved: false
    }
    setAlerts(prev => [alert, ...prev.slice(0, 9)])
  }

  const clearAlerts = () => {
    setAlerts([])
  }

  const value: NeuralDataContextType = {
    devices,
    isAcquiring,
    systemMetrics,
    alerts,
    simulator,
    startAcquisition,
    stopAcquisition,
    clearAlerts
  }

  return (
    <NeuralDataContext.Provider value={value}>
      {children}
    </NeuralDataContext.Provider>
  )
}

export function useNeuralData() {
  const context = useContext(NeuralDataContext)
  if (context === undefined) {
    throw new Error('useNeuralData must be used within a NeuralDataProvider')
  }
  return context
}
