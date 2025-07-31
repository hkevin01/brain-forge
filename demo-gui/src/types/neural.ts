// Neural data types for Brain-Forge BCI Platform
export interface DeviceStatus {
  id: string
  name: string
  type: 'OPM' | 'Kernel' | 'Accelerometer'
  status: 'connected' | 'disconnected' | 'error' | 'calibrating'
  signalQuality: number // 0-100
  lastUpdate: Date
  channels?: number
  sampleRate?: number
  batteryLevel?: number
}

export interface BrainSignal {
  timestamp: number
  channels: number[]
  frequency: number
  amplitude: number
  quality: number
}

export interface OPMData extends BrainSignal {
  magneticField: number[]
  gradientNoise: number
  sensorPositions: Vector3[]
}

export interface KernelData extends BrainSignal {
  flowData: number[]
  fluxData: number[]
  neuronSpeed: number
  oxygenation: number
}

export interface AccelerometerData {
  timestamp: number
  x: number
  y: number
  z: number
  magnitude: number
  rotationRate: Vector3
}

export interface Vector3 {
  x: number
  y: number
  z: number
}

export interface BrainRegion {
  id: string
  name: string
  atlas: 'Harvard-Oxford' | 'AAL' | 'Yeo'
  position: Vector3
  activity: number // 0-1
  connections: string[]
}

export interface NeuralNetwork {
  nodes: NeuralNode[]
  connections: NeuralConnection[]
  activity: number
  timestamp: number
}

export interface NeuralNode {
  id: string
  position: Vector3
  activity: number
  type: 'excitatory' | 'inhibitory'
  firingRate: number
}

export interface NeuralConnection {
  source: string
  target: string
  weight: number
  delay: number
  plasticity: number
}

export interface SpectralData {
  frequencies: number[]
  power: number[]
  phase: number[]
  coherence: number[]
}

export interface SystemMetrics {
  cpuUsage: number
  memoryUsage: number
  diskUsage: number
  networkLatency: number
  dataRate: number // MB/s
  compressionRatio: number
  processingDelay: number // ms
}

export interface AlertData {
  id: string
  type: 'warning' | 'error' | 'info' | 'anomaly'
  message: string
  timestamp: Date
  severity: 'low' | 'medium' | 'high' | 'critical'
  resolved: boolean
}
