import { AccelerometerData, DeviceStatus, KernelData, OPMData, Vector3 } from '../types/neural'

// Realistic neural data simulation utilities
export class NeuralDataSimulator {
  private time: number = 0
  private devices: DeviceStatus[] = []
  
  constructor() {
    this.initializeDevices()
  }

  private initializeDevices(): void {
    this.devices = [
      {
        id: 'opm-helmet-001',
        name: 'OPM Helmet Array',
        type: 'OPM',
        status: 'connected',
        signalQuality: 95,
        lastUpdate: new Date(),
        channels: 306,
        sampleRate: 1000,
        batteryLevel: 87
      },
      {
        id: 'kernel-optical-001',
        name: 'Kernel Flow/Flux System',
        type: 'Kernel',
        status: 'connected',
        signalQuality: 92,
        lastUpdate: new Date(),
        channels: 52,
        sampleRate: 50,
        batteryLevel: 73
      },
      {
        id: 'accelerometer-001',
        name: 'Head Motion Tracker',
        type: 'Accelerometer',
        status: 'connected',
        signalQuality: 98,
        lastUpdate: new Date(),
        channels: 6,
        sampleRate: 200,
        batteryLevel: 91
      }
    ]
  }

  public getDeviceStatus(): DeviceStatus[] {
    // Simulate device status updates
    this.devices.forEach(device => {
      device.lastUpdate = new Date()
      device.signalQuality += (Math.random() - 0.5) * 2
      device.signalQuality = Math.max(80, Math.min(100, device.signalQuality))
      
      if (device.batteryLevel !== undefined) {
        device.batteryLevel -= Math.random() * 0.01
        device.batteryLevel = Math.max(0, device.batteryLevel)
      }
    })
    
    return this.devices
  }

  public generateOPMData(): OPMData {
    this.time += 1/1000 // 1ms increment
    
    const channels = Array.from({ length: 306 }, (_, i) => {
      const baseFreq = 8 + Math.sin(this.time * 0.1) * 2 // Alpha wave variation
      const noise = (Math.random() - 0.5) * 0.1
      return Math.sin(this.time * baseFreq + i * 0.1) + noise
    })

    const magneticField = channels.map(ch => ch * 1e-15) // Tesla scale
    
    const sensorPositions: Vector3[] = Array.from({ length: 306 }, (_, i) => ({
      x: Math.cos(i * 0.02) * 10,
      y: Math.sin(i * 0.02) * 10,
      z: 5 + Math.sin(i * 0.05) * 2
    }))

    return {
      timestamp: Date.now(),
      channels,
      frequency: 8 + Math.sin(this.time * 0.1) * 2,
      amplitude: 1 + Math.sin(this.time * 0.05) * 0.3,
      quality: 95 + Math.sin(this.time * 0.02) * 3,
      magneticField,
      gradientNoise: Math.random() * 0.1,
      sensorPositions
    }
  }

  public generateKernelData(): KernelData {
    const channels = Array.from({ length: 52 }, (_, i) => {
      const hemodynamic = Math.sin(this.time * 0.05 + i * 0.2) * 0.7
      const noise = (Math.random() - 0.5) * 0.2
      return hemodynamic + noise
    })

    const flowData = channels.map(ch => ch * 1.2 + 0.5)
    const fluxData = channels.map(ch => ch * 0.8 + 0.3)

    return {
      timestamp: Date.now(),
      channels,
      frequency: 0.1, // Hemodynamic response frequency
      amplitude: 0.7 + Math.sin(this.time * 0.03) * 0.2,
      quality: 92 + Math.sin(this.time * 0.04) * 4,
      flowData,
      fluxData,
      neuronSpeed: 2.5 + Math.sin(this.time * 0.08) * 0.5,
      oxygenation: 0.98 + Math.sin(this.time * 0.02) * 0.02
    }
  }

  public generateAccelerometerData(): AccelerometerData {
    const headMotion = {
      x: Math.sin(this.time * 0.3) * 0.1 + (Math.random() - 0.5) * 0.02,
      y: Math.cos(this.time * 0.2) * 0.08 + (Math.random() - 0.5) * 0.02,
      z: Math.sin(this.time * 0.4) * 0.05 + (Math.random() - 0.5) * 0.01
    }

    const magnitude = Math.sqrt(headMotion.x ** 2 + headMotion.y ** 2 + headMotion.z ** 2)
    
    const rotationRate: Vector3 = {
      x: (Math.random() - 0.5) * 0.1,
      y: (Math.random() - 0.5) * 0.1,
      z: (Math.random() - 0.5) * 0.05
    }

    return {
      timestamp: Date.now(),
      ...headMotion,
      magnitude,
      rotationRate
    }
  }

  public generateSpectralData(): { frequencies: number[], power: number[] } {
    const N = 512 // FFT size
    const sampleRate = 1000
    
    const frequencies = Array.from({ length: N/2 }, (_, i) => (i * sampleRate) / N)
    
    // Simulate realistic brain power spectrum
    const power = frequencies.map(freq => {
      if (freq < 1) return Math.random() * 10 // DC and very low freq
      else if (freq < 4) return Math.exp(-freq * 0.5) * 20 // Delta
      else if (freq < 8) return Math.exp(-((freq - 6) ** 2) / 2) * 15 // Theta
      else if (freq < 13) return Math.exp(-((freq - 10) ** 2) / 4) * 25 // Alpha
      else if (freq < 30) return Math.exp(-((freq - 20) ** 2) / 10) * 10 // Beta
      else if (freq < 100) return Math.exp(-freq * 0.1) * 5 // Gamma
      else return Math.random() * 2 // High frequency noise
    })

    return { frequencies: frequencies.slice(0, 100), power: power.slice(0, 100) }
  }

  public generateBrainActivity(): { regions: any[], connectivity: number[][] } {
    const regionNames = [
      'Frontal Cortex', 'Parietal Cortex', 'Temporal Cortex', 'Occipital Cortex',
      'Motor Cortex', 'Sensory Cortex', 'Auditory Cortex', 'Visual Cortex',
      'Hippocampus', 'Amygdala', 'Thalamus', 'Cerebellum'
    ]

    const regions = regionNames.map((name, i) => ({
      id: `region-${i}`,
      name,
      activity: 0.3 + Math.sin(this.time * 0.1 + i) * 0.3 + Math.random() * 0.2,
      position: {
        x: Math.cos(i * 0.5) * 8,
        y: Math.sin(i * 0.5) * 8,
        z: Math.sin(i * 0.3) * 4
      }
    }))

    // Generate connectivity matrix
    const connectivity = Array.from({ length: regions.length }, (_, i) =>
      Array.from({ length: regions.length }, (_, j) => {
        if (i === j) return 1
        const distance = Math.abs(i - j)
        return Math.exp(-distance * 0.5) * (0.5 + Math.random() * 0.3)
      })
    )

    return { regions, connectivity }
  }
}
