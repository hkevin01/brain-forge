import { OrbitControls, Text } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { motion } from 'framer-motion'
import { Brain, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { useNeuralData } from '../../contexts/NeuralDataContext'

// Brain mesh component
function BrainMesh() {
  const meshRef = useRef<THREE.Mesh>(null)
  const { simulator, isAcquiring } = useNeuralData()
  const [brainActivity, setBrainActivity] = useState<{regions: any[], connectivity: number[][]}>({
    regions: [],
    connectivity: []
  })

  useEffect(() => {
    if (isAcquiring) {
      const interval = setInterval(() => {
        const newActivity = simulator.generateBrainActivity()
        setBrainActivity(newActivity)
      }, 1000)

      return () => clearInterval(interval)
    }
  }, [isAcquiring, simulator])

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005
    }
  })

  return (
    <group>
      {/* Main brain mesh */}
      <mesh ref={meshRef} position={[0, 0, 0]}>
        <sphereGeometry args={[3, 32, 32]} />
        <meshStandardMaterial
          color="#2a4a6b"
          transparent
          opacity={0.3}
          wireframe={false}
        />
      </mesh>

      {/* Brain regions with activity */}
      {brainActivity.regions.map((region, index) => (
        <group key={region.id} position={[region.position.x, region.position.y, region.position.z]}>
          <mesh>
            <sphereGeometry args={[0.2 + region.activity * 0.3, 8, 8]} />
            <meshStandardMaterial
              color={new THREE.Color().setHSL(0.6 - region.activity * 0.4, 0.8, 0.5)}
              transparent
              opacity={0.7 + region.activity * 0.3}
              emissive={new THREE.Color().setHSL(0.6 - region.activity * 0.4, 0.5, region.activity * 0.3)}
            />
          </mesh>
          
          {/* Region labels */}
          <Text
            position={[0, 0.5, 0]}
            fontSize={0.3}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {region.name.split(' ')[0]}
          </Text>
        </group>
      ))}

      {/* Neural connections */}
      {brainActivity.regions.map((sourceRegion, i) =>
        brainActivity.regions.map((targetRegion, j) => {
          if (i >= j || !brainActivity.connectivity[i] || brainActivity.connectivity[i][j] < 0.5) return null
          
          const start = new THREE.Vector3(sourceRegion.position.x, sourceRegion.position.y, sourceRegion.position.z)
          const end = new THREE.Vector3(targetRegion.position.x, targetRegion.position.y, targetRegion.position.z)
          const curve = new THREE.QuadraticBezierCurve3(
            start,
            new THREE.Vector3((start.x + end.x) / 2, (start.y + end.y) / 2 + 1, (start.z + end.z) / 2),
            end
          )
          
          const points = curve.getPoints(20)
          const geometry = new THREE.BufferGeometry().setFromPoints(points)
          
          return (
            <line key={`${i}-${j}`} geometry={geometry}>
              <lineBasicMaterial
                color={new THREE.Color().setHSL(0.5, 0.8, brainActivity.connectivity[i][j])}
                transparent
                opacity={brainActivity.connectivity[i][j] * 0.6}
              />
            </line>
          )
        })
      )}
    </group>
  )
}

export default function BrainVisualization() {
  const { isAcquiring } = useNeuralData()
  const [viewMode, setViewMode] = useState<'3d' | 'sagittal' | 'coronal' | 'axial'>('3d')

  return (
    <div className="glass-panel p-4 h-full relative">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white flex items-center">
          <Brain className="w-5 h-5 mr-2 text-neural-blue" />
          Brain Visualization
        </h2>
        
        <div className="flex items-center space-x-2">
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as any)}
            className="bg-dark-800 border border-white/10 rounded-lg px-2 py-1 text-sm text-white"
          >
            <option value="3d">3D View</option>
            <option value="sagittal">Sagittal</option>
            <option value="coronal">Coronal</option>
            <option value="axial">Axial</option>
          </select>
          
          <button className="neural-button p-1">
            <RotateCcw className="w-4 h-4" />
          </button>
          <button className="neural-button p-1">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button className="neural-button p-1">
            <ZoomOut className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="h-[calc(100%-4rem)] w-full relative">
        {isAcquiring ? (
          <Canvas camera={{ position: [10, 5, 10], fov: 50 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4f46e5" />
            
            <BrainMesh />
            
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              autoRotate={false}
              autoRotateSpeed={0.5}
            />
          </Canvas>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center">
              <Brain className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400 mb-2">Brain visualization ready</p>
              <p className="text-sm text-gray-500">Start acquisition to view real-time activity</p>
            </div>
          </div>
        )}

        {/* Activity indicators overlay */}
        {isAcquiring && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-4 right-4 space-y-2"
          >
            <div className="glass-panel p-2 text-xs">
              <div className="flex items-center space-x-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-neural-blue animate-pulse" />
                <span className="text-white">Neural Activity</span>
              </div>
              <div className="flex items-center space-x-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-neural-green animate-pulse" />
                <span className="text-white">High Connectivity</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-neural-purple animate-pulse" />
                <span className="text-white">Network Flow</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
