import { motion } from 'framer-motion'
import { Activity, Brain, Zap } from 'lucide-react'

export default function LoadingScreen() {
  return (
    <div className="w-full h-screen bg-dark-900 flex items-center justify-center">
      <div className="text-center">
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mb-8"
        >
          <div className="relative">
            <Brain size={120} className="text-neural-blue mx-auto neural-glow" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              className="absolute inset-0 flex items-center justify-center"
            >
              <Zap size={40} className="text-neural-purple" />
            </motion.div>
          </div>
        </motion.div>

        <motion.h1
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-4xl font-bold text-white mb-4 text-shadow"
        >
          Brain-Forge BCI Platform
        </motion.h1>

        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="text-xl text-gray-300 mb-8"
        >
          Real-time Neural Interface System
        </motion.p>

        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.6 }}
          className="flex items-center justify-center space-x-4 text-neural-green"
        >
          <Activity className="animate-pulse" />
          <span className="font-mono text-lg">Initializing neural networks...</span>
        </motion.div>

        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          transition={{ delay: 1, duration: 2, ease: "easeInOut" }}
          className="w-64 h-1 bg-neural-blue rounded-full mx-auto mt-8 neural-glow"
        />

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2, duration: 0.5 }}
          className="mt-4 text-sm text-gray-400 font-mono"
        >
          Loading complete...
        </motion.div>
      </div>
    </div>
  )
}
