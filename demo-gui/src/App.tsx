import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import Dashboard from './components/Dashboard'
import LoadingScreen from './components/LoadingScreen'
import { NeuralDataProvider } from './contexts/NeuralDataContext'

function App() {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate system initialization
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 3000)

    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return <LoadingScreen />
  }

  return (
    <div className="w-full h-screen bg-dark-900 text-white overflow-hidden">
      <NeuralDataProvider>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
          className="w-full h-full"
        >
          <Dashboard />
        </motion.div>
      </NeuralDataProvider>
    </div>
  )
}

export default App
