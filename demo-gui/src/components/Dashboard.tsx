import { motion } from 'framer-motion'
import Header from './layout/Header'
import AlertsPanel from './panels/AlertsPanel'
import ControlPanel from './panels/ControlPanel'
import DataAcquisitionPanel from './panels/DataAcquisitionPanel'
import DevicePanel from './panels/DevicePanel'
import SignalProcessingPanel from './panels/SignalProcessingPanel'
import SystemMetricsPanel from './panels/SystemMetricsPanel'
import BrainVisualization from './visualization/BrainVisualization'

export default function Dashboard() {
  return (
    <div className="w-full h-screen bg-dark-900 text-white overflow-hidden">
      <Header />
      
      <div className="h-[calc(100vh-4rem)] grid grid-cols-12 grid-rows-8 gap-4 p-4">
        {/* Left Column - Device Status and Controls */}
        <div className="col-span-3 row-span-8 space-y-4">
          <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="h-1/3"
          >
            <DevicePanel />
          </motion.div>
          
          <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="h-1/3"
          >
            <ControlPanel />
          </motion.div>
          
          <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="h-1/3"
          >
            <AlertsPanel />
          </motion.div>
        </div>

        {/* Center Column - Main Brain Visualization */}
        <motion.div
          initial={{ y: 300, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="col-span-6 row-span-5"
        >
          <BrainVisualization />
        </motion.div>

        {/* Center Bottom - Data Acquisition */}
        <motion.div
          initial={{ y: 300, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="col-span-6 row-span-3"
        >
          <DataAcquisitionPanel />
        </motion.div>

        {/* Right Column - Signal Processing and Metrics */}
        <div className="col-span-3 row-span-8 space-y-4">
          <motion.div
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="h-1/2"
          >
            <SignalProcessingPanel />
          </motion.div>
          
          <motion.div
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="h-1/2"
          >
            <SystemMetricsPanel />
          </motion.div>
        </div>
      </div>
    </div>
  )
}
