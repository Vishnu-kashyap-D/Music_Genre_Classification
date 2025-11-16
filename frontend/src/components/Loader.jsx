import { motion } from 'framer-motion'
import { Music2 } from 'lucide-react'

export default function Loader() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh]">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        className="mb-8"
      >
        <div className="relative w-32 h-32">
          {/* Vinyl Record */}
          <div className="absolute inset-0 rounded-full bg-gradient-to-br from-gray-800 to-gray-900 border-8 border-gray-700 shadow-2xl" />
          <div className="absolute inset-4 rounded-full bg-gradient-to-br from-gray-700 to-gray-800" />
          <div className="absolute inset-8 rounded-full bg-gray-900" />
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-gray-800" />
          
          {/* Music Icon */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <Music2 className="w-6 h-6 text-purple-400" />
          </div>
        </div>
      </motion.div>

      <motion.h2
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-2xl font-bold mb-2"
      >
        Deconstructing frequencies...
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-dark-text-muted"
      >
        Analyzing audio patterns and extracting features
      </motion.p>

      {/* Equalizer Animation */}
      <div className="flex items-end gap-1 mt-8 h-16">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="w-2 bg-gradient-to-t from-purple-500 to-pink-500 rounded-t"
            initial={{ height: '20%' }}
            animate={{
              height: [`20%`, `${Math.random() * 80 + 20}%`, '20%'],
            }}
            transition={{
              duration: 1 + Math.random(),
              repeat: Infinity,
              delay: i * 0.1,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>
    </div>
  )
}

