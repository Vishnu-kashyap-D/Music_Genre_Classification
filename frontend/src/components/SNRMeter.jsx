import { motion } from 'framer-motion'
import { Gauge } from 'lucide-react'

export default function SNRMeter({ snr }) {
  // Normalize SNR to 0-100 scale (0-60 dB range)
  const normalized = Math.min(100, (snr / 60) * 100)
  const getColor = () => {
    if (snr > 30) return 'text-green-400'
    if (snr > 15) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getQuality = () => {
    if (snr > 30) return 'Excellent'
    if (snr > 20) return 'Good'
    if (snr > 10) return 'Fair'
    return 'Poor'
  }

  return (
    <div className="bg-dark-card rounded-2xl p-6 border border-dark-border">
      <div className="flex items-center gap-3 mb-4">
        <Gauge className="w-6 h-6 text-purple-400" />
        <h3 className="text-xl font-bold">Audio Quality (SNR)</h3>
      </div>
      
      <div className="space-y-4">
        <div className="relative h-32">
          {/* Gauge Background */}
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="50%"
              cy="50%"
              r="45%"
              fill="none"
              stroke="#333333"
              strokeWidth="8"
            />
            <motion.circle
              cx="50%"
              cy="50%"
              r="45%"
              fill="none"
              stroke="url(#gradient)"
              strokeWidth="8"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: normalized / 100 }}
              transition={{ duration: 1, ease: "easeOut" }}
            />
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="50%" stopColor="#f59e0b" />
                <stop offset="100%" stopColor="#10b981" />
              </linearGradient>
            </defs>
          </svg>
          
          {/* Center Text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.span
              className={`text-3xl font-bold ${getColor()}`}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.5, type: "spring" }}
            >
              {snr.toFixed(1)}
            </motion.span>
            <span className="text-sm text-dark-text-muted">dB</span>
          </div>
        </div>
        
        <div className="text-center">
          <p className="text-lg font-semibold">{getQuality()}</p>
          <p className="text-sm text-dark-text-muted">
            Signal-to-Noise Ratio
          </p>
        </div>
      </div>
    </div>
  )
}

