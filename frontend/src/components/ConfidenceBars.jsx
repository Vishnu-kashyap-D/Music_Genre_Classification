import { motion } from 'framer-motion'

export default function ConfidenceBars({ data }) {
  const sorted = Object.entries(data)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 9) // Top 9 genres

  const getColor = (value) => {
    if (value > 0.75) return 'from-green-500 to-emerald-500'
    if (value > 0.40) return 'from-yellow-500 to-orange-500'
    return 'from-red-500 to-pink-500'
  }

  return (
    <div className="space-y-4">
      {sorted.map(([genre, value], index) => (
        <motion.div
          key={genre}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.05 }}
          className="space-y-2"
        >
          <div className="flex justify-between items-center">
            <span className="font-semibold capitalize">{genre}</span>
            <span className="text-sm text-dark-text-muted">
              {(value * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-3 bg-dark-surface rounded-full overflow-hidden">
            <motion.div
              className={`h-full bg-gradient-to-r ${getColor(value)} rounded-full`}
              initial={{ width: 0 }}
              animate={{ width: `${value * 100}%` }}
              transition={{ duration: 0.8, delay: index * 0.05 }}
            />
          </div>
        </motion.div>
      ))}
    </div>
  )
}

