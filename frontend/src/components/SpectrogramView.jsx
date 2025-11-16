import { motion } from 'framer-motion'

export default function SpectrogramView({ image }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="w-full"
    >
      <img
        src={`data:image/png;base64,${image}`}
        alt="Mel-Spectrogram"
        className="w-full h-auto rounded-lg border border-dark-border"
      />
      <p className="text-sm text-dark-text-muted mt-2 text-center">
        Mel-Spectrogram visualization showing frequency content over time
      </p>
    </motion.div>
  )
}

