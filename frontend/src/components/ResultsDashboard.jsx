import { useState } from 'react'
import { RotateCcw } from 'lucide-react'
import { motion } from 'framer-motion'
import RadarChart from './RadarChart'
import ConfidenceBars from './ConfidenceBars'
import WaveformPlayer from './WaveformPlayer'
import SpectrogramView from './SpectrogramView'
import SNRMeter from './SNRMeter'
import ShareButtons from './ShareButtons'
import PDFGenerator from './PDFGenerator'

export default function ResultsDashboard({ results, audioFile, onReset, displayName }) {
  const [activeTab, setActiveTab] = useState('waveform')

  const topGenre = Object.entries(results.global_confidence)
    .sort(([, a], [, b]) => b - a)[0]
  const filenameToDisplay = displayName || audioFile?.name || 'Audio File'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-7xl mx-auto space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-3xl font-bold mb-2">Analysis Results</h2>
          <p className="text-dark-text-muted">
            {filenameToDisplay}
          </p>
        </div>
        <div className="flex gap-3">
          <ShareButtons results={results} filename={filenameToDisplay} />
          <PDFGenerator results={results} filename={filenameToDisplay} />
          <motion.button
            onClick={onReset}
            className="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:bg-dark-surface transition-colors flex items-center gap-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <RotateCcw className="w-4 h-4" />
            New Analysis
          </motion.button>
        </div>
      </div>

      {/* Top Genre Badge */}
      <motion.div
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
        className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/30 rounded-2xl p-6"
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-dark-text-muted mb-1">Top Genre</p>
            <h3 className="text-4xl font-bold capitalize">{topGenre[0]}</h3>
          </div>
          <div className="text-right">
            <p className="text-sm text-dark-text-muted mb-1">Confidence</p>
            <h3 className="text-4xl font-bold">
              {(topGenre[1] * 100).toFixed(1)}%
            </h3>
          </div>
        </div>
      </motion.div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-dark-card rounded-2xl p-6 border border-dark-border"
        >
          <h3 className="text-xl font-bold mb-4">Genre Similarity Map</h3>
          <RadarChart data={results.global_confidence} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-dark-card rounded-2xl p-6 border border-dark-border"
        >
          <h3 className="text-xl font-bold mb-4">Confidence Scores</h3>
          <ConfidenceBars data={results.global_confidence} />
        </motion.div>
      </div>

      {/* Audio Player & Spectrogram */}
      <div className="bg-dark-card rounded-2xl p-6 border border-dark-border">
        <div className="flex gap-2 mb-4 border-b border-dark-border">
          <button
            onClick={() => setActiveTab('waveform')}
            className={`px-4 py-2 font-semibold transition-colors ${
              activeTab === 'waveform'
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-dark-text-muted hover:text-dark-text'
            }`}
          >
            Waveform
          </button>
          <button
            onClick={() => setActiveTab('spectrogram')}
            className={`px-4 py-2 font-semibold transition-colors ${
              activeTab === 'spectrogram'
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-dark-text-muted hover:text-dark-text'
            }`}
          >
            Spectrogram
          </button>
        </div>

        {activeTab === 'waveform' ? (
          <WaveformPlayer
            audioFile={audioFile}
            timeline={results.timeline}
          />
        ) : (
          <SpectrogramView image={results.spectrogram_image} />
        )}
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SNRMeter snr={results.metrics.snr} />
        
        <div className="bg-dark-card rounded-2xl p-6 border border-dark-border">
          <h3 className="text-xl font-bold mb-4">Audio Information</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-dark-text-muted">Duration</span>
              <span className="font-semibold">
                {results.metrics.duration.toFixed(2)}s
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-dark-text-muted">Segments Analyzed</span>
              <span className="font-semibold">{results.metrics.num_segments}</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

