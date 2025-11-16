import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { motion } from 'framer-motion'

export default function WaveformPlayer({ audioFile, timeline }) {
  const waveformRef = useRef(null)
  const wavesurferRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const hasAudio = audioFile instanceof File || audioFile instanceof Blob
  const objectUrlRef = useRef(null)

  useEffect(() => {
    if (!waveformRef.current || !hasAudio) return

    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#4a5568',
      progressColor: '#a855f7',
      cursorColor: '#ec4899',
      barWidth: 2,
      barRadius: 3,
      responsive: true,
      height: 100,
      normalize: true,
    })

  const url = URL.createObjectURL(audioFile)
  objectUrlRef.current = url
  wavesurfer.load(url)

    wavesurfer.on('play', () => setIsPlaying(true))
    wavesurfer.on('pause', () => setIsPlaying(false))
    wavesurfer.on('timeupdate', (time) => setCurrentTime(time))

    wavesurferRef.current = wavesurfer

    return () => {
      wavesurfer.destroy()
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current)
        objectUrlRef.current = null
      }
    }
  }, [audioFile, hasAudio])

  const togglePlay = () => {
    if (wavesurferRef.current && hasAudio) {
      wavesurferRef.current.playPause()
    }
  }

  const getGenreColor = (genre) => {
    const colors = {
      blues: '#3b82f6',
      classical: '#8b5cf6',
      country: '#10b981',
      disco: '#f59e0b',
      hiphop: '#ef4444',
      jazz: '#ec4899',
      metal: '#6b7280',
      pop: '#f97316',
      reggae: '#22c55e',
      rock: '#dc2626',
    }
    return colors[genre.toLowerCase()] || '#a855f7'
  }

  return (
    <div className="space-y-4">
      <div ref={waveformRef} className="w-full min-h-[120px] flex items-center justify-center">
        {!hasAudio && (
          <p className="text-sm text-dark-text-muted text-center px-4">
            Waveform preview unavailable because the original audio file isn&apos;t stored. Timeline analysis remains below.
          </p>
        )}
      </div>
      
      {/* Timeline Heatmap */}
      <div className="space-y-2">
        <p className="text-sm font-semibold text-dark-text-muted">Timeline Analysis</p>
        <div className="flex gap-1 h-12 rounded-lg overflow-hidden">
          {timeline.map((segment, index) => (
            <motion.div
              key={index}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: index * 0.05 }}
              className="flex-1 relative group cursor-pointer"
              style={{ backgroundColor: getGenreColor(segment.top_genre) }}
              title={`${segment.start}s - ${segment.end}s: ${segment.top_genre} (${(segment.confidence * 100).toFixed(0)}%)`}
            >
              <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors" />
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-white/30" />
            </motion.div>
          ))}
        </div>
        <div className="flex justify-between text-xs text-dark-text-muted">
          <span>0s</span>
          <span>{timeline[timeline.length - 1]?.end || 0}s</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <motion.button
          onClick={togglePlay}
          disabled={!hasAudio}
          className={`w-16 h-16 rounded-full flex items-center justify-center text-white ${
            hasAudio
              ? 'bg-gradient-to-r from-purple-500 to-pink-500'
              : 'bg-dark-border cursor-not-allowed'
          }`}
          whileHover={hasAudio ? { scale: 1.1 } : undefined}
          whileTap={hasAudio ? { scale: 0.9 } : undefined}
        >
          {isPlaying ? (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
              <path d="M6 4h4v12H6V4zm4 0h4v12h-4V4z" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
              <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
            </svg>
          )}
        </motion.button>
        <span className="text-sm text-dark-text-muted">
          {Math.floor(currentTime)}s / {Math.floor(timeline[timeline.length - 1]?.end || 0)}s
        </span>
      </div>
    </div>
  )
}

