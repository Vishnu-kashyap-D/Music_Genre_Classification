import { useState, useRef, useCallback } from 'react'
import { Upload, Mic, Music, X } from 'lucide-react'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import { useTheme } from '../context/ThemeContext'

export default function AudioInput({ onAnalyze }) {
  const { isDark } = useTheme()
  const [isDragging, setIsDragging] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordedAudio, setRecordedAudio] = useState(null)
  const [audioPreview, setAudioPreview] = useState(null)
  const fileInputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const waveformCanvasRef = useRef(null)
  const animationFrameRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('audio/')) {
      handleFile(file)
      toast.success(`✅ File dropped: ${file.name}`, {
        icon: '🎵',
        duration: 2000,
      })
    } else {
      toast.error('Please drop an audio file')
    }
  }, [])

  const handleFile = (file) => {
    // Validate file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
      toast.error('File too large. Maximum size is 50MB')
      return
    }

    // Validate duration (will be checked on backend, but preview here)
    const audio = new Audio()
    audio.src = URL.createObjectURL(file)
    audio.onloadedmetadata = () => {
      if (audio.duration < 3) {
        toast.error('Audio too short. Minimum 3 seconds required.')
        return
      }
      setAudioPreview(file)
    }
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFile(file)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      // Setup audio visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
      analyserRef.current = audioContextRef.current.createAnalyser()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      source.connect(analyserRef.current)
      analyserRef.current.fftSize = 256
      
      mediaRecorderRef.current = new MediaRecorder(stream)
      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' })
        setRecordedAudio(audioFile)
        setAudioPreview(audioFile)
        stream.getTracks().forEach(track => track.stop())
        stopVisualization()
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)
      startVisualization()
      
      // Auto-stop after 10 seconds
      setTimeout(() => {
        if (isRecording) {
          stopRecording()
        }
      }, 10000)
    } catch (error) {
      toast.error('Microphone access denied')
      console.error('Recording error:', error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      stopVisualization()
    }
  }

  const startVisualization = () => {
    const canvas = waveformCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const bufferLength = analyserRef.current.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const draw = () => {
      if (!isRecording) return

      animationFrameRef.current = requestAnimationFrame(draw)
      analyserRef.current.getByteFrequencyData(dataArray)

      ctx.fillStyle = '#1a1a1a'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      const barWidth = canvas.width / bufferLength * 2.5
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height
        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height)
        gradient.addColorStop(0, '#a855f7')
        gradient.addColorStop(1, '#ec4899')
        ctx.fillStyle = gradient
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)
        x += barWidth + 1
      }
    }

    draw()
  }

  const stopVisualization = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    const canvas = waveformCanvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }

  const handleAnalyze = () => {
    if (audioPreview) {
      onAnalyze(audioPreview)
    } else {
      toast.error('Please select or record an audio file first')
    }
  }

  const clearSelection = () => {
    setAudioPreview(null)
    setRecordedAudio(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h2 className="text-3xl font-bold mb-2" style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}>
          Analyze Your Music
        </h2>
        <p style={{ color: isDark ? '#a0a0a0' : '#666666' }}>
          Upload or record audio to discover its genre
        </p>
      </motion.div>

      {/* Drag & Drop Zone */}
      <motion.div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
          className={`
          relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
          transition-all duration-300 overflow-hidden
          ${isDragging 
            ? 'border-purple-400 bg-purple-400/20 scale-105 shadow-2xl shadow-purple-500/50' 
            : isDark 
              ? 'border-dark-border hover:border-purple-400/50 hover:bg-dark-card/50'
              : 'border-light-border hover:border-purple-400/50 hover:bg-light-surface/50'
          }
        `}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {/* Animated background when dragging */}
        {isDragging && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-purple-500/20 via-pink-500/20 to-purple-500/20"
            animate={{
              backgroundPosition: ['0%', '100%'],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              repeatType: 'reverse',
            }}
          />
        )}
        
        <motion.div
          className="relative z-10"
          animate={isDragging ? { scale: 1.1, rotate: 5 } : { scale: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          {audioPreview ? (
            <>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-16 h-16 mx-auto mb-4 bg-green-500 rounded-full flex items-center justify-center"
              >
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </motion.div>
              <p className="text-xl font-semibold mb-2 text-green-400">
                ✅ File Ready: {audioPreview.name}
              </p>
              <p className="text-sm" style={{ color: isDark ? '#a0a0a0' : '#666666' }}>
                Click "Analyze Genre" below to proceed
              </p>
            </>
          ) : (
            <>
              <Upload className={`w-16 h-16 mx-auto mb-4 ${isDragging ? 'text-purple-300' : 'text-purple-400'}`} />
              <p className="text-xl font-semibold mb-2" style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}>
                {isDragging ? '🎵 Drop your music here! 👇' : 'Drop your music here 👇'}
              </p>
              <p style={{ color: isDark ? '#a0a0a0' : '#666666' }}>or click to browse</p>
              <p className="text-sm mt-2" style={{ color: isDark ? '#a0a0a0' : '#666666' }}>
                Supports MP3, WAV, FLAC, M4A (Max 50MB)
              </p>
            </>
          )}
        </motion.div>
      </motion.div>

      {/* Live Recording */}
      <div 
        className="rounded-2xl p-6 border"
        style={{
          backgroundColor: isDark ? '#242424' : '#ffffff',
          borderColor: isDark ? '#333333' : '#e5e5e5',
        }}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Mic className="w-5 h-5 text-pink-400" />
            <span className="font-semibold">Live Recording</span>
          </div>
          {isRecording && (
            <span className="text-sm text-pink-400 animate-pulse">Recording...</span>
          )}
        </div>
        
        <canvas
          ref={waveformCanvasRef}
          className="w-full h-24 mb-4 rounded-lg"
          style={{ backgroundColor: isDark ? '#1a1a1a' : '#f5f5f5' }}
        />
        
        <div className="flex gap-3">
          {!isRecording ? (
            <motion.button
              onClick={startRecording}
              className="flex-1 bg-gradient-to-r from-pink-500 to-purple-500 text-white py-3 rounded-lg font-semibold"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Mic className="w-5 h-5 inline mr-2" />
              Start Recording
            </motion.button>
          ) : (
            <motion.button
              onClick={stopRecording}
              className="flex-1 bg-red-500 text-white py-3 rounded-lg font-semibold"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Stop Recording
            </motion.button>
          )}
        </div>
      </div>

      {/* Audio Preview */}
      {audioPreview && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl p-6 border"
          style={{
            backgroundColor: isDark ? '#242424' : '#ffffff',
            borderColor: isDark ? '#333333' : '#e5e5e5',
          }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Music className="w-5 h-5 text-purple-400" />
              <span className="font-semibold">{audioPreview.name}</span>
            </div>
            <button
              onClick={clearSelection}
              className="p-2 hover:bg-dark-surface rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          <audio
            src={URL.createObjectURL(audioPreview)}
            controls
            className="w-full"
          />
          
          <motion.button
            onClick={handleAnalyze}
            className="w-full mt-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-lg font-semibold"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Analyze Genre
          </motion.button>
        </motion.div>
      )}
    </div>
  )
}

