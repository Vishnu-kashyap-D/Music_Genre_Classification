import { Share2, MessageCircle, Instagram, Twitter } from 'lucide-react'
import { motion } from 'framer-motion'
import html2canvas from 'html2canvas'
import toast from 'react-hot-toast'

export default function ShareButtons({ results, filename }) {
  const topGenre = Object.entries(results.global_confidence)
    .sort(([, a], [, b]) => b - a)[0]

  const shareText = `🎵 Just analyzed "${filename}" - Top Genre: ${topGenre[0].toUpperCase()} (${(topGenre[1] * 100).toFixed(1)}% confidence)!\n\n#MusicGenre #AI #MusicAnalysis`

  const shareWhatsApp = () => {
    const url = `https://wa.me/?text=${encodeURIComponent(shareText)}`
    window.open(url, '_blank')
  }

  const shareTwitter = () => {
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}`
    window.open(url, '_blank')
  }

  const downloadForInstagram = async () => {
    try {
      toast.loading('Generating image...', { id: 'instagram' })
      
      // Create a canvas with the results
      const canvas = document.createElement('canvas')
      canvas.width = 1080
      canvas.height = 1080
      const ctx = canvas.getContext('2d')
      
      // Background
      ctx.fillStyle = '#1a1a1a'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      // Title
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 48px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('🎵 Music Genre Analysis', canvas.width / 2, 100)
      
      // Top Genre
      ctx.font = 'bold 72px Arial'
      ctx.fillStyle = '#a855f7'
      ctx.fillText(topGenre[0].toUpperCase(), canvas.width / 2, 250)
      
      // Confidence
      ctx.font = '48px Arial'
      ctx.fillStyle = '#ffffff'
      ctx.fillText(`${(topGenre[1] * 100).toFixed(1)}% Confidence`, canvas.width / 2, 320)
      
      // Spectrogram
      const img = new Image()
      img.src = `data:image/png;base64,${results.spectrogram_image}`
      
      img.onload = () => {
        ctx.drawImage(img, 100, 400, 880, 400)
        
        // Download
        canvas.toBlob((blob) => {
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `${filename}_analysis.png`
          a.click()
          URL.revokeObjectURL(url)
          toast.success('Image downloaded!', { id: 'instagram' })
        })
      }
    } catch (error) {
      toast.error('Failed to generate image', { id: 'instagram' })
      console.error(error)
    }
  }

  return (
    <div className="flex gap-2">
      <motion.button
        onClick={shareWhatsApp}
        className="px-4 py-2 bg-green-500/20 hover:bg-green-500/30 border border-green-500/50 rounded-lg transition-colors flex items-center gap-2"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        title="Share on WhatsApp"
      >
        <MessageCircle className="w-4 h-4" />
        <span className="hidden sm:inline">WhatsApp</span>
      </motion.button>
      
      <motion.button
        onClick={downloadForInstagram}
        className="px-4 py-2 bg-pink-500/20 hover:bg-pink-500/30 border border-pink-500/50 rounded-lg transition-colors flex items-center gap-2"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        title="Download for Instagram"
      >
        <Instagram className="w-4 h-4" />
        <span className="hidden sm:inline">Instagram</span>
      </motion.button>
      
      <motion.button
        onClick={shareTwitter}
        className="px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/50 rounded-lg transition-colors flex items-center gap-2"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        title="Share on Twitter"
      >
        <Twitter className="w-4 h-4" />
        <span className="hidden sm:inline">Twitter</span>
      </motion.button>
    </div>
  )
}

