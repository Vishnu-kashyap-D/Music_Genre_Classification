import { Download } from 'lucide-react'
import { motion } from 'framer-motion'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'
import toast from 'react-hot-toast'

export default function PDFGenerator({ results, filename }) {
  const generatePDF = async () => {
    try {
      toast.loading('Generating PDF report...', { id: 'pdf' })
      
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()
      
      // Title
      pdf.setFontSize(24)
      pdf.setTextColor(168, 85, 247)
      pdf.text('Music Genre Analysis Report', pageWidth / 2, 20, { align: 'center' })
      
      // File info
      pdf.setFontSize(12)
      pdf.setTextColor(160, 160, 160)
      pdf.text(`File: ${filename}`, 20, 35)
      pdf.text(`Date: ${new Date().toLocaleDateString()}`, 20, 42)
      
      // Top Genre
      const topGenre = Object.entries(results.global_confidence)
        .sort(([, a], [, b]) => b - a)[0]
      
      pdf.setFontSize(18)
      pdf.setTextColor(255, 255, 255)
      pdf.text(`Top Genre: ${topGenre[0].toUpperCase()}`, 20, 60)
      pdf.setFontSize(14)
      pdf.text(`Confidence: ${(topGenre[1] * 100).toFixed(1)}%`, 20, 70)
      
      // Metrics
      pdf.setFontSize(12)
      pdf.text(`Duration: ${results.metrics.duration.toFixed(2)}s`, 20, 85)
      pdf.text(`SNR: ${results.metrics.snr.toFixed(1)} dB`, 20, 92)
      pdf.text(`Segments Analyzed: ${results.metrics.num_segments}`, 20, 99)
      
      // Genre Confidence Table
      pdf.setFontSize(14)
      pdf.text('Genre Confidence Scores:', 20, 115)
      
      const sortedGenres = Object.entries(results.global_confidence)
        .sort(([, a], [, b]) => b - a)
      
      let yPos = 125
      pdf.setFontSize(10)
      sortedGenres.forEach(([genre, confidence], index) => {
        if (yPos > pageHeight - 20) {
          pdf.addPage()
          yPos = 20
        }
        pdf.setTextColor(255, 255, 255)
        pdf.text(`${index + 1}. ${genre.charAt(0).toUpperCase() + genre.slice(1)}`, 25, yPos)
        pdf.setTextColor(160, 160, 160)
        pdf.text(`${(confidence * 100).toFixed(1)}%`, pageWidth - 20, yPos, { align: 'right' })
        yPos += 8
      })
      
      // Spectrogram
      if (results.spectrogram_image) {
        const img = new Image()
        img.src = `data:image/png;base64,${results.spectrogram_image}`
        
        await new Promise((resolve) => {
          img.onload = () => {
            const imgWidth = pageWidth - 40
            const imgHeight = (img.height / img.width) * imgWidth
            
            if (yPos + imgHeight > pageHeight - 20) {
              pdf.addPage()
              yPos = 20
            }
            
            pdf.addImage(
              `data:image/png;base64,${results.spectrogram_image}`,
              'PNG',
              20,
              yPos,
              imgWidth,
              imgHeight
            )
            resolve()
          }
        })
      }
      
      // Save
      pdf.save(`${filename}_analysis_report.pdf`)
      toast.success('PDF report downloaded!', { id: 'pdf' })
    } catch (error) {
      toast.error('Failed to generate PDF', { id: 'pdf' })
      console.error(error)
    }
  }

  return (
    <motion.button
      onClick={generatePDF}
      className="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:bg-dark-surface transition-colors flex items-center gap-2"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <Download className="w-4 h-4" />
      <span className="hidden sm:inline">PDF Report</span>
    </motion.button>
  )
}

