import AudioInput from '../components/AudioInput'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'

export default function Home() {
  const navigate = useNavigate()
  const { isAuthenticated } = useAuth()

  const handleAnalyze = async (file) => {
    // Warn if not logged in (history won't be saved)
    if (!isAuthenticated) {
      const proceed = window.confirm(
        'You are not logged in. Your analysis will not be saved to history. Continue anyway?'
      )
      if (!proceed) {
        return
      }
    }
    
    // Navigate to analysis page with file
    navigate('/analysis', { state: { file } })
  }

  return (
    <div className="max-w-6xl mx-auto">
      <AudioInput onAnalyze={handleAnalyze} />
    </div>
  )
}

