import { useState, useEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import Loader from '../components/Loader'
import ResultsDashboard from '../components/ResultsDashboard'
import { useAuth } from '../context/AuthContext'
import { saveToUserHistory } from '../utils/historyStorage'
import { API_ENDPOINTS } from '../config/api'

export default function Analysis() {
  const location = useLocation()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [analysisState, setAnalysisState] = useState('loading')
  const [results, setResults] = useState(null)
  const [audioFile, setAudioFile] = useState(location.state?.file || null)
  const [analysisFilename, setAnalysisFilename] = useState(
    location.state?.file?.name || location.state?.filename || ''
  )
  const [errorMessage, setErrorMessage] = useState('')
  const isAnalyzingRef = useRef(false) // Prevent multiple simultaneous analyses

  useEffect(() => {
    // Check if loading from history first
    if (location.state?.results && location.state?.fromHistory) {
      // If results are passed from history, use them directly
      console.log('Loading from history - State:', location.state)
      console.log('Results data:', location.state.results)
      
      if (!location.state.results || !location.state.results.global_confidence) {
        console.error('Invalid results in state:', location.state)
        toast.error('Failed to load history data')
        navigate('/history')
        return
      }
      
      setResults(location.state.results)
      const historicalName = location.state.filename || location.state.results?.filename || ''
      if (historicalName) {
        setAnalysisFilename(historicalName)
      }
      setAudioFile(null)
      setAnalysisState('results')
      isAnalyzingRef.current = false
      return
    }

    // Otherwise, analyze the file
    if (!audioFile) {
      navigate('/')
      return
    }

    // Prevent multiple simultaneous analyses
    if (isAnalyzingRef.current) {
      console.log('Analysis already in progress, skipping...')
      return
    }

    isAnalyzingRef.current = true
    analyzeFile(audioFile).finally(() => {
      isAnalyzingRef.current = false
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state])

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.health, {
        method: 'GET',
        signal: AbortSignal.timeout(5000), // 5 second timeout for health check
      })
      const data = await response.json()
      return data.status === 'healthy' || data.status === 'degraded'
    } catch (error) {
      console.error('Health check failed:', error)
      return false
    }
  }

  const analyzeFile = async (file, retryCount = 0) => {
    setAnalysisState('loading')
    setAnalysisFilename(file?.name || '')
    setErrorMessage('')
    
    const maxRetries = 2
    const timeoutDuration = 300000 // 5 minutes (300 seconds) - increased from 90 seconds
    const errorToastId = 'analysis-error' // Use consistent ID to prevent duplicates

    try {
      // Check backend health before making request
      if (retryCount === 0) {
        toast.loading('Checking backend connection...', { id: 'health-check' })
        const isHealthy = await checkBackendHealth()
        toast.dismiss('health-check')
        
        if (!isHealthy) {
          const errorMsg = `Backend server is not responding. Please ensure the backend is running:\n\n1. Open a terminal\n2. Navigate to the project directory\n3. Run: python app.py\n\nThe server should be running on ${API_ENDPOINTS.base}`
          setErrorMessage(errorMsg)
          setAnalysisState('error')
          // Dismiss any existing error toasts and show only one
          toast.dismiss(errorToastId)
          toast.error(errorMsg, { id: errorToastId, duration: 8000 })
          return
        }
      }

      toast.loading(`Analyzing audio${retryCount > 0 ? ` (retry ${retryCount}/${maxRetries})` : ''}...`, { id: 'analysis' })
      
      const formData = new FormData()
      formData.append('audio', file)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), timeoutDuration)

      try {
        const response = await fetch(API_ENDPOINTS.predict, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        })
        clearTimeout(timeoutId)
        toast.dismiss('analysis')

        if (!response.ok) {
          let errorData
          try {
            errorData = await response.json()
          } catch {
            errorData = { error: `Server returned status ${response.status}` }
          }
          throw new Error(errorData.error || `Analysis failed with status ${response.status}`)
        }

        const data = await response.json()
        console.log('Analysis complete, saving to history for user:', user?.id)
        
        setResults(data)
        setAnalysisState('results')
        toast.success('Analysis complete!', { duration: 2000 })
        
        // Save to user-specific history (skip guest sessions)
        if (user?.id && !user?.isGuest) {
          try {
            const saved = saveToUserHistory(user.id, file.name, data, user)
            if (saved) {
              console.log('History saved successfully:', saved.id)
            } else {
              console.warn('Failed to save history - no user ID')
            }
          } catch (error) {
            console.error('Error saving history:', error)
          }
        } else {
          console.warn('History not saved - guest or logged-out user')
          toast(user?.isGuest ? 'Guest mode does not store history' : 'Please login to save analysis history', {
            duration: 3000,
            icon: user?.isGuest ? 'ℹ️' : undefined,
          })
        }
      } catch (fetchError) {
        clearTimeout(timeoutId)
        toast.dismiss('analysis')
        throw fetchError
      }
    } catch (error) {
      console.error('Error:', error)
      let friendlyMessage = error.message || 'Analysis failed'
      
      if (error.name === 'AbortError') {
        friendlyMessage = 'Analysis request timed out after 5 minutes. The file might be too large or the server is slow. Please try with a smaller file or check your connection.'
      } else if (friendlyMessage.includes('Failed to fetch') || friendlyMessage.includes('NetworkError')) {
        friendlyMessage = `Unable to reach the analysis server at ${API_ENDPOINTS.base}.\n\nPlease ensure:\n1. The backend is running (python app.py)\n2. The server is accessible from your browser\n3. There are no firewall issues blocking the connection`
        
        // Retry if it's a network error and we haven't exceeded max retries
        if (retryCount < maxRetries) {
          toast.dismiss(errorToastId) // Dismiss any existing error toasts
          toast.error(`Connection failed. Retrying... (${retryCount + 1}/${maxRetries})`, { id: 'retry-toast', duration: 3000 })
          await new Promise(resolve => setTimeout(resolve, 2000)) // Wait 2 seconds before retry
          return analyzeFile(file, retryCount + 1)
        }
      } else if (friendlyMessage.includes('timeout')) {
        friendlyMessage = 'The request took too long. Please try with a smaller audio file or check your connection.'
      }
      
      setErrorMessage(friendlyMessage)
      setAnalysisState('error')
      // Dismiss any existing error toasts and show only one
      toast.dismiss(errorToastId)
      toast.dismiss('retry-toast')
      toast.error(friendlyMessage, { id: errorToastId, duration: 8000 })
    }
  }


  const handleReset = () => {
    navigate('/')
  }

  if (analysisState === 'loading') {
    return <Loader />
  }

  if (analysisState === 'error') {
    return (
      <div className="max-w-4xl mx-auto text-center py-12 px-4">
        <div className="mb-6">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-3xl font-bold mb-4">Analysis Failed</h2>
          {errorMessage && (
            <div className="mb-6">
              <p className="text-gray-400 whitespace-pre-line text-left bg-gray-800 p-4 rounded-lg border border-red-500/30">
                {errorMessage}
              </p>
            </div>
          )}
        </div>
        <div className="flex gap-4 justify-center">
          <button
            onClick={handleReset}
            className="px-6 py-3 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
          >
            Go Back
          </button>
          {audioFile && (
            <button
              onClick={() => analyzeFile(audioFile)}
              className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
            >
              Retry Analysis
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <ResultsDashboard 
      results={results}
      audioFile={audioFile}
      displayName={analysisFilename || audioFile?.name}
      onReset={handleReset}
    />
  )
}

