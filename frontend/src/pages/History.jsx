import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { History as HistoryIcon, Clock, Music, Trash2, LogIn } from 'lucide-react'
import { motion } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import { useAuth } from '../context/AuthContext'
import { getUserHistory, deleteHistoryItem, clearUserHistory } from '../utils/historyStorage'
import { Link } from 'react-router-dom'
import toast from 'react-hot-toast'

export default function History() {
  const [history, setHistory] = useState([])
  const navigate = useNavigate()
  const { isDark } = useTheme()
  const { user, isAuthenticated } = useAuth()
  const isGuest = user?.isGuest

  useEffect(() => {
    if (isAuthenticated && user?.id && !isGuest) {
      loadHistory()
    } else {
      setHistory([])
    }
  }, [user, isAuthenticated, isGuest])

  // Also listen for custom storage events (for same-tab updates)
  useEffect(() => {
    const handleStorageUpdate = () => {
      if (isAuthenticated && user?.id && !isGuest) {
        loadHistory()
      }
    }
    
    // Listen for storage events
    window.addEventListener('storage', handleStorageUpdate)
    
    // Also listen for custom event we'll dispatch when saving
    window.addEventListener('historyUpdated', handleStorageUpdate)
    
    return () => {
      window.removeEventListener('storage', handleStorageUpdate)
      window.removeEventListener('historyUpdated', handleStorageUpdate)
    }
  }, [user, isAuthenticated])

  const loadHistory = () => {
    if (!user?.id || isGuest) {
      console.log('No user ID, clearing history')
      setHistory([])
      return
    }
    
    console.log('Loading history for user:', user.id, user.email)
  const userHistory = getUserHistory(user.id, user.email)
    console.log('Found history items:', userHistory.length)
    
    // Filter out any items without data (safety check)
    const validHistory = userHistory.filter(item => item.data && item.filename)
    console.log('Valid history items:', validHistory.length)
    
    setHistory(validHistory)
  }

  const formatDate = (timestamp) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now - date
    
    if (diff < 60000) return 'Just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
    return date.toLocaleDateString()
  }

  const handleLoad = (item) => {
    console.log('Loading history item:', item)
    
    if (!item.data) {
      console.error('History item missing data:', item)
      toast.error('History item is missing analysis data')
      return
    }
    
    // Verify data structure
    if (!item.data.global_confidence || !item.data.timeline) {
      console.error('Invalid data structure:', item.data)
      toast.error('History data is corrupted')
      return
    }
    
    console.log('Navigating to analysis with data:', {
      hasResults: !!item.data,
      hasGlobalConfidence: !!item.data.global_confidence,
      hasTimeline: !!item.data.timeline,
      filename: item.filename
    })
    
    navigate('/analysis', { 
      state: { 
        file: null, 
        results: item.data, 
        filename: item.filename,
        fromHistory: true 
      },
      replace: false
    })
  }

  const handleDelete = (itemId) => {
    if (!user?.id || isGuest) return
    
    const updated = deleteHistoryItem(user.id, itemId, user.email)
    setHistory(updated)
  }

  const clearAll = () => {
    if (!user?.id || isGuest) return
    
    if (window.confirm('Clear all your analysis history?')) {
  clearUserHistory(user.id, user.email)
      setHistory([])
    }
  }

  if (!isAuthenticated || isGuest) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`text-center py-12 ${isDark ? 'bg-dark-card' : 'bg-light-card'} rounded-2xl border ${isDark ? 'border-dark-border' : 'border-light-border'}`}
        >
          <LogIn className={`w-16 h-16 mx-auto mb-4 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'} opacity-50`} />
          <p className={`text-xl mb-2 ${isDark ? 'text-dark-text' : 'text-light-text'}`}>Please login to view your history</p>
          <p className={`mb-4 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}`}>
            {isGuest
              ? 'Guest sessions can analyze music but history is only stored for registered accounts.'
              : 'Your analysis history is saved per user account'}
          </p>
          <Link
            to="/login"
            className="inline-block px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-semibold hover:opacity-90 transition-opacity"
          >
            Go to Login
          </Link>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <HistoryIcon className={`w-8 h-8 ${isDark ? 'text-purple-400' : 'text-purple-600'}`} />
          <div>
            <h1 className="text-3xl font-bold" style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}>
              Analysis History
            </h1>
            <p className="text-sm" style={{ color: isDark ? '#a0a0a0' : '#666666' }}>
              {user?.name}'s analyses
            </p>
          </div>
        </div>
        {history.length > 0 && (
          <button
            onClick={clearAll}
            className={`px-4 py-2 rounded-lg border ${isDark ? 'border-red-500/50 text-red-400 hover:bg-red-500/20' : 'border-red-500 text-red-600 hover:bg-red-50'} transition-colors`}
          >
            Clear All
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`text-center py-12 ${isDark ? 'bg-dark-card' : 'bg-light-card'} rounded-2xl border ${isDark ? 'border-dark-border' : 'border-light-border'}`}
        >
          <HistoryIcon className={`w-16 h-16 mx-auto mb-4 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'} opacity-50`} />
          <p className={`text-xl mb-2 ${isDark ? 'text-dark-text' : 'text-light-text'}`}>No analysis history yet</p>
          <p className={isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}>
            Your recent analyses will appear here
          </p>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {history.map((item, index) => (
            <motion.div
              key={item.id || index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`${isDark ? 'bg-dark-card' : 'bg-light-card'} rounded-xl p-6 border ${isDark ? 'border-dark-border' : 'border-light-border'} hover:shadow-lg transition-all cursor-pointer group`}
              onClick={() => handleLoad(item)}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-purple-500/20 rounded-lg">
                    <Music className="w-5 h-5 text-purple-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className={`font-semibold truncate ${isDark ? 'text-dark-text' : 'text-light-text'}`}>
                      {item.filename}
                    </h3>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(item.id)
                  }}
                  className={`opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 transition-opacity`}
                >
                  <Trash2 className="w-4 h-4 text-red-400" />
                </button>
              </div>
              
              <div className="flex items-center gap-2 mb-3">
                <span className={`px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm font-semibold capitalize`}>
                  {item.topGenre}
                </span>
              </div>
              
              <div className="flex items-center gap-1 text-sm" style={{ color: isDark ? '#a0a0a0' : '#666666' }}>
                <Clock className="w-3 h-3" />
                <span>{formatDate(item.timestamp)}</span>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}

