import { X, History, Clock } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { getUserHistory } from '../utils/historyStorage'

export default function Sidebar({ isOpen, onClose, onLoadHistory }) {
  const [history, setHistory] = useState([])
  const { user, isAuthenticated } = useAuth()

  useEffect(() => {
    if (isAuthenticated && user?.id) {
      loadHistory()
    } else {
      setHistory([])
    }
  }, [user, isAuthenticated])

  // Listen for history updates
  useEffect(() => {
    const handleStorageUpdate = () => {
      if (isAuthenticated && user?.id) {
        loadHistory()
      }
    }
    
    window.addEventListener('storage', handleStorageUpdate)
    window.addEventListener('historyUpdated', handleStorageUpdate)
    
    return () => {
      window.removeEventListener('storage', handleStorageUpdate)
      window.removeEventListener('historyUpdated', handleStorageUpdate)
    }
  }, [user, isAuthenticated])

  const loadHistory = () => {
    if (!user?.id) {
      setHistory([])
      return
    }
  const userHistory = getUserHistory(user.id, user.email)
    // Filter valid items and show last 5
    const validHistory = userHistory.filter(item => item.data && item.filename)
    setHistory(validHistory.slice(0, 5))
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

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          />
          
          {/* Sidebar */}
          <motion.aside
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed left-0 top-0 h-full w-80 bg-dark-surface border-r border-dark-border z-50 overflow-y-auto"
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <History className="w-5 h-5 text-purple-400" />
                  <h2 className="text-xl font-bold">History</h2>
                </div>
                <button
                  onClick={onClose}
                  className="lg:hidden p-2 hover:bg-dark-card rounded-lg transition-colors"
                  aria-label="Close sidebar"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {!isAuthenticated ? (
                <div className="text-center py-8 text-dark-text-muted">
                  <History className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="text-sm">Login to view history</p>
                </div>
              ) : history.length === 0 ? (
                <div className="text-center py-12 text-dark-text-muted">
                  <History className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No analysis history yet</p>
                  <p className="text-sm mt-2">Your recent analyses will appear here</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {history.map((item) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: history.indexOf(item) * 0.1 }}
                      onClick={() => {
                        if (item.data && item.data.global_confidence) {
                          onLoadHistory(item)
                          onClose()
                        } else {
                          console.error('Invalid history item:', item)
                        }
                      }}
                      className="bg-dark-card border border-dark-border rounded-lg p-4 cursor-pointer hover:bg-dark-surface transition-colors group"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h3 className="font-semibold text-sm truncate flex-1">
                          {item.filename}
                        </h3>
                      </div>
                      
                      <div className="flex items-center gap-2 mb-2">
                        <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs font-semibold capitalize">
                          {item.topGenre}
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-1 text-xs text-dark-text-muted">
                        <Clock className="w-3 h-3" />
                        <span>{formatDate(item.timestamp)}</span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

