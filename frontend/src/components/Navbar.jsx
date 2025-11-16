import { Link, useNavigate, useLocation } from 'react-router-dom'
import { Moon, Sun, Menu, User, LogOut, History, Home, Music } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'
import { useAuth } from '../context/AuthContext'
import { useState } from 'react'

export default function Navbar() {
  const { isDark, toggleTheme } = useTheme()
  const { user, logout, isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [showUserMenu, setShowUserMenu] = useState(false)

  const handleLogout = () => {
    logout()
    navigate('/login')
    setShowUserMenu(false)
  }

  const isActive = (path) => location.pathname === path

  return (
    <nav 
      className="sticky top-0 z-50 backdrop-blur-md border-b transition-colors"
      style={{
        backgroundColor: isDark ? 'rgba(26, 26, 26, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        borderColor: isDark ? '#333333' : '#e5e5e5',
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-2">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <Music className="w-6 h-6 text-purple-400" />
              </motion.div>
              <motion.h1 
                className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                Music Genre Classifier
              </motion.h1>
            </Link>
          </div>

          <div className="flex items-center gap-2">
            {/* Navigation Links */}
            <div className="hidden md:flex items-center gap-1">
              <Link
                to="/"
                className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                  isActive('/')
                    ? isDark ? 'bg-purple-500/20 text-purple-400' : 'bg-purple-100 text-purple-600'
                    : isDark ? 'hover:bg-dark-card text-dark-text' : 'hover:bg-light-surface text-light-text'
                }`}
              >
                <Home className="w-4 h-4" />
                Home
              </Link>
              <Link
                to="/history"
                className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                  isActive('/history')
                    ? isDark ? 'bg-purple-500/20 text-purple-400' : 'bg-purple-100 text-purple-600'
                    : isDark ? 'hover:bg-dark-card text-dark-text' : 'hover:bg-light-surface text-light-text'
                }`}
              >
                <History className="w-4 h-4" />
                History
              </Link>
            </div>

            {/* Theme Toggle */}
            <motion.button
              onClick={toggleTheme}
              className="p-2 rounded-lg transition-colors"
              style={{
                backgroundColor: isDark ? 'rgba(36, 36, 36, 0.5)' : 'rgba(245, 245, 245, 0.5)',
              }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              aria-label="Toggle theme"
            >
              {isDark ? (
                <Sun className="w-5 h-5 text-yellow-400" />
              ) : (
                <Moon className="w-5 h-5 text-blue-400" />
              )}
            </motion.button>

            {/* User Menu */}
            {isAuthenticated ? (
              <div className="relative">
                <motion.button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg transition-colors"
                  style={{
                    backgroundColor: isDark ? 'rgba(36, 36, 36, 0.5)' : 'rgba(245, 245, 245, 0.5)',
                  }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {user?.avatar && (
                    <img
                      src={user.avatar}
                      alt={user.name}
                      className="w-8 h-8 rounded-full"
                      onError={(e) => {
                        // Fallback to initials if avatar fails to load
                        e.target.style.display = 'none'
                        const fallback = e.target.nextSibling
                        if (!fallback || fallback.className !== 'avatar-fallback') {
                          const span = document.createElement('span')
                          span.className = 'avatar-fallback w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold bg-purple-500 text-white'
                          span.textContent = user.name?.charAt(0).toUpperCase() || 'U'
                          e.target.parentNode.insertBefore(span, e.target.nextSibling)
                        }
                      }}
                    />
                  )}
                  {!user?.avatar && (
                    <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold bg-purple-500 text-white">
                      {user?.name?.charAt(0).toUpperCase() || 'U'}
                    </div>
                  )}
                  <span className="hidden sm:inline" style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}>
                    {user?.name}
                  </span>
                </motion.button>

                <AnimatePresence>
                  {showUserMenu && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute right-0 mt-2 w-48 rounded-lg shadow-lg border overflow-hidden"
                      style={{
                        backgroundColor: isDark ? '#242424' : '#ffffff',
                        borderColor: isDark ? '#333333' : '#e5e5e5',
                      }}
                    >
                      <Link
                        to="/profile"
                        onClick={() => setShowUserMenu(false)}
                        className="flex items-center gap-2 px-4 py-3 hover:bg-opacity-50 transition-colors"
                        style={{
                          color: isDark ? '#ffffff' : '#1a1a1a',
                          backgroundColor: isDark ? 'rgba(36, 36, 36, 0.5)' : 'rgba(245, 245, 245, 0.5)',
                        }}
                      >
                        <User className="w-4 h-4" />
                        Profile
                      </Link>
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-opacity-50 transition-colors text-red-400"
                        style={{
                          backgroundColor: isDark ? 'rgba(36, 36, 36, 0.5)' : 'rgba(245, 245, 245, 0.5)',
                        }}
                      >
                        <LogOut className="w-4 h-4" />
                        Logout
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <Link
                to="/login"
                className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold transition-opacity hover:opacity-90"
              >
                Login
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
