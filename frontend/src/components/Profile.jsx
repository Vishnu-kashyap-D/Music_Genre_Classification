import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { User, Mail, Calendar, Save, Edit2, Camera } from 'lucide-react'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import { useTheme } from '../context/ThemeContext'
import { useNavigate } from 'react-router-dom'

export default function Profile() {
  const { user, updateProfile, logout } = useAuth()
  const [isEditing, setIsEditing] = useState(false)
  const [name, setName] = useState(user?.name || '')
  const [email, setEmail] = useState(user?.email || '')
  const navigate = useNavigate()
  const { isDark } = useTheme()

  if (!user) {
    navigate('/login')
    return null
  }

  const handleSave = () => {
    updateProfile({ name, email })
    setIsEditing(false)
    toast.success('Profile updated successfully!')
  }

  const handleLogout = () => {
    logout()
    toast.success('Logged out successfully')
    navigate('/login')
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`${isDark ? 'bg-dark-card' : 'bg-light-card'} rounded-2xl p-8 border ${isDark ? 'border-dark-border' : 'border-light-border'}`}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold" style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}>
            Profile Settings
          </h2>
          {!isEditing && (
            <motion.button
              onClick={() => setIsEditing(true)}
              className={`p-2 rounded-lg ${isDark ? 'bg-dark-surface hover:bg-dark-border' : 'bg-light-surface hover:bg-light-border'} transition-colors`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Edit2 className={`w-5 h-5 ${isDark ? 'text-dark-text' : 'text-light-text'}`} />
            </motion.button>
          )}
        </div>

        <div className="flex flex-col items-center mb-8">
          <div className="relative mb-4">
            <img
              src={user.avatar}
              alt={user.name}
              className="w-32 h-32 rounded-full border-4 border-purple-500"
              onError={(e) => {
                // Fallback to initials if avatar fails to load
                e.target.style.display = 'none'
                const fallback = e.target.nextSibling
                if (!fallback || !fallback.classList.contains('avatar-fallback')) {
                  const div = document.createElement('div')
                  div.className = 'avatar-fallback w-32 h-32 rounded-full border-4 border-purple-500 flex items-center justify-center text-4xl font-semibold bg-purple-500 text-white'
                  div.textContent = user.name?.charAt(0).toUpperCase() || 'U'
                  e.target.parentNode.insertBefore(div, e.target.nextSibling)
                }
              }}
            />
            {!user?.avatar && (
              <div className="w-32 h-32 rounded-full border-4 border-purple-500 flex items-center justify-center text-4xl font-semibold bg-purple-500 text-white">
                {user?.name?.charAt(0).toUpperCase() || 'U'}
              </div>
            )}
            {isEditing && (
              <motion.button
                className="absolute bottom-0 right-0 p-2 bg-purple-500 rounded-full text-white"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Camera className="w-4 h-4" />
              </motion.button>
            )}
          </div>
          {!isEditing && (
            <h3 className="text-2xl font-bold" style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}>
              {user.name}
            </h3>
          )}
        </div>

        <div className="space-y-6">
          <div>
            <label className={`block text-sm font-semibold mb-2 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}`}>
              <User className="w-4 h-4 inline mr-2" />
              Name
            </label>
            {isEditing ? (
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className={`w-full px-4 py-3 rounded-lg border ${isDark ? 'bg-dark-surface border-dark-border text-dark-text' : 'bg-light-surface border-light-border text-light-text'} focus:outline-none focus:ring-2 focus:ring-purple-500`}
              />
            ) : (
              <p className={`px-4 py-3 rounded-lg ${isDark ? 'bg-dark-surface text-dark-text' : 'bg-light-surface text-light-text'}`}>
                {user.name}
              </p>
            )}
          </div>

          <div>
            <label className={`block text-sm font-semibold mb-2 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}`}>
              <Mail className="w-4 h-4 inline mr-2" />
              Email
            </label>
            {isEditing ? (
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`w-full px-4 py-3 rounded-lg border ${isDark ? 'bg-dark-surface border-dark-border text-dark-text' : 'bg-light-surface border-light-border text-light-text'} focus:outline-none focus:ring-2 focus:ring-purple-500`}
              />
            ) : (
              <p className={`px-4 py-3 rounded-lg ${isDark ? 'bg-dark-surface text-dark-text' : 'bg-light-surface text-light-text'}`}>
                {user.email}
              </p>
            )}
          </div>

          <div>
            <label className={`block text-sm font-semibold mb-2 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}`}>
              <Calendar className="w-4 h-4 inline mr-2" />
              Member Since
            </label>
            <p className={`px-4 py-3 rounded-lg ${isDark ? 'bg-dark-surface text-dark-text' : 'bg-light-surface text-light-text'}`}>
              {new Date(user.createdAt).toLocaleDateString()}
            </p>
          </div>

          {isEditing && (
            <div className="flex gap-3">
              <motion.button
                onClick={handleSave}
                className="flex-1 bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-lg font-semibold flex items-center justify-center gap-2"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Save className="w-5 h-5" />
                Save Changes
              </motion.button>
              <motion.button
                onClick={() => {
                  setIsEditing(false)
                  setName(user.name)
                  setEmail(user.email)
                }}
                className={`px-6 py-3 rounded-lg border ${isDark ? 'border-dark-border text-dark-text hover:bg-dark-surface' : 'border-light-border text-light-text hover:bg-light-surface'} transition-colors`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Cancel
              </motion.button>
            </div>
          )}
        </div>

        <div className="mt-8 pt-8 border-t" style={{ borderColor: isDark ? '#333333' : '#e5e5e5' }}>
          <motion.button
            onClick={handleLogout}
            className="w-full bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-red-400 py-3 rounded-lg font-semibold transition-colors"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Logout
          </motion.button>
        </div>
      </motion.div>
    </div>
  )
}

