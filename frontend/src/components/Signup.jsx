import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { UserPlus, Mail, Lock, User, Music } from 'lucide-react'
import { motion } from 'framer-motion'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import { useTheme } from '../context/ThemeContext'

export default function Signup() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const { signup, isLoading } = useAuth()
  const navigate = useNavigate()
  const { isDark } = useTheme()

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!name || !email || !password) {
      toast.error('Please fill in all fields')
      return
    }

    if (password.length < 6) {
      toast.error('Password must be at least 6 characters')
      return
    }

    const result = await signup(name, email, password)
    
    if (result.success) {
      toast.success('Account created successfully!')
      navigate('/')
    } else {
      toast.error(result.error || 'Signup failed')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{
      background: isDark ? '#0a0a0a' : '#ffffff'
    }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`w-full max-w-md ${isDark ? 'bg-dark-card' : 'bg-light-card'} rounded-2xl shadow-2xl p-8 border ${isDark ? 'border-dark-border' : 'border-light-border'}`}
      >
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 mb-4"
          >
            <Music className="w-8 h-8 text-white" />
          </motion.div>
          <h2 className="text-3xl font-bold mb-2" style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}>
            Create Account
          </h2>
          <p className={isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}>
            Join us to start analyzing music
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className={`block text-sm font-semibold mb-2 ${isDark ? 'text-dark-text' : 'text-light-text'}`}>
              Name
            </label>
            <div className="relative">
              <User className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}`} />
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className={`w-full pl-10 pr-4 py-3 rounded-lg border ${isDark ? 'bg-dark-surface border-dark-border text-dark-text' : 'bg-light-surface border-light-border text-light-text'} focus:outline-none focus:ring-2 focus:ring-purple-500`}
                placeholder="Your Name"
              />
            </div>
          </div>

          <div>
            <label className={`block text-sm font-semibold mb-2 ${isDark ? 'text-dark-text' : 'text-light-text'}`}>
              Email
            </label>
            <div className="relative">
              <Mail className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}`} />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`w-full pl-10 pr-4 py-3 rounded-lg border ${isDark ? 'bg-dark-surface border-dark-border text-dark-text' : 'bg-light-surface border-light-border text-light-text'} focus:outline-none focus:ring-2 focus:ring-purple-500`}
                placeholder="your@email.com"
              />
            </div>
          </div>

          <div>
            <label className={`block text-sm font-semibold mb-2 ${isDark ? 'text-dark-text' : 'text-light-text'}`}>
              Password
            </label>
            <div className="relative">
              <Lock className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 ${isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}`} />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`w-full pl-10 pr-4 py-3 rounded-lg border ${isDark ? 'bg-dark-surface border-dark-border text-dark-text' : 'bg-light-surface border-light-border text-light-text'} focus:outline-none focus:ring-2 focus:ring-purple-500`}
                placeholder="••••••••"
              />
            </div>
          </div>

          <motion.button
            type="submit"
            disabled={isLoading}
            className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-lg font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Creating account...
              </>
            ) : (
              <>
                <UserPlus className="w-5 h-5" />
                Sign Up
              </>
            )}
          </motion.button>
        </form>

        <div className="mt-6 text-center">
          <p className={isDark ? 'text-dark-text-muted' : 'text-light-text-muted'}>
            Already have an account?{' '}
            <Link to="/login" className="text-purple-400 hover:text-purple-300 font-semibold">
              Sign in
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  )
}

