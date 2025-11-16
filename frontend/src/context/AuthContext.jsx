import { createContext, useContext, useState, useEffect } from 'react'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('user')
    return saved ? JSON.parse(saved) : null
  })

  const [isLoading, setIsLoading] = useState(false)

  const createAvatar = (seed) => {
    // Fallback to a simple data URI if external service fails
    try {
      return `https://ui-avatars.com/api/?name=${encodeURIComponent(seed)}&background=random`
    } catch {
      // Return a simple colored circle as fallback
      const colors = ['6366f1', '8b5cf6', 'ec4899', 'f59e0b', '10b981']
      const color = colors[seed.length % colors.length]
      return `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='50' fill='%23${color}'/><text x='50' y='65' font-size='40' text-anchor='middle' fill='white' font-family='Arial'>${seed.charAt(0).toUpperCase()}</text></svg>`
    }
  }

  useEffect(() => {
    if (user) {
      localStorage.setItem('user', JSON.stringify(user))
    } else {
      localStorage.removeItem('user')
    }
  }, [user])

  const login = async (email, password) => {
    setIsLoading(true)
    try {
      // Simulate API call - replace with actual backend
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // Check if user exists in localStorage (for persistence)
      const existingUsers = JSON.parse(localStorage.getItem('users') || '{}')
      let userData
      
      if (existingUsers[email]) {
        // User exists, use their data
        userData = existingUsers[email]
      } else {
        // New user, create account
        userData = {
          id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          email,
          name: email.split('@')[0],
          avatar: createAvatar(email.split('@')[0]),
          createdAt: new Date().toISOString(),
          isGuest: false,
        }
        // Save to users registry
        existingUsers[email] = userData
        localStorage.setItem('users', JSON.stringify(existingUsers))
      }
      
      setUser(userData)
      setIsLoading(false)
      return { success: true }
    } catch (error) {
      setIsLoading(false)
      return { success: false, error: error.message }
    }
  }

  const signup = async (name, email, password) => {
    setIsLoading(true)
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // Check if user already exists
      const existingUsers = JSON.parse(localStorage.getItem('users') || '{}')
      if (existingUsers[email]) {
        setIsLoading(false)
        return { success: false, error: 'User with this email already exists' }
      }
      
      const userData = {
        id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        email,
        name,
        avatar: createAvatar(name),
        createdAt: new Date().toISOString(),
        isGuest: false,
      }
      
      // Save to users registry
      existingUsers[email] = userData
      localStorage.setItem('users', JSON.stringify(existingUsers))
      
      setUser(userData)
      setIsLoading(false)
      return { success: true }
    } catch (error) {
      setIsLoading(false)
      return { success: false, error: error.message }
    }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem('user')
  }

  const updateProfile = (updates) => {
    setUser(prev => {
      const updated = { ...prev, ...updates }
      // Also update in users registry
      if (prev?.email && !prev?.isGuest) {
        const existingUsers = JSON.parse(localStorage.getItem('users') || '{}')
        existingUsers[prev.email] = updated
        localStorage.setItem('users', JSON.stringify(existingUsers))
      }
      return updated
    })
  }

  const loginAsGuest = async () => {
    setIsLoading(true)
    try {
      await new Promise(resolve => setTimeout(resolve, 300))
      const guestUser = {
        id: `guest_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
        email: null,
        name: 'Guest',
        avatar: createAvatar('Guest User'),
        createdAt: new Date().toISOString(),
        isGuest: true,
      }
      setUser(guestUser)
      setIsLoading(false)
      return { success: true }
    } catch (error) {
      setIsLoading(false)
      return { success: false, error: error.message || 'Unable to start guest session' }
    }
  }

  return (
    <AuthContext.Provider value={{
      user,
      isLoading,
      login,
      signup,
      logout,
      updateProfile,
      loginAsGuest,
      isAuthenticated: !!user,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

