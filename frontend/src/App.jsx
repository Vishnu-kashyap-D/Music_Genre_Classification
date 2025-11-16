import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { ThemeProvider } from './context/ThemeContext'
import { AuthProvider } from './context/AuthContext'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Analysis from './pages/Analysis'
import History from './pages/History'
import Login from './components/Login'
import Signup from './components/Signup'
import Profile from './components/Profile'
import ProtectedRoute from './components/ProtectedRoute'
import { useTheme } from './context/ThemeContext'

function AppContent() {
  const { isDark } = useTheme()

  return (
    <div 
      className="min-h-screen transition-colors duration-300"
      style={{
        backgroundColor: isDark ? '#0a0a0a' : '#ffffff',
        color: isDark ? '#ffffff' : '#1a1a1a',
      }}
    >
      <Navbar />
      <main className="p-6 lg:p-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route 
            path="/history" 
            element={
              <ProtectedRoute>
                <History />
              </ProtectedRoute>
            } 
          />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route 
            path="/profile" 
            element={
              <ProtectedRoute>
                <Profile />
              </ProtectedRoute>
            } 
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      
      <Toaster 
        position="top-right"
        limit={3}
        toastOptions={{
          style: {
            background: isDark ? '#242424' : '#ffffff',
            color: isDark ? '#ffffff' : '#1a1a1a',
            border: `1px solid ${isDark ? '#333333' : '#e5e5e5'}`,
          },
        }}
      />
    </div>
  )
}

function App() {
  return (
    <Router>
      <ThemeProvider>
        <AuthProvider>
          <AppContent />
        </AuthProvider>
      </ThemeProvider>
    </Router>
  )
}

export default App
