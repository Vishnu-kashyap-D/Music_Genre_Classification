// Utility functions for user-specific history storage

export const getUserHistoryKey = (userId, userEmail) => {
  const identifier = userId || userEmail
  if (!identifier) {
    return null
  }
  return `analysisHistory_${identifier}`
}

export const saveToUserHistory = (userId, filename, data, userProfile = {}) => {
  const key = getUserHistoryKey(userId, userProfile?.email)
  if (!key) {
    console.warn('Cannot save history: No user identifier')
    return null
  }

  if (!data || !data.global_confidence) {
    console.error('Invalid data provided to saveToUserHistory:', data)
    return null
  }

  const topGenre = Object.entries(data.global_confidence)
    .sort(([, a], [, b]) => b - a)[0][0]
  const analysisSnapshot = JSON.parse(JSON.stringify(data))
  
  const historyItem = {
    id: `hist_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`, // More unique ID
    filename,
    topGenre,
    timestamp: new Date().toISOString(),
    userDetails: {
      id: userProfile?.id || userId,
      name: userProfile?.name || userProfile?.email?.split('@')[0] || 'Unknown User',
      email: userProfile?.email || null,
    },
    data: {
      ...analysisSnapshot,
      filename: data.filename || filename,
    },
  }

  const history = JSON.parse(localStorage.getItem(key) || '[]')
  
  // Remove any duplicate entries for the same filename (optional)
  const filtered = history.filter(item => item.filename !== filename)
  
  filtered.unshift(historyItem)
  const updated = filtered.slice(0, 50) // Keep last 50 analyses per user
  
  try {
    localStorage.setItem(key, JSON.stringify(updated))
    console.log(`Saved history item for user ${historyItem.userDetails.id}:`, {
      key,
      itemId: historyItem.id,
      filename,
      totalItems: updated.length
    })
    
    // Dispatch custom event to notify other components
    window.dispatchEvent(new Event('historyUpdated'))
    
    return historyItem
  } catch (error) {
    console.error('Error saving to localStorage:', error)
    return null
  }
}

export const getUserHistory = (userId, userEmail) => {
  const key = getUserHistoryKey(userId, userEmail)
  if (!key) {
    console.warn('getUserHistory: No user identifier provided')
    return []
  }
  const stored = localStorage.getItem(key)
  
  if (!stored) {
    console.log(`No history found for user ${userId || userEmail} (key: ${key})`)
    return []
  }
  
  try {
    const history = JSON.parse(stored)
    // Filter out invalid entries
    const validHistory = history.filter(item => 
      item && 
      item.id && 
      item.filename && 
      item.data && 
      item.data.global_confidence
    )
    
    console.log(`Loaded ${validHistory.length} valid history items for user ${userId || userEmail}`)
    return validHistory
  } catch (error) {
    console.error('Error parsing history from localStorage:', error)
    return []
  }
}

export const deleteHistoryItem = (userId, itemId, userEmail) => {
  const key = getUserHistoryKey(userId, userEmail)
  if (!key) return []
  
  const history = getUserHistory(userId, userEmail)
  const updated = history.filter(item => item.id !== itemId)
  localStorage.setItem(key, JSON.stringify(updated))
  
  // Dispatch custom event
  window.dispatchEvent(new Event('historyUpdated'))
  
  return updated
}

export const clearUserHistory = (userId, userEmail) => {
  const key = getUserHistoryKey(userId, userEmail)
  if (!key) return
  
  localStorage.removeItem(key)
  
  // Dispatch custom event
  window.dispatchEvent(new Event('historyUpdated'))
}

