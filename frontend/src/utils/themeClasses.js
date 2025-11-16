// Utility functions for theme-aware classes
export const getThemeClasses = (isDark) => {
  return {
    bg: isDark ? 'bg-dark-bg' : 'bg-light-bg',
    surface: isDark ? 'bg-dark-surface' : 'bg-light-surface',
    card: isDark ? 'bg-dark-card' : 'bg-light-card',
    border: isDark ? 'border-dark-border' : 'border-light-border',
    text: isDark ? 'text-dark-text' : 'text-light-text',
    textMuted: isDark ? 'text-dark-text-muted' : 'text-light-text-muted',
  }
}

// Helper to create conditional classes
export const cn = (...classes) => {
  return classes.filter(Boolean).join(' ')
}

