const normalizeBaseUrl = (url) => {
  if (!url || typeof url !== 'string') {
    return ''
  }
  return url.endsWith('/') ? url.slice(0, -1) : url
}

const DEFAULT_BASE_URL = 'http://127.0.0.1:5000'

export const API_ENDPOINTS = {
  base: normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL) || DEFAULT_BASE_URL,
}

API_ENDPOINTS.predict = `${API_ENDPOINTS.base}/predict`
API_ENDPOINTS.health = `${API_ENDPOINTS.base}/health`
