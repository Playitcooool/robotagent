import { ref } from 'vue'

const AUTH_TOKEN_KEY = 'robotagent_auth_token'

const authToken = ref(localStorage.getItem(AUTH_TOKEN_KEY) || '')
const authUser = ref(null)
const authLoading = ref(true)

export function useAuth () {
  function authHeaders (extra = {}) {
    const headers = { ...extra }
    if (authToken.value) headers.Authorization = `Bearer ${authToken.value}`
    return headers
  }

  async function apiFetch (url, options = {}) {
    const merged = {
      ...options,
      headers: authHeaders(options.headers || {})
    }
    const res = await fetch(url, merged)
    if (res.status === 401) handleAuthExpired()
    return res
  }

  function handleAuthExpired () {
    authToken.value = ''
    authUser.value = null
    localStorage.removeItem(AUTH_TOKEN_KEY)
  }

  async function checkAuth () {
    authLoading.value = true
    try {
      if (!authToken.value) {
        authUser.value = null
        return
      }
      const res = await apiFetch('/api/auth/me')
      if (!res.ok) {
        authUser.value = null
        return
      }
      const data = await res.json().catch(() => ({}))
      authUser.value = data?.user || null
    } finally {
      authLoading.value = false
    }
  }

  function onAuthed (payload) {
    authToken.value = payload.token || ''
    authUser.value = payload.user || null
    localStorage.setItem(AUTH_TOKEN_KEY, authToken.value)
  }

  async function onLogout () {
    try {
      await apiFetch('/api/auth/logout', { method: 'POST' })
    } catch (_) {}
    handleAuthExpired()
  }

  return {
    authToken,
    authUser,
    authLoading,
    authHeaders,
    apiFetch,
    handleAuthExpired,
    checkAuth,
    onAuthed,
    onLogout
  }
}
