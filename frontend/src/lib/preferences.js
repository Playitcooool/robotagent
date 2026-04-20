export const LANG_KEY = 'robotagent_lang'
export const THEME_KEY = 'robotagent_theme'
export const FONT_SIZE_KEY = 'robotagent_font_size'
export const DEFAULT_FONT_SIZE = 15

export function createStorageAdapter(seed = {}) {
  const store = new Map(Object.entries(seed))

  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null
    },
    setItem(key, value) {
      store.set(key, String(value))
    },
    removeItem(key) {
      store.delete(key)
    }
  }
}

export function readPreferences(storage = globalThis?.localStorage) {
  const lang = storage?.getItem?.(LANG_KEY) || 'zh'
  const theme = storage?.getItem?.(THEME_KEY) === 'light' ? 'light' : 'dark'
  const rawFontSize = Number.parseInt(storage?.getItem?.(FONT_SIZE_KEY) || `${DEFAULT_FONT_SIZE}`, 10)

  return {
    lang,
    theme,
    fontSize: Number.isFinite(rawFontSize) ? rawFontSize : DEFAULT_FONT_SIZE
  }
}

export function toggleTheme(theme = 'dark') {
  return theme === 'light' ? 'dark' : 'light'
}

export function toggleLanguage(lang = 'zh') {
  return lang === 'zh' ? 'en' : 'zh'
}

export function applyTheme(theme, target = document?.documentElement) {
  if (!target?.classList) return
  if (theme === 'light') target.classList.add('light')
  else target.classList.remove('light')
}
