import { computed, ref, watch } from 'vue'

import {
  applyTheme,
  DEFAULT_FONT_SIZE,
  FONT_SIZE_KEY,
  LANG_KEY,
  readPreferences,
  THEME_KEY,
  toggleLanguage,
  toggleTheme
} from '../lib/preferences.js'

const initial = readPreferences()

const lang = ref(initial.lang)
const theme = ref(initial.theme)
const fontSize = ref(initial.fontSize || DEFAULT_FONT_SIZE)

let initialized = false

function persistPreferences() {
  if (typeof localStorage === 'undefined') return
  localStorage.setItem(LANG_KEY, lang.value)
  localStorage.setItem(THEME_KEY, theme.value)
  localStorage.setItem(FONT_SIZE_KEY, String(fontSize.value))
}

function applyVisualPreferences() {
  applyTheme(theme.value)
  if (typeof document !== 'undefined') {
    document.documentElement.style.setProperty('--font-size-base', `${fontSize.value}px`)
  }
}

export function usePreferences() {
  if (!initialized) {
    initialized = true

    watch([lang, theme, fontSize], () => {
      persistPreferences()
      applyVisualPreferences()
    }, { immediate: true })
  }

  const isDark = computed(() => theme.value !== 'light')

  function flipTheme() {
    theme.value = toggleTheme(theme.value)
  }

  function flipLanguage() {
    lang.value = toggleLanguage(lang.value)
  }

  function setFontSize(size) {
    const numeric = Number(size)
    fontSize.value = Number.isFinite(numeric) ? numeric : DEFAULT_FONT_SIZE
  }

  return {
    lang,
    theme,
    fontSize,
    isDark,
    toggleTheme: flipTheme,
    toggleLang: flipLanguage,
    setFontSize
  }
}
