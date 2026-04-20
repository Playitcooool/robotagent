<template>
  <a href="#main-content" class="skip-link">{{ lang.value === 'zh' ? '跳转到内容' : 'Skip to content' }}</a>

  <div v-if="auth.authLoading.value" class="auth-shell">
    <div class="auth-card skeleton-card">
      <div class="skeleton-title"></div>
      <div class="skeleton-subtitle"></div>
      <div class="skeleton-form">
        <div class="skeleton-input"></div>
        <div class="skeleton-input"></div>
      </div>
    </div>
  </div>

  <div v-else-if="!auth.authUser.value" class="auth-shell">
    <AuthView @authed="auth.onAuthed" />
  </div>

  <div v-else class="app-shell">
    <AppTopbar
      :authUser="auth.authUser.value"
      :isDark="isDark.value"
      :lang="lang.value"
      :fontSize="fontSize.value"
      :simStreamActive="workbench.simStreamActive.value"
      @toggle-lang="toggleLang"
      @toggle-theme="toggleTheme"
      @set-font-size="setFontSize"
      @logout="auth.onLogout"
    />

    <main id="main-content" class="app-main">
      <RouterView />
    </main>

    <ShortcutHelp
      :visible="showShortcutHelp"
      :lang="lang.value"
      @close="showShortcutHelp = false"
    />
  </div>
</template>

<script>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { RouterView } from 'vue-router'

import AppTopbar from './components/AppTopbar.vue'
import AuthView from './components/AuthView.vue'
import ShortcutHelp from './components/ShortcutHelp.vue'
import { useAuth } from './composables/useAuth.js'
import { usePreferences } from './composables/usePreferences.js'
import { useWorkbenchStore } from './composables/useWorkbenchStore.js'

export default {
  name: 'App',
  components: { AppTopbar, AuthView, RouterView, ShortcutHelp },
  setup () {
    const auth = useAuth()
    const { lang, fontSize, isDark, toggleLang, toggleTheme, setFontSize } = usePreferences()
    const workbench = useWorkbenchStore()
    const showShortcutHelp = ref(false)

    function handleKeydown(e) {
      if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const tag = document.activeElement?.tagName
        if (tag !== 'INPUT' && tag !== 'TEXTAREA') showShortcutHelp.value = true
      }

      if (e.key === 'Escape' && showShortcutHelp.value) {
        showShortcutHelp.value = false
      }
    }

    onMounted(() => {
      auth.checkAuth()
      document.addEventListener('keydown', handleKeydown)
    })

    onBeforeUnmount(() => {
      workbench.teardown()
      document.removeEventListener('keydown', handleKeydown)
    })

    return {
      auth,
      workbench,
      lang,
      fontSize,
      isDark,
      toggleLang,
      toggleTheme,
      setFontSize,
      showShortcutHelp
    }
  }
}
</script>

<style scoped>
.app-main {
  min-height: 0;
}
</style>
