import { createApp } from 'vue'
import App from './App.vue'
import './assets/styles.css'
// KaTeX styles for math rendering
import 'katex/dist/katex.min.css'
// Highlight.js code block styles
import 'highlight.js/styles/github-dark.css'

createApp(App).mount('#app')

// Register service worker for PWA / offline support
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // SW registration failed — app works without it
    })
  })
}
