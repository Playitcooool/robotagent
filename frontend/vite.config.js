import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      output: {
        manualChunks (id) {
          if (!id.includes('node_modules')) return
          if (id.includes('/vue/') || id.includes('vue-router')) return 'vue-vendor'
          if (id.includes('highlight.js')) return 'highlight-vendor'
          if (
            id.includes('markdown-it-highlightjs') ||
            id.includes('markdown-it-multimd-table')
          ) return 'markdown-plugins'
          if (
            id.includes('markdown-it') ||
            id.includes('linkify-it') ||
            id.includes('mdurl') ||
            id.includes('uc.micro') ||
            id.includes('/entities/')
          ) return 'markdown-core'
          if (id.includes('katex')) return 'katex-vendor'
        }
      }
    }
  },
  server: {
    // allow access from LAN if needed; toggle with --host when running
    host: 'localhost',
    // Forward API requests to the backend during development to avoid CORS and 404 from Vite
    proxy: {
      // proxy any request starting with /api to the backend running on :8000
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        // keep the /api prefix so backend routes like /api/chat/send match
        rewrite: (path) => path,
      },
    },
  }
})
