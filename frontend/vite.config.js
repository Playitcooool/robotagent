import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
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
