import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    // allow access from LAN if needed; toggle with --host when running
    host: 'localhost'
  }
})
