import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  // The dev server proxies to FastAPI so the frontend calls the same paths in
  // development as it does when served from the API in production.
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
