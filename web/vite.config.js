import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs on :8000 (uvicorn api.main:app); the dev server proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { rollupOptions: { input: { landing: 'index.html', console: 'console.html' } } }, // two pages: / and /console.html
})
