import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5183,
    headers: { 'X-Frame-Options': 'ALLOWALL' },
  },
  resolve: {
    alias: {
      '@bssuite/ui': path.resolve(__dirname, '../../packages/ui/src'),
    },
  },
})
