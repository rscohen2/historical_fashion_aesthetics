import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Relative base so the build works under the GitHub Pages project path
  // (/historical_fashion_aesthetics/) as well as at a root.
  base: './',
})
