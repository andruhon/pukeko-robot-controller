import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  define: {
    __AGUI_URL__: JSON.stringify(process.env.AGUI_URL || ''),
  },
  build: {
    outDir: fileURLToPath(new URL('dist/client', import.meta.url)),
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      // Local-source aliases for @galvanized-pukeko/vue-ui — useful when iterating
      // on the vue-ui source in `_refs/galvanized-pukeko-ai-ui/` without re-publishing.
      // Uncomment together with the matching entry in `tsconfig.app.json` `paths`.
      // '@galvanized-pukeko/vue-ui/style.css': fileURLToPath(
      //   new URL(
      //     '_refs/galvanized-pukeko-ai-ui/packages/galvanized-pukeko-vue-ui/src/assets/global.css',
      //     import.meta.url,
      //   ),
      // ),
      // '@galvanized-pukeko/vue-ui': fileURLToPath(
      //   new URL(
      //     '_refs/galvanized-pukeko-ai-ui/packages/galvanized-pukeko-vue-ui/src',
      //     import.meta.url,
      //   ),
      // ),
    },
  },
  server: {
    port: 5173,
  },
})
