import vuePlugin from '@vitejs/plugin-vue'
// import AutoImport from 'unplugin-auto-import/vite'
// import Components from 'unplugin-vue-components/vite'
// import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import { defineConfig } from 'vite'
import path from 'path'
// import { viteCommonjs } from '@originjs/vite-plugin-commonjs'
// import commonjs from '@rollup/plugin-commonjs'
import {getDirname} from '../scripts/private/utils.mjs'

const __dirname = getDirname(import.meta.url)

/**
 * https://vitejs.dev/config
 */
export const config = defineConfig({
    root: path.join(__dirname, '..', 'src', 'renderer'),
    publicDir: 'public',
    server: {
        port: 8080,
    },
    open: false,
    build: {
        outDir: path.join(__dirname, '..', 'build', 'renderer'),
        emptyOutDir: true,
        minify: false,
    },
    plugins: [
        vuePlugin(),
        // AutoImport({
        //     resolvers: [ElementPlusResolver()],
        // }),
        // Components({
        // resolvers: [ElementPlusResolver()],
        // }),
    ],
    resolve: {
        alias: {
            '@': path.join(__dirname, '..', 'src', 'renderer')
        }
    }
});
