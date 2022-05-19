import vuePlugin from '@vitejs/plugin-vue'
// import AutoImport from 'unplugin-auto-import/vite'
// import Components from 'unplugin-vue-components/vite'
// import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import { defineConfig } from 'vite'
import path from 'path'
import {getDirname} from '../scripts/private/utils.mjs'

const __dirname = getDirname(import.meta.url)
const rollupOptions = {
    output: {
        manualChunks(idx) {
            if (!idx.includes('node_modules')) {
                return 'normal'
            }
            const arr = idx.split('node_modules/')[1].split('/')
            console.log('arr[0] :>> ', arr[0])
            switch(arr[0]) {
            case 'vue':
            case 'vuex':
                return 'vue'
            case '@element-plus':
                return '@element-plus'
            case 'element-plus':
                return 'element-plus'
            // 但这样分开会造成NavBar.vue里对element-plus的样式修改失效
            case 'tone':
                return 'tone'
            default:
                return 'vendor'
            }
        },
    }
}

/**
 * https://vitejs.dev/config
 */
export const config = defineConfig({
    root: path.join(__dirname, '..', 'src', 'renderer'),
    publicDir: 'assets',
    server: {
        port: 8080,
    },
    open: false,
    build: {
        outDir: path.join(__dirname, '..', 'build', 'renderer'),
        emptyOutDir: true,
        minify: true,
        // rollupOptions,
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
})
