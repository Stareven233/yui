const Path = require('path');
const vuePlugin = require('@vitejs/plugin-vue')
// const AutoImport = require('unplugin-auto-import/vite')
// const Components = require('unplugin-vue-components/vite')
// const { ElementPlusResolver } = require('unplugin-vue-components/resolvers')
const { defineConfig } = require('vite');

/**
 * https://vitejs.dev/config
 */
const config = defineConfig({
    root: Path.join(__dirname, '..', 'src', 'renderer'),
    publicDir: 'public',
    server: {
        port: 8080,
    },
    open: false,
    build: {
        outDir: Path.join(__dirname, '..', 'build', 'renderer'),
        emptyOutDir: true,
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
            '@': Path.join(__dirname, '..', 'src', 'renderer')
        }
    }
});

module.exports = config;
