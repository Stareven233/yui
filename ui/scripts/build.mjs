import { join } from 'path'
import chalk from 'chalk'
import { rmSync } from 'fs'
import compileTs from './private/tsc.mjs'
import Vite from 'vite'
import {config as viteConfig} from '../config/vite.mjs'
import {getDirname} from './private/utils.mjs'


const { blueBright, greenBright } = chalk
const __dirname = getDirname(import.meta.url)

function buildRenderer() {
    // const Vite = require('vite')
    // const viteConfig = require(join(__dirname, '..', 'config', 'vite.mjs'))

    return Vite.build({
        ...viteConfig,
        base: './',
        mode: 'production'
    })
}

function buildMain() {
    const mainPath = join(__dirname, '..', 'src', 'main')
    return compileTs(mainPath)
}

rmSync(join(__dirname, '..', 'build'), {
    recursive: true,
    force: true,
})

console.log(blueBright('Transpiling renderer & main...'))

Promise.allSettled([
    buildRenderer(),
    buildMain(),
]).then(() => {
    console.log(greenBright('Renderer & main successfully transpiled! (ready to be built with electron-builder)'))
})
