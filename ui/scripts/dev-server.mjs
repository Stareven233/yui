process.env.NODE_ENV = 'development'

import { createServer } from 'vite'
import { spawn } from 'child_process'
import chalk from 'chalk'
const { redBright, blueBright, white } = chalk
import { watch } from 'chokidar'
import Electron from 'electron'
import compileTs from './private/tsc.mjs'
import { cpSync } from 'fs'
import path from 'path'
import { config as viteConfig } from '../config/vite.mjs'
import {getDirname} from './private/utils.mjs'
// 2022-5-14 18:48:30 因为打包失败试着改成ES module，但Electron竟然还不支持，浪费半天时间

const __dirname = getDirname(import.meta.url)
let electronProcess = null
let rendererPort = 0


async function startRenderer() {
    // const config = require(path.join('..', 'config', 'vite.ts'))

    const server = await createServer({
        ...viteConfig,
        mode: 'development',
    })

    return server.listen()
}

async function startElectron() {
    if (electronProcess) { // single instance lock
        return
    }

    try {
        await compileTs(path.join(__dirname, '..', 'src', 'main'))
    } catch {
        console.log(redBright('Could not start Electron because of the above typescript error(s).'))
        return
    }

    const args = [
        path.join(__dirname, '..', 'build', 'main', 'main.js'),
        rendererPort,
    ]
    electronProcess = spawn(Electron, args)

    electronProcess.stdout.on('data', data => {
        console.log(blueBright(`[elecron] `) + white(data.toString()))
    })

    electronProcess.stderr.on('data', data => {
        console.log(blueBright(`[electron] `) + white(data.toString()))
    })
}

function restartElectron() {
    if (electronProcess) {
        electronProcess.kill()
        electronProcess = null
    }

    startElectron()
}


async function start() {
    console.log(`${blueBright('===============================')}`)
    console.log(`${blueBright('Starting Electron + Vite Dev Server...')}`)
    console.log(`${blueBright('===============================')}`)

    const devServer = await startRenderer()
    rendererPort = devServer.config.server.port

    cpSync(path.join(__dirname, '..', 'src', 'static'), path.join(__dirname, '..', 'build', 'static'), { recursive: true })

    startElectron()

    watch(path.join(__dirname, '..', 'src', 'main')).on('change', () => {
        restartElectron()
    })
}

start()
