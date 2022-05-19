import { app, BrowserWindow, ipcMain, dialog, session, Menu } from 'electron'
import path from 'path'
import { readdir } from 'fs/promises'

let mainWindow: BrowserWindow
const vueDevToolsDir = 'C:/Users/Noetu/AppData/Local/Google/Chrome/User Data/Default/Extensions/nhdogjmejiglipccpnnnanhbledajbpd/'
const isDevMode = process.env.NODE_ENV === 'development'

async function loadDevTool() {
  try {
    const files = await readdir(vueDevToolsDir)
    if(files.length === 0) {
      console.error(`${vueDevToolsDir} has no devTool folder`)
      return
    }
    await session.defaultSession.loadExtension(path.join(vueDevToolsDir, files[0]))
    mainWindow.webContents.openDevTools()
  } catch (err) {
    console.error(err)
  }
}

async function createWindow () {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 760,
    title: "ui v0.1.0",
    // title: "ui --臭√宝专享封闭内测初回至尊纪念版001",
    // title: "ui --最最最好兄弟竖大拇指小蓝专享封闭内测初回至尊纪念版000",
    icon: path.join(__dirname, '..', 'static/logo.ico'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      devTools: true,
      nodeIntegration: false,
      contextIsolation: true,
    },
    // __dirname 被定位到了 ui/src/build
  })

  if (isDevMode) {
    const rendererPort = process.argv[2]
    mainWindow.loadURL(`http://localhost:${rendererPort}`)
    await loadDevTool()
    return
  }
  mainWindow.loadFile(path.join(app.getAppPath(), 'renderer', 'index.html'))
}

app.whenReady().then(async () => {
  createWindow()

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })

  Menu.setApplicationMenu(null)
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

ipcMain.on('message', (event, message) => {
  console.log(message)
})


import {openUPR, saveUPR, exportMidi} from './navHandlers'

ipcMain.handle('open-upr', openUPR)
ipcMain.handle('save-upr', saveUPR)
ipcMain.handle('export-midi', exportMidi)

export {
  dialog,
  mainWindow,
}
