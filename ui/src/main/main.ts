const { app, BrowserWindow, ipcMain, dialog, Menu, session } = require('electron')
const path = require('path')

let mainWindow;
const vueDevToolsPath = 'C:/Users/Noetu/AppData/Local/Google/Chrome/User Data/Default/Extensions/nhdogjmejiglipccpnnnanhbledajbpd/6.1.4_0'

async function createWindow () {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 760,
    title: "Ui",
    icon: path.join(__dirname, "./static/logo.ico"),
    // __dirname: \yui\ui\build\main
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      devTools: true,
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  if (process.env.NODE_ENV === 'development') {
    const rendererPort = process.argv[2];
    mainWindow.loadURL(`http://localhost:${rendererPort}`);
  }
  else {
    mainWindow.loadFile(path.join(app.getAppPath(), 'renderer', 'index.html'));
  }
  
  await session.defaultSession.loadExtension(vueDevToolsPath)
  await mainWindow.webContents.openDevTools()
}

app.whenReady().then(async () => {
  createWindow();

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });

  // Menu.setApplicationMenu(null);
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
});

ipcMain.on('message', (event, message) => {
  console.log(message);
})


import {openUPR, saveUPR, exportMidi} from './navHandlers'

ipcMain.handle('open-upr', openUPR)
ipcMain.handle('save-upr', saveUPR)
ipcMain.handle('export-midi', exportMidi)

export {
  dialog,
  mainWindow,
}
