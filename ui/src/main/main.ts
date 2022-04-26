const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const path = require('path')

let mainWindow;
function createWindow () {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 760,
    title: "ui of yui",
    icon: path.join(__dirname, "./static/kuro.ico"),
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
  
  mainWindow.webContents.openDevTools();
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
});

ipcMain.on('message', (event, message) => {
  console.log(message);
})

ipcMain.handle('open-dialog', (event, message) => {
  // console.log('main', event, message);
  
  return dialog.showOpenDialog(mainWindow, {
    title: "打开音频或MIDI文件",
    filters: [
      {
        name: 'audio or midi',
        extensions: ['wav', 'mp3', 'mid', 'midi'],
      },{
        name: 'All files',
        extensions: ['*'],
      }
    ],
    properties: ['openFile'],
  })
})


ipcMain.handle('save-dialog', (event, message) => {
  // console.log('main', event, message);
  
  return dialog.showSaveDialog(mainWindow, {
    title: "选择存储目录并写入文件名 或 替换已存在文件",
    defaultPath: 'C:/',
    buttonLabel: '保存',
    filters: [
      {
        name: 'midi',
        extensions: ['mid', 'midi'],
      },
    ],
  })
})
