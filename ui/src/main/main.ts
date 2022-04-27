const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron')
const path = require('path')

let mainWindow;
function createWindow () {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 760,
    title: "Ui of Yui",
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

  const {Menu} = require('electron');  // 引入 Menu 模块
  Menu.setApplicationMenu(null);
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
    title: "打开音频/MIDI/ui钢琴卷帘文件",
    filters: [
      {
        name: 'ui pianoroll or audio or midi',
        extensions: ['upr', 'wav', 'mp3', 'mid', 'midi'],
      },{
        name: 'All files',
        extensions: ['*'],
      }
    ],
    properties: ['openFile'],
  })
})

ipcMain.handle('export-midi', (event, message) => {
  return dialog.showSaveDialog(mainWindow, {
    title: "导出MIDI",
    defaultPath: message,
    buttonLabel: '导出',
    filters: [{
      name: 'midi',
      extensions: ['mid', 'midi'],
    }],
  })
})

ipcMain.handle('save-pianoroll', (event, message) => {
  return dialog.showSaveDialog(mainWindow, {
    title: "保存为ui pianoroll",
    defaultPath: message,
    buttonLabel: '保存',
    filters: [{
      name: 'ui pianoroll',
      extensions: ['upr'],
    }],
  })
})
