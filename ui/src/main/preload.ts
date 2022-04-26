import {contextBridge, ipcRenderer, dialog} from 'electron';

contextBridge.exposeInMainWorld('electron', {
  ipcRenderer: ipcRenderer,
  dialog,
})
