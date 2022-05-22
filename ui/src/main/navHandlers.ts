import {dialog, mainWindow} from './main'
import * as spawnYui from './SpawnYui'
import fs from 'fs/promises'

export const returnObj = (success: boolean, message: string|Array<number>|Error) => {
  if(message instanceof Error) {
    message = `${message.name}: ${message.message}`
  }
  return {
    success,
    message,
  }
}

export const openUPR = async (event: object, message: string) => {
  // message不传就是 undefined
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
  }).then(async res => {
    if(res.canceled || !res.filePaths) {
      return returnObj(false, 'openUPR: canceled or empty path')
    }

    const filename: string = res.filePaths[0]
    const tmp: string[] = filename.split('.')
    const ext: string = tmp[tmp.length - 1]

    if(['mid', 'midi'].includes(ext)) {
      return spawnYui.infer('midi', filename)
    } else if(['wav', 'mp3'].includes(ext)) {
      return spawnYui.infer('audio', filename)
    } else {
      const fh: fs.FileHandle = await fs.open(filename, 'r')
      let readResult: fs.FileReadResult<Buffer>
      let uprJson: string = ''
      do {
        readResult = await fh.read()
        uprJson += readResult.buffer.slice(0, readResult.bytesRead).toString()
      } while(readResult.bytesRead > 0)
      await fh?.close()
      return returnObj(true, uprJson)
    }
  }).catch(err => {
    console.error(err)
    throw err
  })
}

export const saveUPR = async (event: object, message: string) => {
  // console.log('message :>> ', message)
  const res = await dialog.showSaveDialog(mainWindow, {
    title: "保存为ui pianoroll",
    defaultPath: 'C:/',
    buttonLabel: '保存',
    filters: [{
      name: 'ui pianoroll',
      extensions: ['upr'],
    },{
      name: 'All files',
      extensions: ['*'],
    }],
  })
  // console.log('saveUPR :>> ', res)
  if(res.canceled || !res.filePath) {
    return returnObj(false, 'saveUPR: canceled or empty path')
  }

  const fh: fs.FileHandle = await fs.open(res.filePath, 'w')
  fh.write(message)
  await fh?.close()
  return returnObj(true, 'save upr successfully')
}


export const exportMidi = async (event: object, message: string) => {
  const res = await dialog.showSaveDialog(mainWindow, {
    title: "导出MIDI",
    defaultPath: 'C:/',
    buttonLabel: '导出',
    filters: [{
      name: 'midi',
      extensions: ['midi', 'mid'],
    }],
  })
  if(res.canceled || !res.filePath) {
    return returnObj(false, 'exportMidi: canceled or empty path')
  }
  return spawnYui.exportMIDI(res.filePath, message)
}
