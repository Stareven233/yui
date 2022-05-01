import {dialog, mainWindow} from './main'
import {spawnYui} from './SpawnYui'

export const returnObj = (success: boolean, message: string|Array<number>|Error) => {
  if(message instanceof Error) {
    message = `${message.name}: ${message.message}`
  }
  return {
    success,
    message,
  }
}

// TODO 增加成功或失败消息框
// TODO 换成 async, await
export const openPianoroll = (event, message) => {
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
  }).then(res => {
    if(res.canceled || !res.filePaths) {
      return returnObj(false, 'openPianoroll: canceled or empty path')
    }

    const filename: string = res.filePaths[0]
    const tmp: string[] = filename.split('.')
    const ext: string = tmp[tmp.length - 1]

    if(['mid', 'midi'].includes(ext)) {
      return spawnYui('midi', filename)
    } else if(['wav', 'mp3'].includes(ext)) {
      return spawnYui('audio', filename)
    } else {
      // TODO readUpr()
      // return returnObj(true, pianoroll)
    }
  }).catch(err => {
    console.error(err)
    return returnObj(false, err)
  })
}

export const exportMidi = (event, message) => {
  return dialog.showSaveDialog(mainWindow, {
    title: "导出MIDI",
    defaultPath: message,
    buttonLabel: '导出',
    filters: [{
      name: 'midi',
      extensions: ['mid', 'midi'],
    }],
  })
}

export const saveUPR = (event, message) => {
  return dialog.showSaveDialog(mainWindow, {
    title: "保存为ui pianoroll",
    defaultPath: message,
    buttonLabel: '保存',
    filters: [{
      name: 'ui pianoroll',
      extensions: ['upr'],
    }],
  })
}

