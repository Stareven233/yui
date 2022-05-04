const ChildProcess = require('child_process')
import {returnObj} from './navHandlers'


const yui_infer_path = 'D:/A日常/大学/毕业设计/code/yui/yui/inference.py';
const prReg = new RegExp('##Piano(\\{.+?\\})Roll##')
// flush无效，只能是期待整个pr一起传过来然后用正则提取
const errReg = new RegExp('error:', 'i')

const errHandle = (data: string, reject?: any) => {
  if(errReg.test(data.toString())) {
    reject(new Error(data))
  }
  // 忽略警告
}

export const infer = (type: string, filename: string) => {
  return new Promise((resolve, reject) => {
    const yui = ChildProcess.spawn('py', ['-3.9', yui_infer_path, `--${type}`, filename])
    let upr: string

    yui.stdout.on('data', (data: string) => {
      const regRes = prReg.exec(data.toString())
      if(regRes) {
        upr = regRes[1]
        return
        // 拿到了钢琴卷帘数组，后面的都丢弃
      }
    })

    yui.on('close', (code: number) => {
      resolve(returnObj(code===0, upr))
    })

    yui.stderr.on('data', (d: string) => errHandle(d, reject))
  })
}

export const exportMIDI = (path: string, uprJSON: string) => {
  return new Promise((resolve, reject) => {
    const yui = ChildProcess.spawn('py', ['-3.9', yui_infer_path, `--upr`, path])
    let out: string = ''
    yui.stdin.write(uprJSON)
    yui.stdin.end()

    yui.stdout.on('data', (data: string) => {
      out = data.toString()
    })

    yui.on('close', (code: number) => {
      console.log('code :>> ', code, out)
      resolve(returnObj(code===0, out))
    })

    yui.stderr.on('data', (d: string) => errHandle(d, reject))
  })
}
