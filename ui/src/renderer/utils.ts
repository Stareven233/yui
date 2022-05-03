import { ElMessage } from 'element-plus'


export const pxPerSecond = 160  // 音符每秒对应的音符条长度(px)

export const showMsg = (msg: string, type: any) => {
  ElMessage({
    showClose: true,
    message: msg,
    type: type,
  })
}

export const getPrRowArray = () => {
  const prRow = document.querySelectorAll('#pianoroll .el-table .el-table__row > .el-table_1_column_2 > .cell')
  return Array.from(prRow)
  // HTMLCollection, NodeList 竟然不能迭代 却可以转换为Array
}

export const clearPianoRoll = () => {
  for(const cell of getPrRowArray()) {
    cell.innerHTML = ''
  }
}
