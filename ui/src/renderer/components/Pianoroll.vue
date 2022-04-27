<template>
  <div id="pianoroll">  
    <el-table 
      :data="generateKeysData()" 
      style="width: 100%" 
      :show-header=false 
      :border=true 
      empty-text=""
      :row-style="rowStyle"
      :cell-style="cellStyle"
      ref="tableRef"
      :scrollbar-always-on=false
      @cell-click="noteAdd"
      max-height=720
    >
      <el-table-column prop="name" fixed="left" align="center" label="key" width="180" />
      <el-table-column prop="" label="grid" width="2200" />
    </el-table>

  </div>
</template>

<script setup lang="ts">
// TODO addNote处添加对列宽的判定，点击位置距列宽太近就加长列宽
import { ref, onMounted } from 'vue'
import { store } from '../store'

const tableRef = ref(null)
// console.log(Object.keys(tableRef));
let quarterNoteTime = 80

onMounted(() => {
  // console.log(tableRef.value);
  tableRef.value.setScrollTop(2000)
  // tableRef.value.setScrollLeft(800)

  const scrollView = document.querySelector("#pianoroll .el-table__body-wrapper .el-scrollbar__view")
  const placeholder = document.createElement('div')
  placeholder.className = 'placeholder'
  placeholder.style.height = '150px'
  placeholder.style.width = '100%'
  scrollView.appendChild(placeholder)
  // 占位，不然表格拉不到最下面
})

const pianoKeys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];  // 先黑白键一样长，以后再改
const isBlackKey = [false, true, false, true, false, false, true, false, true, false, true, false]

const rowStyle = {height: '30px'}  // eltable限制最小只能33px
function cellStyle({ rowIndex, columnIndex }) {
  // console.log(row, column)
  const style = {
    'border': '1px solid #616161', 
    'border-radius': '0 6px 6px 0'
  }

  if( columnIndex == 0 && isBlackKey[rowIndex%12]) {
    Object.assign(style, {'background': 'black', 'color': 'white'})
  }
  else if( columnIndex == 0) {
    Object.assign(style, {'background': 'white', 'color': 'black'})
  }
  else {
    return 
  }
  return style
}

function generateKeysData() {
  const keyData = []
  let k: string = ''
  let group_id = 0
  for(let i=0; i<128; i++) {
    group_id = Math.floor(i / 12)
    k = pianoKeys[i - group_id * 12]
    keyData.push({
      name: k + group_id.toString(),
    })
  }
  return keyData.reverse()
}

function noteBgColor(velocity: number): string {
  // note.style.backgroundColor = '#f09d63'
  // note.style.backgroundColor = 'rgba(180, 75, 0, 0.4)'

  // const colorNum = Math.round(((store.state.noteVelocity - 1) / 126) * 400)
  // const hue = Math.floor(colorNum / 11) + 20
  // const light = (colorNum % 11) * 3 + 50
  // 色相 20-60 乘 亮度 50-60 共400种变化

  let hue = (127 - store.state.noteVelocity + 1) / 126
  // 将数字翻转后归一，使力度大的对应值小，方便后面对应深色
  hue = hue * 60 + 10
  return `hsl(${hue}, 100%, 60%)`
}

function noteAdd(row, column, cell, event) {
  const note = document.createElement('span')
  note.style.backgroundColor = noteBgColor(store.state.noteVelocity)
  note.style.position = 'absolute'
  note.style.left = `${event.offsetX}px`
  note.style.top = '5%'
  note.style.width = `${quarterNoteTime * store.state.noteTimeRatio}px`
  note.style.height = '90%'
  note.style.borderRadius = '6%'
  note.style.display = 'inline-block'
  note.addEventListener("click", (e) => {
    const n: any = e.target
    n.parentElement.removeChild(n)
  }, true)
  // 一个音符持续时无法再次产生，yui中也是如此
  cell.appendChild(note)
}

</script>


<style scoped lang="less">
#pianoroll {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  // text-align: center;
  color: #133a62;
}
</style>

<style lang="less">
#pianoroll {
  .el-scrollbar__bar.is-horizontal {
    position: absolute;
    bottom: 85px;
    height: 8px;
  }
  .el-scrollbar__bar.is-vertical {
    right: 8px;
    width: 8px;
  }
}
// 全局域css，不用加deep也能作用于子组件
</style>
