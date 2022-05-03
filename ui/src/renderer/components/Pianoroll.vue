<template>
  <div id="pianoroll">
    <el-table 
      :data="generateKeysData()"
      style="width: 100%"
      :show-header="false" 
      :border="true"
      :fit="true"
      empty-text=""
      :row-style="rowStyle"
      :cell-style="cellStyle"
      ref="tableRef"
      :scrollbar-always-on="false"
      @cell-click="noteAdd"
      max-height=720
    >
      <el-table-column prop="name" fixed="left" align="center" label="key" width="180" />
      <el-table-column prop="" label="grid" :width="reactObj.prWidth" :resizable="true" />
      <template #append><div class="placeholder" style="width: 100%; height: 200px;"></div></template>
      <!-- 占位，不然表格拉不到最下面 -->
    </el-table>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, reactive } from 'vue'
import { store } from '../store'
import { Upr } from '../typings/ui'
import * as utils from '../utils'
// import {default as lodash} from 'lodash'

const tableRef = ref()
const reactObj = reactive({
  prWidth: 1000,
})

onMounted(() => {
  tableRef.value.setScrollTop(1700)
})

watch(
  () => store.state.upr,
  (val, prev) => {
    utils.clearPianoRoll()
    drawPianoRoll(val, prev.updatedAt)
  },
  {deep: true}
)


const pianoKeys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];  // 先黑白键一样长，以后再改
const isBlackKey = [false, true, false, true, false, false, true, false, true, false, true, false]
const rowStyle = {height: '30px'}  // eltable限制最小只能33px

function cellStyle({ rowIndex, columnIndex }: { rowIndex: number; columnIndex: number }) {
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
      name: k + (group_id - 1).toString(),
      // 减一后音域才是C-1到G9，A4=400Hz
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

  let hue = (127 - velocity + 1) / 126
  // 将数字翻转后归一，使力度大的对应值小，方便后面对应深色
  hue = hue * 60 + 10
  return `hsl(${hue}, 100%, 60%)`
}

function newNote(x: number, velocity: number, length: number) {
  const note = document.createElement('span')
  note.style.backgroundColor = noteBgColor(velocity)
  note.style.position = 'absolute'
  note.style.left = `${x}px`
  note.style.top = '5%'
  note.style.width = `${length}px`
  note.style.height = '90%'
  note.style.borderRadius = '6%'
  note.style.display = 'inline-block'
  note.className = 'note'
  note.dataset.velocity = velocity.toString()
  note.addEventListener("click", (e) => {
    const n: any = e.target
    n.parentElement.removeChild(n)
  }, true)
  // 一个音符持续时无法再次产生，yui中也是如此
  return note
}

function drawPianoRoll(upr: Upr, lastUpdatedAt: number) {
  if(upr.updatedAt <= lastUpdatedAt) {
    return
  }
  // 因此在NavBar单独更改qpm时不会更改这里
  const { fps, pianoroll } = upr
  const noteReg = new RegExp('^t(\\d+)v(\\d+)c(\\d+)$')
  const lpc = utils.pxPerSecond / fps  // length per count, 指upr中一列对应多少像素
  const prRow = utils.getPrRowArray()
  let firstPitchIdx = -9
  let noteMaxOffset = reactObj.prWidth
  // for(const i in pianoroll) {
  for(let i=127; i>-1; i--) {
    if(!pianoroll[i]) {
      continue
    }
    if(firstPitchIdx < 0) {
      firstPitchIdx = 127 - i
    }
    for(const n of pianoroll[i].split(' ')) {
      const [_, t, v, c] = noteReg.exec(n)!.map(x => parseInt(x))  // key 1-3 分别是 t,v,c
      const note = newNote(t*lpc, v, c*lpc)
      noteMaxOffset = Math.max((t+c)*lpc, noteMaxOffset)
      prRow[127-i].appendChild(note)
    }
    // pianoroll下标从小到大表示其中音高从低到高， prRow相反
  }
  tableRef.value.setScrollTop((prRow[firstPitchIdx] as HTMLElement).parentElement?.offsetTop)
  // ? 加上后大概element为null就什么都不做
  reactObj.prWidth = noteMaxOffset + 100
}

function noteAdd(row: any, column: any, cell: any, event: any) {
  // console.log('column :>> ', column.no)
  // console.log('cell :>> ', cell)
  if(column.no != 1) {
    return
  }
  const length = (60 / store.state.upr.qpm) * utils.pxPerSecond * store.state.noteTimeRatio
  cell.firstElementChild.appendChild(newNote(event.offsetX, store.state.noteVelocity, length))
  // 把音符都放进 td > .cell 里面
  if(event.offsetX + 300 > reactObj.prWidth) {
    reactObj.prWidth = event.offsetX + 1000
  }
  // 第二列，卷帘部分才能加入音符;
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
    position: fixed;
    bottom: 10px;
    height: 8px;
  }
  .el-scrollbar__bar.is-vertical {
    right: 8px;
    width: 8px;
  }
}
// 全局域css，不用加deep也能作用于子组件
</style>
