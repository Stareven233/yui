<template>
  <div id="pianoroll" @click="updateUPR">
    <!-- <el-button>prLastUpdatedAt: {{prLastUpdatedAt}} </el-button> -->
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
// TODO 滚动到当前最大音高
import { ref, onMounted, watch, reactive } from 'vue'
import { store } from '../store'

const tableRef = ref()
// console.log(Object.keys(tableRef));
const pxPerSecond = 160  // 音符每秒对应的音符条长度(px)

const reactObj = reactive({fps: store.state.upr.fps, vel: store.state.noteVelocity})
let prLastUpdatedAt: number = 0


onMounted(() => {
  // console.log(tableRef.value);
  tableRef.value.setScrollTop(1700)
  // tableRef.value.setScrollLeft(800)

  const scrollView: Element = document.querySelector("#pianoroll .el-table__body-wrapper .el-scrollbar__view")!
  const placeholder = document.createElement('div')
  placeholder.className = 'placeholder'
  placeholder.style.height = '150px'
  placeholder.style.width = '100%'
  scrollView.appendChild(placeholder)
  // 占位，不然表格拉不到最下面
})

// watch(
//   reactObj,
//   (val, prev) => {
//     pianoroll = val.upr.pianoroll
//     console.log('upr :>> ', val)
//     console.log('prevUpr :>> ', prev)
//   },
//   {deep: true}
// )

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
  note.addEventListener("click", (e) => {
    const n: any = e.target
    n.parentElement.removeChild(n)
  }, true)
  // 一个音符持续时无法再次产生，yui中也是如此
  return note
}

function updateUPR() {
  if(store.state.upr.updatedAt <= prLastUpdatedAt) {
    return
  }
  const { fps, pianoroll } = store.state.upr
  const noteReg = new RegExp('^t(\\d+)v(\\d+)c(\\d+)$')
  const lpc = pxPerSecond / fps  // length per count, 指upr中一列对应多少像素
  const prRow = document.querySelectorAll('#pianoroll .el-table .el-table__row > .el-table_1_column_2')
  let firstPitchIdx = -9
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
      prRow[127-i].appendChild(note)
    }
  }
  tableRef.value.setScrollTop((prRow[firstPitchIdx] as HTMLElement).offsetTop)
  prLastUpdatedAt = store.state.upr.updatedAt
  // TODO 之后应检测state变动自动调用
}

function noteAdd(row: any, column: any, cell: any, event: any) {
  // console.log('column :>> ', column.no)
  // console.log('cell :>> ', cell)
  if(column.no != 1) {
    return
  }
  const length = (60 / store.state.upr.qpm) * pxPerSecond * store.state.noteTimeRatio
  cell.appendChild(newNote(event.offsetX, store.state.noteVelocity, length))
  // 第二列，卷帘部分才能加入音符
}

// 59: "t428v96c20"
// 60: "t85v96c21"
// 62: "t171v96c20 t514v96c20"
// 63: ""
// 64: "t342v96c21"
// 67: "t107v96c20 t257v96c81 t449v96c21"
// 71: "t128v96c10 t471v96c10"
// 79: "t160v96c10 t224v96c21 t503v96c10 t567v96c21"
// 80: ""
// 81: "t171v96c20 t203v96c20 t514v96c20 t546v96c20"

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
