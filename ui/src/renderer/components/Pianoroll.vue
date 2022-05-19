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
      :cell-class-name="cellClassName"
      ref="tableRef"
      :scrollbar-always-on="false"
      @cell-click="cellClicked"
      @cell-contextmenu="cellClickedRight"
      max-height=720
      v-loading="store.state.uprLoading"
      element-loading-text="加载UPR中，用时与乐曲长度正相关..."
    >
      <el-table-column prop="name" fixed="left" align="center" label="key" :width="keyRowWidth" />
      <el-table-column prop="" label="grid" :width="reactObj.prWidth" :resizable="true" />
      <template #append><div class="placeholder" style="width: 100%; height: 120px;"></div></template>
      <!-- 占位，不然表格拉不到最下面 -->
    </el-table>
    <div class="note-slider left" ></div>
    <div class="note-slider right" ></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, reactive } from 'vue'
import { store, uprPlayer } from '../store'
import { Upr } from '../typings/ui'
import * as utils from '../utils'
// import {default as lodash} from 'lodash'

const tableRef = ref()
const reactObj = reactive({
  prWidth: 1000,
})
const keyRowWidth = 180
const rowStyle = {height: '32px'}  // eltable限制最小只能32px
// const noteSliders: Element[] = []
let noteSlider: utils.noteLengthSlider
const timeline = new utils.draggableElement('div', 'timeline', {
  elemStyle: {
    height: `${parseFloat(rowStyle.height) * 128}px`,
    left: `${keyRowWidth}px`
  },
  hooks: {
    mouseup: () => {
      const offset = parseFloat(timeline.elem.style.left) - keyRowWidth
      uprPlayer.position = offset / utils.pxPerSecond
    }
  }
})

onMounted(() => {
  tableRef.value.setScrollTop(1700)
  noteSlider = new utils.noteLengthSlider(document.querySelectorAll('#pianoroll .note-slider'))
  setTimeout(() => {
    timeline.mount('#pianoroll .el-table .el-table__body')
  }, 0)
})

watch(
  () => store.state.upr,
  (val, prev) => {
    drawPianoRoll(val, prev.updatedAt)
  },
  {deep: true}
)
// 监听upr变动，并据此更新钢琴卷帘
watch(
  () => uprPlayer.position,
  (val, prev) => {
    const offset = val * utils.pxPerSecond
    const scrollGap = Math.max(document.body.scrollWidth - 300, 1)
    const cnt = Math.floor(offset / scrollGap)
    const cnt2 = Math.floor((prev * utils.pxPerSecond) / scrollGap)
    if(cnt !== cnt2) {
      tableRef.value.setScrollLeft(offset)
      // 当越过设定的间隔时移动滚动条
    }
    timeline.elem.style.left = `${offset + keyRowWidth}px`
  }
)
// 根据player time的变化调整timeline

const pianoKeys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];  // 先黑白键一样长，以后再改
const isBlackKey = [false, true, false, true, false, false, true, false, true, false, true, false]

function cellClassName({ rowIndex, columnIndex}: { rowIndex: number; columnIndex: number }) {
  if(columnIndex !== 0) {
    return
  }
  rowIndex = (127 - rowIndex) % 12
  return isBlackKey[rowIndex]? 'key black': 'key white'
  // generateKeysData里用了reverse，这里行号与实际音符排列相反
}

function pitchId2Name(pitchId: number) {
  const groupId = Math.floor(pitchId / 12)
  const key = pianoKeys[pitchId - groupId * 12]
  return key + (groupId - 1).toString()
  // 减一后音域才是C-1到G9，A4=400Hz
}

function generateKeysData() {
  const keyData = []
  for(let i=0; i<128; i++) {
    keyData.push({
      name: pitchId2Name(i),
    })
  }
  return keyData.reverse()
}

function newNote(x: number, velocity: number, length: number, pitch: string) {
  const note = document.createElement('span')
  note.style.backgroundColor = utils.noteBgColor(velocity)
  note.style.position = 'absolute'
  note.style.left = `${x}px`
  note.style.top = '5%'
  note.style.width = `${length}px`
  note.style.height = '90%'
  note.style.borderRadius = '4px'
  note.style.display = 'inline-block'
  note.className = 'note'
  note.dataset.pitch = pitch
  note.dataset.velocity = velocity.toString()
  // 一个音符持续时无法再次产生，yui中也是如此
  note.dataset.eventId = uprPlayer.add(note)
  return note
}

function drawPianoRoll(upr: Upr, lastUpdatedAt: number) {
  if(upr.updatedAt <= lastUpdatedAt) {
    return
  }
  // 因此在NavBar单独更改qpm时不会有影响
  noteSlider.hide()
  utils.clearPianoRoll()
  uprPlayer.cancel()
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
      const note = newNote(t*lpc, v, c*lpc, pitchId2Name(i))
      noteMaxOffset = Math.max((t+c)*lpc, noteMaxOffset)
      prRow[127-i].appendChild(note)
    }
    // pianoroll下标从小到大表示其中音高从低到高， prRow相反，即此时i为音高
  }
  tableRef.value.setScrollTop((prRow[firstPitchIdx] as HTMLElement).parentElement?.offsetTop)
  // ? 加上后大概element为null就什么都不做
  reactObj.prWidth = noteMaxOffset + 100
}

function checkNote(start: number, end: number, cell: Element): null|Element {
  // 检查现在要添加的音符条是否会与已存在的重叠
  let ns, ne, n: any  // note-start, note-end, note
  for(n of Array.from(cell.children)) {
    [ns, ne] = [n.style.left, n.style.width].map(x => parseFloat(x))
    ne += ns
    if (ns > start && ns < end) {
      return n
    }
    // if(ns <= start && ne >= start) {
    // 点在音符条上，cellClicked中已经做了判断不会出现这种情况
  }
  return null
}

function cellClicked(row: any, column: any, td: any, event: any) {
  if(noteSlider.selectedId !== null) {
    return
    // 此时点击事件发生在音符长度滑块上，不可新增音符
  }
  if(column.no !== 1) {
    uprPlayer.instrument.triggerAttackRelease(row.name, '4n')
    return
    // 左键点击左侧钢琴键盘
  }
  else if(event.target.className === 'note') {
    const note = event.target
    const duration = parseFloat(note.style.width) / utils.pxPerSecond
    const velocity = parseInt(note.dataset.velocity) / utils.maxVelocity
    noteSlider.showAt(td, note)
    uprPlayer.instrument.triggerAttackRelease(row.name, duration, undefined, velocity)
    return
    // 左键点击已添加的音符
  }
  noteSlider.hide()

  // 在左键点击位置新增音符
  const cell = td.firstElementChild, x = event.offsetX
  const length = (60 / store.state.upr.qpm) * utils.pxPerSecond * store.state.noteTimeRatio
  const olNote: any = checkNote(x, x+length, cell)
  if(olNote) {
    olNote.style.backgroundColor = '#00ff59'
    setTimeout(() => {
      olNote.style.backgroundColor = utils.noteBgColor(olNote.dataset.velocity)
    }, 3000)
    utils.showMsg('音符添加失败！因为亮绿色标示的音符会被重叠', 'warning')
    return
  }

  cell.appendChild(newNote(x, store.state.noteVelocity, length, row.name))
  // 把音符都放进 td > .td 里面
  if(event.offsetX + 300 > reactObj.prWidth) {
    reactObj.prWidth = event.offsetX + 1000
  }
  // 第二列，卷帘部分才能加入音符;
}

function cellClickedRight(row: any, column: any, td: any, event: any) {
  // 删去右键命中的音符
  noteSlider.hide()
  if(column.no !== 1 || event.target.className !== 'note') {
    return
    // 此时点击事件发生在钢琴键盘或不在音符条上
  }
  const note: HTMLElement = event.target
  note.parentElement?.removeChild(note)
  uprPlayer.remove(note)
}
// TODO 增加播放进度条，显示总时长
// TODO 换用虚拟化表格 https://element-plus.org/zh-CN/component/table-v2.html
// TODO 或许不该用表格，timeline只能相对整个表格定位，移动滚动条会导致滑到琴键上去，目前通过更改z-index救急
</script>


<style scoped lang="less">
@menuItemColor: #87e8aa;
@menuItemDarkColor: #29c560;

#pianoroll {
  :deep(.el-table__body-wrapper .el-table__row) {
    .key {
      &:hover {
        cursor: pointer;
        font-size: 14px;
      }
      font-size: 12px;
      border-radius: 0 4% 4% 0;
    }

    .black {
      color: #eee;
      border-right: 1px solid #000;
      border-bottom: 1px solid #000;
      box-shadow: -1px -1px 2px rgba(255,255,255,0.2) inset, -3px 0 2px 3px rgba(0,0,0,0.6) inset, 2px 0 3px rgba(0,0,0,0.5);
      background: linear-gradient(-10deg, #444 0%,#222 100%)
    }
    .black:active {
      box-shadow: -1px -1px 2px rgba(255,255,255,0.2) inset, -2px 0 2px 3px rgba(0,0,0,0.6) inset, 1px 0 2px rgba(0,0,0,0.5);
      background: linear-gradient(-45deg, #222 0%, #444 100%)
    }

    .white {
      color: #222;
      border-right: 1px solid #bbb;
      border-bottom: 1px solid #bbb;
      box-shadow: -1px 0 0 rgba(255,255,255,0.8) inset, -2px 0 2px #ccc inset, 2px 0 3px rgba(0,0,0,0.2);
      background: linear-gradient(to right, #eee 0%, #fff 100%)
    }

    .white:active {
      border-left: 1px solid #888;
      border-bottom: 1px solid #999;
      border-right: 1px solid #999;
      box-shadow: 0 2px 3px rgba(0,0,0,0.1) inset, 5px -2px 4px rgba(0,0,0,0.1) inset, 0 0 3px rgba(0,0,0,0.2);
      background: linear-gradient(to right, #fff 0%, #d2d2d2 100%)
    }
  }

  .note-slider {
    &:hover {
      cursor: col-resize;
    }
    position: absolute;
    display: none;
    top: 5%;
    width: 4px;
    height: 90%;
    background-color: @menuItemColor;
  }
  .note-slider.left {
    border-top-left-radius: 35%;
    border-bottom-left-radius: 35%;
  }
  .note-slider.right {
    border-top-right-radius: 35%;
    border-bottom-right-radius: 35%;
  }

  :deep(.el-table__body-wrapper  .el-table__body) {
    position: relative;
    // 为了timeline能够相对其绝对定位
  }

  :deep(.el-table__body-wrapper .el-table__body) {
    .timeline:hover {
      cursor: ew-resize;
    }
    .timeline {
      position: absolute;
      top: 0;
      width: 2px;
      z-index: 1;
      background-color: @menuItemColor;
      border: 1px solid @menuItemDarkColor;
    }
    .timeline.active {
      background-color: #f19d63;
      border: 1px solid #ff5050;
    }
  }

  font-family: 'Microsoft YaHei', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  // text-align: center;
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
:root {
  --el-color-primary: #29c560;
}
// 全局域css，不用加deep也能作用于子组件
</style>
