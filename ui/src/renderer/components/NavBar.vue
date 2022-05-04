<template>
  <div id="NavBar">
  <el-row class="menu-file">

    <el-col :span="1" class="menu-item" @click="utils.clearPianoRoll">
    <el-tooltip effect="dark" placement="bottom-start" content="清除UPR">
      <el-icon :size="24" color="#3b3f45" ><document-add /></el-icon>
      <!-- 打开新的pianoroll，放在另一个tab页上，先不处理 -->
    </el-tooltip>
    </el-col>

    <el-col :span="1" class="menu-item" @click="openUPR">
    <el-tooltip effect="dark" placement="bottom-start" content="打开UPR">
      <el-icon :size="24" color="#3b3f45" ><folder /></el-icon>
      <!-- 从audio/midi/upr加载钢琴卷帘 -->
    </el-tooltip>
    </el-col>

    <el-col :span="1" class="menu-item">
    <el-tooltip effect="dark" placement="bottom-start" content="保存UPR">
      <el-icon :size="24" color="#3b3f45" @click="saveUPR" ><management /></el-icon>
      <!-- 将当前窗口的钢琴卷帘保存为自定义的upr文件 -->
    </el-tooltip>
    </el-col>

    <el-col :span="1" class="menu-item" @click="exportMIDI">
    <el-tooltip effect="dark" placement="bottom-start" content="导出MIDI">
      <el-icon :size="24" color="#3b3f45" ><upload-filled /></el-icon>
      <!-- 将当前编辑的钢琴卷帘传给yui处理为midi文件并保存 -->
    </el-tooltip>
    </el-col>

  </el-row>

  <el-row class="menu-note">
    <el-col :span="1" class="note-duration menu-item" v-for="(url, index) in noteImgSrc()" :key="index" >
      <el-image style="width: 20px; height: auto" :src="url" fit="contain" @click.self="changeNote" />
      <!-- <span class="note-desc">音符</span> -->
    </el-col>

    <el-col :span="8" class="note-velocity" >
    <el-tooltip effect="dark" placement="top" content="力度调节">
      <el-slider label="velocity" :min="1" :max="127" v-model="reactObj.noteVelocity" show-input @change="changeVelocity" />
    </el-tooltip>
    </el-col>

    <el-col :span="2" class="note-input" >
      <el-tooltip
        effect="dark"
        placement="top-start"
      >
        <template #content> 每分钟四分音符数<br/>仅影响手动添加的新音符 </template>
        <label>qpm:</label>
      </el-tooltip>
      <el-input-number
        v-model="reactObj.qpm"
        :min="1"
        :max="1000"
        :step="0.1"
        :step-strictly="true"
        label="qpm"
        :controls="false"
        @change="changeQPM"
      />
    </el-col>

    <el-col :span="1" class="note-input" >
      <el-tooltip
        effect="dark"
        placement="top-start"
      >
        <template #content> 拍号(Time Signature)<br/>导出MIDI时与qpm一起用于计算bpm </template>
        <label>TS:</label>
      </el-tooltip>

      <div class="time-signature">
        <el-input-number
          v-model="reactObj.timeSignature[0]"
          :min="1"
          :step="1"
          :step-strictly="true"
          label="timeSignature-numerator"
          :controls="false"
          @change="changeTimeSignature(0)"
        />
        <div class="divider"></div>
        <el-input-number
          v-model="reactObj.timeSignature[1]"
          :min="1"
          :step="1"
          :step-strictly="true"
          label="timeSignature-denominator"
          :controls="false"
          @change="changeTimeSignature(1)"
        />
      </div>
    </el-col>
  
  </el-row>

  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive } from 'vue'
import { store } from '../store'
import { ipcRenderer } from '../electron'
import * as utils from '../utils'
// const store = useStore(key)

const staticPath = "../assets"
let lastSelectedCol: Element
const reactObj = reactive({
  noteVelocity: store.state.noteVelocity,
  qpm: store.state.upr.qpm,
  timeSignature: store.state.upr.timeSignature,
})
// qpm=120: 一分钟120个四分音符，一个四分音符占0.5秒，对应80px

onMounted(() => {
  // console.log(tableRef.value);
  const quarterNote: Element = document.querySelector('#NavBar > .el-row.menu-note > .el-col:nth-child(3)')!
  quarterNote.className += ' selected'
  lastSelectedCol = quarterNote
  // 默认选中四分音符
})


function noteImgSrc() {
  const noteImgSrc = []
  for(let i=0; i<7; i++) {
    noteImgSrc.push(`${staticPath}/note${1<<i}.png`)
  }
  // 按顺序从0 -> 6, 时值从全音符到六十四分音符
  return noteImgSrc
}

const noteDurationRegex = /(\d+)\.png$/
function changeNote(e: any) {
  const n: any = e.target
  const ratio = 4 / parseInt(noteDurationRegex.exec(n.src)![1])
  // 以四分音符时值比例为1
  if(lastSelectedCol) {
    lastSelectedCol.className = lastSelectedCol.className.replace(' selected', '')
  }
  const elcol = n.parentElement.parentElement
  elcol.className += ' selected'
  lastSelectedCol = elcol
  store.commit("noteTimeRatio", ratio)
  // console.log('n.src :>> ', n.src, store.state.noteTimeRatio)
}

function changeVelocity(e: number) {
  // console.log('e :>> ', typeof e, e);
  // noteVelocity.value === e
  store.commit("noteVelocity", e)
}

function changeQPM(e: number) {
  reactObj.qpm = Number(e.toFixed(2).slice(0,-1))
  store.commit("changeQPM", reactObj.qpm)
}

function changeTimeSignature(idx: number) {
  return (e: number) => {
    reactObj.timeSignature[idx] = e
    store.commit("changeTimeSignature", reactObj.timeSignature)
  }
}

function openUPR() {
  ipcRenderer.invoke('open-upr').then(res => {
    if(!res.success) {
      console.warn(res.message)
      utils.showMsg(res.message, 'warning')
      return
    }
    const upr = JSON.parse(res.message)
    // console.log('upr :>> ', upr)
    upr.updatedAt = new Date().getTime()
    reactObj.qpm = upr.qpm
    reactObj.timeSignature = upr.timeSignature
    store.commit("updateUPR", upr)
  }).catch(err => {
    console.error(err)
    utils.showMsg(err.toString(), 'error')
  })
}

function getUprJSON() {
  const pianoroll: string[] = []
  for(const cell of utils.getPrRowArray()) {
    const line: string[] = []
    for(const note of (cell.children as any)) {
      const tc = [note.style.left, note.style.width]
      const [t, c] = tc.map(x => {
        let ret = parseFloat(x.slice(0, -2))
        ret = Math.round((ret / utils.pxPerSecond) * store.state.upr.fps)
        return ret  // 映射到钢琴卷帘矩阵里的列数
      })
      const v = note.dataset.velocity
      line.push(`t${t}v${v}c${c}`)
    }
    pianoroll.push(line.join(' '))
  }

  return JSON.stringify({
    qpm: reactObj.qpm,
    pianoroll: pianoroll.reverse(),
    // 使pianoroll下标从小到大表示其中音高从低到高
    fps: store.state.upr.fps,
    timeSignature: reactObj.timeSignature,
  })
}

function saveUPR() {
  ipcRenderer.invoke('save-upr', getUprJSON()).then(res => {
    if(!res.success) {
      console.warn(res.message)
      utils.showMsg(res.message, 'warning')
      return
    }
    utils.showMsg('saved successfully', 'success')
  }).catch(err => {
    console.error(err)
  })
}

function exportMIDI() {
  ipcRenderer.invoke('export-midi', getUprJSON()).then(res => {
    if(!res.success) {
      console.warn(res.message)
      utils.showMsg(res.message, 'warning')
      return
    }
    utils.showMsg('exported successfully', 'success')
  }).catch(err => {
    console.error(err)
  })
}
// TODO 导出的tempo不正确
// TODO 加loading遮罩
</script>


<style scoped lang="less">
@menuItemColor: #87e8aa;
@menuItemDarkColor: #29c560;

#NavBar {
  position: relative;
  top: 0%;
  // height: 100px;

  .el-row:last-child {
    margin-bottom: 0;
  }
  .el-col {
    border-radius: 4px;

    :deep(.el-input .el-input__wrapper) {
      --el-input-focus-border-color: @menuItemDarkColor;
    }
  }
  
  .menu-file {
    height: 35px;
    display: flex;
    text-align: center;
    align-items:center;
    justify-content: flex-start;
  }

  .note-duration {
    text-align: center;
  }
  .menu-item:hover {
    cursor: pointer;
    box-shadow: var(--el-box-shadow-light);
  }
  .menu-item:active {
    cursor: pointer;
    box-shadow: var(--el-box-shadow-light);
    background-color: @menuItemDarkColor;
  }
  .note-duration.selected {
    box-shadow: var(--el-box-shadow-light);
    background-color: @menuItemColor;
  }

  .note-velocity {
    :deep(.el-slider__runway) {
      margin-right: 10px;
      .el-slider__bar, .el-slider__button {
        background-color: @menuItemColor;
      }
      .el-slider__button {
        border: 1px solid @menuItemDarkColor;
      }
    }

    :deep(.el-slider__input) {
      width: 90px;
    }

    display: flex;
    align-items: flex-end;
    margin-left: 30px;
    margin-right: 30px;
  }
  // /deep/: 用于 scoped 域修改子组件的深度作用选择器， >>> 的别名
  // 不过这俩都已经 deprecated
  .note-input {
    :deep(.el-input-number) {
      .el-input__wrapper {
        padding: 0;
        box-shadow: none;
        font-size: 16px;
      }
      width: 35px;
    }

    .el-tooltip__trigger {
      margin-right: 5px;
      font-size: 15px;
    }

    .time-signature {
      :deep(.el-input-number) {
        .el-input__wrapper {
          font-size: 18px;
        }
        height: 18px;
        width: 20px;
      }
      :deep(.el-input-number:last-child) {
        border-top: 1px solid #333333;
      }
      .divider {
        background-color: #333333;
        height: 1px;
        width: 100%;
      }
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      justify-content: space-between;
    }
    
    display: flex;
    align-items: flex-end;
    line-height: 26px;
    color: #333333;
  }

}
</style>
