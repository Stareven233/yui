<template>
  <div id="NavBar">
  <el-row class="menu-file">

    <el-col :span="1" class="menu-item" @click="clearUPR">
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

    <el-tooltip effect="dark" placement="bottom-start" content="音符播放控制器，分别为开始、暂停与复位">
    <el-col :span="3" class="menu-item player-button">
      <el-icon :size="24" color="#3b3f45" @click="() => {uprPlayer.play()}" ><video-play /></el-icon>
      <!-- 将当前编辑的钢琴卷帘传给yui处理为midi文件并保存 -->
      <el-icon :size="24" color="#3b3f45" @click="() => {uprPlayer.pause()}" ><video-pause /></el-icon>
      <!-- 将当前编辑的钢琴卷帘传给yui处理为midi文件并保存 -->
      <el-icon :size="24" color="#3b3f45" @click="() => {uprPlayer.stop()}" ><refresh-left /></el-icon>
      <!-- 将当前编辑的钢琴卷帘传给yui处理为midi文件并保存 -->
    </el-col>
    </el-tooltip>

    <div class="note-input player-timer" >
      <el-tooltip effect="dark" placement="bottom-start" content="播放器时间（秒）" >
        <label for="playerTime">time:</label>
      </el-tooltip>
      <el-input-number
        v-model="uprPlayer.position"
        :min="0"
        :step="0.005"
        :precision="3"
        :step-strictly="true"
        label="playerTime"
        controls-position="right"
      />
    </div>

    <div class="player-inst" >
      <!-- <el-select 
        v-model="reactObj.playerInst"
        @change="changePlayerInst"
        :filterable="true"
      >
        <el-option
          v-for="(inst, index) in uprPlayer.instrumentList"
          :key="index"
          :label="inst"
          :value="inst"
        />
      </el-select> -->
      <el-select-v2
        v-model="reactObj.playerInst"
        :options="instSelectOptions"
        placeholder="select instrument"
        :filterable="true"
        @change="changePlayerInst"
        size="small"
      />
    </div>

  </el-row>

  <el-row class="menu-note">
    <el-col :span="1" class="note-duration menu-item" v-for="(url, index) in noteImgSrc" :key="index" >
      <el-image style="width: 20px; height: auto" :src="url" fit="contain" @click.self="changeNote" />
      <!-- <span class="note-desc">音符</span> -->
    </el-col>
    
    <div class="note-input note-ks" >
      <el-tooltip
        effect="dark"
        placement="top-start"
      >
        <template #content> 调号(Key Signature)</template>
        <el-select 
          v-model="reactObj.keySignature"
          @change="changeKeySignature"
          :filterable="true"
          class="key-signature"
        >
          <el-option-group
            v-for="group in generateKeyHGroups()"
            :key="group.label"
            :label="group.label"
          >
            <el-option
              v-for="item in group.options"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-option-group>
        </el-select>
      </el-tooltip>
    </div>

    <div class="note-input note-ts" >
      <el-tooltip
        effect="dark"
        placement="bottom-start"
      >
        <template #content> 拍号(Time Signature)<br/>导出MIDI时与qpm一起用于计算bpm </template>
        <div class="time-signature">
          <el-input-number
            v-model="reactObj.timeSignature[0]"
            :min="1"
            :step="1"
            :step-strictly="true"
            label="timeSignature-numerator"
            :controls="false"
            @change="changeTimeSignature"
          />
          <el-input-number
            v-model="reactObj.timeSignature[1]"
            :min="1"
            :step="1"
            :step-strictly="true"
            label="timeSignature-denominator"
            :controls="false"
            @change="changeTimeSignature"
          />
        </div>
      </el-tooltip>
    </div>

    <div class="note-input note-qpm" >
      <el-tooltip
        effect="dark"
        placement="top-start"
      >
        <template #content> 每分钟四分音符数(Quarter-note Per Minute)<br/>影响添加的音符长度，仅对手动添加的音符生效 </template>
        <label for="qpm">
          <el-image style="width: 16px; height: auto" :src="noteImgSrc[2]" fit="contain" />
          =
        </label>
      </el-tooltip>
      <el-input-number
        v-model="reactObj.qpm"
        :min="1"
        :max="1000"
        :step="0.1"
        :step-strictly="true"
        class="qpm"
        label="qpm"
        :controls="false"
        @change="changeQPM"
      />
    </div>

    <el-col :span="8" class="note-velocity" >
    <el-tooltip effect="dark" placement="top" content="力度(Velocity)">
      <el-slider label="velocity" :min="1" :max="utils.maxVelocity" v-model="store.state.noteVelocity" show-input />
    </el-tooltip>
    </el-col>

  </el-row>

  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive } from 'vue'
import { store, uprPlayer } from '../store'
import { ipcRenderer } from '../electron'
import * as utils from '../utils'
import { KeySignatureOption } from '../typings/ui'
// const store = useStore(key)

const reactObj = reactive({
  qpm: store.state.upr.qpm,
  timeSignature: store.state.upr.timeSignature,
  keySignature: store.state.upr.keySignature,
  playerTime: uprPlayer.position,
  playerInst: uprPlayer.currentInst,
})
// qpm=120: 一分钟120个四分音符，一个四分音符占0.5秒，对应80px
// let synth = utils.pianoSynth
// 后续可以提供自定义synth功能
let lastSelectedCol: Element
const instSelectOptions = function() {
  const ret: any = []
  uprPlayer.instrumentList.forEach((x: string) => ret.push({
    value: x,
    label: uprPlayer.instrumentChiMap[x] || x,
  }))
  return ret
}()

onMounted(() => {
  // console.log(tableRef.value);
  const quarterNote: Element = document.querySelector('#NavBar > .el-row.menu-note > .el-col:nth-child(3)')!
  quarterNote.classList.add('selected')
  lastSelectedCol = quarterNote
  // 默认选中四分音符
})


const noteImgSrc = function() {
  const arr = []
  for(let i=0; i<7; i++) {
    arr.push(`./note${1<<i}.png`)
    // 根据vite的设置，这里./就是 ../assets/
  }
  // 按顺序从0 -> 6, 时值从全音符到六十四分音符
  return arr
}()

const noteDurationRegex = /(\d+)\.png$/
function changeNote(e: any) {
  const n: any = e.target
  const ratio = 4 / parseInt(noteDurationRegex.exec(n.src)![1])
  // 以四分音符时值比例为1
  if(lastSelectedCol) {
    lastSelectedCol.classList.remove('selected')
  }
  const elcol = n.parentElement.parentElement
  elcol.classList.add('selected')
  lastSelectedCol = elcol
  store.commit("noteTimeRatio", ratio)
  // console.log('n.src :>> ', n.src, store.state.noteTimeRatio)
}

function changeQPM(e: number) {
  reactObj.qpm = Number(e.toFixed(2).slice(0,-1))
  store.commit("changeQPM", reactObj.qpm)
}

function changeTimeSignature(e: number) {
  store.commit("changeTimeSignature", reactObj.timeSignature)
}

// function changePlayerTime(val: number, prev: number) {
  // uprPlayer.position = val
  // setTimeout(() => {
  //   uprPlayer.position = uprPlayer.position
  // }, 0)
// }

function clearUPR() {
  utils.clearPianoRoll()
  uprPlayer.cancel()
  uprPlayer.position = 0
  uprPlayer.refreshLenth()
}

function openUPR() {
  store.commit("uprLoading", true)
  ipcRenderer.invoke('open-upr').then(res => {
    if(!res.success) {
      store.commit("uprLoading", false)
      console.warn(res.message)
      utils.showMsg(res.message, 'warning')
      return
    }
    const upr = JSON.parse(res.message)
    // console.log('upr :>> ', upr)
    upr.updatedAt = new Date().getTime()
    reactObj.qpm = upr.qpm
    reactObj.timeSignature = upr.timeSignature
    reactObj.keySignature = upr.keySignature
    uprPlayer.position = 0
    store.commit("updateUPR", upr)
    store.commit("uprLoading", false)
  }).catch(err => {
    console.error(err)
    utils.showMsg(err.toString(), 'error')
    store.commit("uprLoading", false)
  })
}

function getUpr() {
  const pianoroll: string[] = []
  for(const cell of utils.getPrRowArray()) {
    const line: string[] = []
    for(const note of (cell.children as any)) {
      const tc = [note.style.left, note.style.width]
      const [t, c] = tc.map(x => {
        return Math.round((parseFloat(x) / utils.pxPerSecond) * store.state.upr.fps)
        // 映射到钢琴卷帘矩阵里的列数
      })
      const v = note.dataset.velocity
      line.push(`t${t}v${v}c${c}`)
    }
    pianoroll.push(line.join(' '))
  }

  return {
    qpm: reactObj.qpm,
    pianoroll: pianoroll.reverse(),
    // 使pianoroll下标从小到大表示其中音高从低到高
    fps: store.state.upr.fps,
    timeSignature: reactObj.timeSignature,
    keySignature: reactObj.keySignature,
  }
}

function saveUPR() {
  const uprJSON = JSON.stringify(getUpr())
  ipcRenderer.invoke('save-upr', uprJSON).then(res => {
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
  utils.showMsg('exportMIDI: this may take some time, depending on the performance of the machine', 'info')
  const uprJSON = JSON.stringify(getUpr())
  ipcRenderer.invoke('export-midi', uprJSON).then(res => {
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

const keySignatureNames = [
  'C','D♭','D','E♭','E','F','G♭','G','A♭','A','B♭','B',   // major
  'c','c♯','d','e♭','e','f','f♯','g','g♯','a','b♭','b',  // minor
]
function generateKeyHGroups() {
  const groups = [
    {label: 'Major', options: [] as KeySignatureOption[]},
    {label: 'Minor', options: [] as KeySignatureOption[]},
  ]

  for(let i=0; i<24; i++) {
    const mode = Math.floor(i / 12)
    groups[mode].options.push({
      value: i,
      label: keySignatureNames[i],
    })
  }
  return groups
}

function changeKeySignature(val: number) {
  store.commit("changeKeySignature", val)
}

function changePlayerInst(val: string) {
  uprPlayer.setInstrument(val)
}

</script>


<style scoped lang="less">
@menuItemColor: #87e8aa;
@menuItemDarkColor: #29c560;

:root {
  --el-color-primary: @menuItemDarkColor;
}

#NavBar {
  position: relative;
  top: 0%;
  padding-bottom: 6px;

  .el-row:last-child {
    margin-bottom: 0;
  }
  .el-col {
    border-radius: 4px;
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
  }

  .note-qpm {
    label.el-tooltip__trigger {
      display: flex;
      align-items: flex-end;
      margin-right: 0;
    }

    .qpm {
      width: 32px;
    }
  }

  // /deep/: 用于 scoped 域修改子组件的深度作用选择器， >>> 的别名
  // 不过这俩都已经 deprecated
  .note-input {
    :deep(.el-input .el-input__wrapper) {
      padding: 0;
      box-shadow: none;
      font-size: 16px;
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
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      justify-content: space-between;
    }

    .key-signature {
      :deep(.el-input__wrapper) {
        .el-input__suffix {
          display: none;
        }
        box-shadow: none;
        font-size: 18px;
        width: 24px;
      }
    }
    
    display: flex;
    align-items: flex-end;
    line-height: 26px;
    color: #333333;
    margin-left: 20px;
  }

}

#NavBar .player-button {
  .el-icon {
    &:hover {
      font-size: 26px !important;
    }
    &:active {
      color: @menuItemDarkColor;
    }
    cursor: pointer;
    height: 28px;
  }

  &:active {
    background: none;
  }
  display: flex;
  text-align: center;
  align-items:center;
  justify-content: space-around;
  margin-left: 155px;
}

#NavBar .player-timer {
  :deep(.el-input-number) {
    width: 86px;
  }
  :deep(.el-input-number .el-input__inner) {
    text-align: left;
  }
}

#NavBar .player-inst {
  :deep(.el-select-v2 .el-select-v2__placeholder) {
    font-size: 15px;
  }
  width: 180px;
  margin-left: 30px;
}

</style>
