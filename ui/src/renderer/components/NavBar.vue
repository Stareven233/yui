<template>
  <div id="NavBar">
  <el-row class="menu-file">
    <el-col :span="1" class="menu-item">
      <el-icon :size="24" color="#3b3f45" ><document-add /></el-icon>
      <!-- 打开新的pianoroll，放在另一个tab页上，先不处理 -->
    </el-col>
    <el-col :span="1" class="menu-item" @click="openPianoroll">
      <el-icon :size="24" color="#3b3f45" ><folder /></el-icon>
      <!-- 从audio/midi/upr加载钢琴卷帘 -->
    </el-col>
    <el-col :span="1" class="menu-item">
      <el-icon :size="24" color="#3b3f45" @click="saveUPR" ><management /></el-icon>
      <!-- 将当前窗口的钢琴卷帘保存为自定义的upr文件 -->
    </el-col>
    <el-col :span="1" class="menu-item" @click="saveMIDI">
      <el-icon :size="24" color="#3b3f45" ><upload-filled /></el-icon>
      <!-- 将当前编辑的钢琴卷帘传给yui处理为midi文件并保存 -->
    </el-col>
  </el-row>

  <el-row class="menu-note">
    <el-col :span="1" class="note-duration menu-item" v-for="(url, index) in noteImgSrc()" :key="index" >
      <el-image style="width: 20px; height: auto" :src="url" fit="contain" @click.self="changeNote" />
      <!-- <span class="note-desc">音符</span> -->
    </el-col>

    <el-col :span="8" class="note-velocity" >
        <el-slider label="velocity" :min="1" :max="127" v-model="noteVelocity" show-input @change="changeVelocity" />
    </el-col>
  </el-row>

  </div>
</template>

<script setup lang="ts">
// TODO 滚动到当前最大音高
// TODO addNote处添加对列宽的判定，点击位置距列宽太近就加长列宽
import { onMounted, ref } from 'vue'
import { key, store } from '../store'
import { ipcRenderer } from '../electron';
// const store = useStore(key)

let lastSelectedCol: Element
const noteVelocity = ref(store.state.noteVelocity)  //通过ref赋初值

onMounted(() => {
  // console.log(tableRef.value);
  const quarterNote: Element = document.querySelector('#NavBar > .el-row.menu-note > .el-col:nth-child(3)')!
  quarterNote.className += ' selected'
  lastSelectedCol = quarterNote
  // 默认选中四分音符
})


const staticPath = "../assets"
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
  store.commit("noteVelocity", noteVelocity.value)
}

function openPianoroll() {
  ipcRenderer.invoke('open-dialog').then(res => {
    if(!res.success) {
      console.warn(res.message)
      return
    }
    const upr = JSON.parse(res.message)
    store.commit("updateUPR", upr)
  }).catch(err => {
    console.error(err)
  })
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

function saveUPR() {
  const filepath = 'F:/'
  ipcRenderer.invoke('save-upr', filepath).then(res => {
    if(res.canceled || !res.filePath) {
      return
    }
    const filename: string = res.filePath
    console.log('filename :>> ', filename);
  }).catch(err => {
    console.error(err)
  })
}

function saveMIDI() {
  const filepath = 'F:/'
  ipcRenderer.invoke('export-midi', filepath).then(res => {
    // console.log('renderer res :>> ', res)
    if(res.canceled || !res.filePath) {
      return
    }
    const filename: string = res.filePath
    console.log('filename :>> ', filename);
  }).catch(err => {
    console.error(err)
  })
}

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
    display: flex;
    align-items: flex-end;
    margin-left: 80px;

    :deep(.el-slider__runway) {
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
  }
  // /deep/: 用于 scoped 域修改子组件的深度作用选择器， >>> 的别名
  // 不过这俩都已经 deprecated
  
}
</style>
