<template>
  <div id="NavBar">
  <el-row class="menu-file">
    <el-col :span="1" class="menu-item">
      <el-icon :size="24" color="#3b3f45" ><document-add /></el-icon>
      <!-- 打开新的pianoroll，放在另一个tab页上，先不处理 -->
    </el-col>
    <el-col :span="1" class="menu-item" @click="openPianoroll">
      <el-icon :size="24" color="#3b3f45" ><folder /></el-icon>
      <!-- 从audio或者midi加载钢琴卷帘 -->
    </el-col>
    <el-col :span="1" class="menu-item" @click="saveMIDI">
      <el-icon :size="24" color="#3b3f45" ><management /></el-icon>
      <!-- 将当前编辑的钢琴卷帘存为midi文件 -->
    </el-col>
    <el-col :span="1" class="menu-item">
      <el-icon :size="24" color="#3b3f45" ><refresh /></el-icon>
      <!-- 清空当前tab的钢琴卷帘 -->
    </el-col>
  </el-row>

  <el-row class="menu-note">
    <el-col :span="1" class="note-duration" v-for="(url, index) in noteImgSrc()" :key="index" >
      <el-image style="width: 20px; height: auto" :src="url" fit="contain" @click.self="changeNote" />
      <!-- <span class="note-desc">音符</span> -->
    </el-col>
  </el-row>

  </div>
</template>

<script setup lang="ts">
// TODO 滚动到当前最大音高
// TODO addNote处添加对列宽的判定，点击位置距列宽太近就加长列宽
import { onMounted } from 'vue'
import { useStore } from 'vuex'
import { key, store } from '../store'
import { ipcRenderer } from '../electron';
// const store = useStore(key)
let lastSelectedCol = null
onMounted(() => {
  // console.log(tableRef.value);
  const quarterNote = document.querySelector('#NavBar > .el-row.menu-note > .el-col:nth-child(3)')
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
function changeNote(e) {
  const n: any = e.target
  const ratio = 4 / parseInt(noteDurationRegex.exec(n.src)[1])
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

function openPianoroll() {
  ipcRenderer.invoke('open-dialog', 'none').then(res => {
    // console.log('renderer res :>> ', res)
    if(res.canceled || !res.filePaths) {
      return
    }
    const filename: string = res.filePaths[0]
    console.log('filename :>> ', filename);
  }).catch(err => {
    console.error(err)
  })
}
function saveMIDI() {
  ipcRenderer.invoke('save-dialog', './3423/dfw/fwev/v').then(res => {
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
  .note-duration:hover, .menu-item:hover {
    cursor: pointer;
    box-shadow: var(--el-box-shadow-light);
  }
  .note-duration.selected {
    box-shadow: var(--el-box-shadow-light);
    background-color: #87e8aa;
  }
  

}
</style>
