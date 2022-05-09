<script setup lang="ts">
import PR from './components/Pianoroll.vue';
import NavBar from './components/NavBar.vue';
import { ipcRenderer } from './electron';

ipcRenderer.send('message', 'Hello from App.vue!');
</script>

<template>
  <!-- <span>Fucccckkkkk</span> -->
  <el-container id="main">
    <el-header><NavBar/></el-header>
    <el-main><PR/></el-main>
  </el-container>
  <!-- <Hello/> -->
</template>

<style scoped lang="less">
html {
  overflow-y: hidden;
}
::webkit-scrollbar { 
  display: none;
}
</style>
