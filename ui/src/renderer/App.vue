<script setup lang="ts">
import PR from './components/Pianoroll.vue';
import NavBar from './components/NavBar.vue';
import { ipcRenderer } from './electron';

ipcRenderer.send('message', 'Hello from App.vue!');
</script>

<template>
  <div id="App">  
  
    <!-- <span>Fucccckkkkk</span> -->
    <el-container>
      <el-header><NavBar/></el-header>
      <el-main><PR/></el-main>
    </el-container>
    <!-- <Hello/> -->
  </div>
</template>

<style scoped>
html {
  overflow-y: hidden;
}
::webkit-scrollbar { 
   display: none;
}
</style>
