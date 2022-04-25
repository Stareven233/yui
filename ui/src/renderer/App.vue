<script setup lang="ts">
import PR from './components/Pianoroll.vue';
import Hello from './components/Hello.vue';
import { ipcRenderer } from './electron';

ipcRenderer.send('message', 'Hello from App.vue!');
</script>

<template>
  <div id="App">  
  
    <!-- <span>Fucccckkkkk</span> -->
    <PR></PR>
    <!-- <Hello/> -->
  </div>
</template>

<style scoped>
::webkit-scrollbar { 
   display: none;
}
</style>
