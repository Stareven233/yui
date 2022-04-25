<template>
  <div id="pianoroll">  
    <el-table :data="generateKeysData()" style="width: 100%" :show-header=false border=true empty-text="">
      <el-table-column :style="{background: 'black'}" prop="name" fixed="left" align="center" label="key" width="180" />
      <el-table-column prop="" label="grid" width="1600" />
    </el-table>
  </div>
</template>

<script setup lang="ts">
// const whiteKeys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
// const blackKeys = ['C#', 'D#', 'F#', 'G#', 'A#']
const pianoKeys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];  // 先黑白键一样长，以后再改
const group_num = 11  // 最后一组不完整，少了最后4个，共128音符

function generateKeysData() {
  const keyData = []
  let k: string = ''
  let group_id = 0
  for(let i=0; i<128; i++) {
    group_id = Math.floor(i / 12)
    k = pianoKeys[i - group_id * 12]
    keyData.push({
      name: k + group_id.toString(),
      color: k.endsWith("#")? "black": "white",
    })
  }
  return keyData
}

</script>


<style scoped lang="less">
#pianoroll {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 20px;

  .el-table__body.el-table__row.el-table_1_column_1 {
    .white.key {
      background-color: #ffffff;
      color: #000000;
    }
    .key {
      background-color: #000000;
      color: #ffffff;
    }
  }

  .el-table__body-wrapper tr td.el-table-fixed-column--left, .el-table__body-wrapper tr td.el-table-fixed-column--right, .el-table__body-wrapper tr th.el-table-fixed-column--left, .el-table__body-wrapper tr th.el-table-fixed-column--right, .el-table__footer-wrapper tr td.el-table-fixed-column--left, .el-table__footer-wrapper tr td.el-table-fixed-column--right, .el-table__footer-wrapper tr th.el-table-fixed-column--left, .el-table__footer-wrapper tr th.el-table-fixed-column--right, .el-table__header-wrapper tr td.el-table-fixed-column--left, .el-table__header-wrapper tr td.el-table-fixed-column--right, .el-table__header-wrapper tr th.el-table-fixed-column--left, .el-table__header-wrapper tr th.el-table-fixed-column--right {
    background: #000000 !important;
  }
}
</style>
