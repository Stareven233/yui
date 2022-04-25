import { createApp } from 'vue'
import App from './App.vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css';
import less from 'less';
const app = createApp(App);

app.use(ElementPlus, { size: 'small', zIndex: 3000 }).use(less);
app.mount('#app');
