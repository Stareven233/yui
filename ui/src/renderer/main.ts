import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import * as ElIcons from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'
import { store, key } from './store'

import App from './App.vue'
const app = createApp(App)

app.use(ElementPlus, { size: 'small', zIndex: 3000 })
app.use((store as any), key)
for (const name in ElIcons){
	app.component(name, (ElIcons as any)[name])
}
app.mount('#app')
