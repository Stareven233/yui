import { ElMessage } from 'element-plus'
import { store } from './store'
import * as Tone from 'tone'

export const pxPerSecond = 180  // 音符每秒对应的音符条长度(px)

export const showMsg = (msg: string, type: any) => {
  ElMessage({
    showClose: true,
    message: msg,
    type: type,
  })
}

export const getPrRowArray = () => {
  const prRow = document.querySelectorAll('#pianoroll .el-table .el-table__row > .el-table_1_column_2 > .cell')
  return Array.from(prRow)
  // HTMLCollection, NodeList 竟然不能迭代 却可以转换为Array
}

export const clearPianoRoll = () => {
  for(const cell of getPrRowArray()) {
    cell.innerHTML = ''
  }
}

export class noteLengthSlider {
  sliders: HTMLElement[]
  // 左右滑块
  pianoroll: HTMLElement
  note?: HTMLElement
  // 被控制的音符条
  selectedId: number|null
  // true的时候组织noteAdd

  private _prMousemoveBind: any
  private _prMouseupBind: any
  // 绑定了this的鼠标事件处理函数

  constructor(ns: any) {
    this.sliders = Array.from(ns)
    this.selectedId = null
    this.pianoroll = document.querySelector('#pianoroll')!
    this._prMousemoveBind = this._prMousemove.bind(this)
    this._prMouseupBind = this._prMouseup.bind(this)
    this.sliders.forEach(x => x.addEventListener('mousedown', this._prMousedown.bind(this)))
  }

  _prMousemove(e: MouseEvent) {
    const slider = this.sliders[this.selectedId!]
    const newLeft = parseFloat(slider.style.left) + e.movementX
    const newWidth = parseFloat(this.note!.style.width) - e.movementX * (-this.selectedId! || 1)
    // selectedId=0时表示控制左滑块，往左移的时候movementX < 0, width正好加上其绝对值，反之右滑块则减去
    if(newWidth < 3.75 * pxPerSecond / store.state.upr.qpm) {
      return
      // 简化的表达式，使音符条长度最小不能小于六十四分音符长度
    }
    this.note!.style.width = `${newWidth}px`
    if(this.selectedId === 0) {
      this.note!.style.left = `${newLeft + 2}px`
    }
    slider.style.left = `${newLeft}px`
  }

  _prMouseup(e: MouseEvent) {
    // e.stopImmediatePropagation()  // 没用，无法阻止向noteAdd的传播
    // document.onmousemove = null
    // document.onmouseup = null
    setTimeout(() => {
      this.selectedId = null
    }, 0)
    this.pianoroll.removeEventListener('mousemove', this._prMousemoveBind)
    this.pianoroll.removeEventListener('mouseup', this._prMouseupBind, true)  // 要跟add时一样加上第三个参数
  }

  _prMousedown(e: Event) {
    this.selectedId = (e.target as HTMLElement).classList[1] === 'left'? 0: 1
    // document.onmousemove = this._prMousemove
    // document.onmouseup = this._prMouseup
    this.pianoroll.addEventListener('mousemove', this._prMousemoveBind)
    this.pianoroll.addEventListener('mouseup', this._prMouseupBind, true)
  }

  showAt(row: HTMLElement, note: HTMLElement) {
    this.note = note
    const hasi = [note.style.left, note.style.width].map(x => parseFloat(x))
    hasi[1] += hasi[0]
    // note两段的位置
    this.sliders.forEach((x, idx) => {
      row.appendChild(x)
      x.style.display = 'inline-block'
      x.style.left = `${hasi[idx] - 2}px`
    })
  }

  hide() {
    this.sliders.forEach(x => x.style.display = 'none')
  }
}

export const pianoSynth = new Tone.PolySynth(Tone.MonoSynth, {
	"volume": 16,
	"detune": -19,
	"portamento": 10,
	"envelope": {
		"attack": 0.05,
		"attackCurve": "linear",
		"decay": 0.5,
		"decayCurve": "exponential",
		"release": 1,
		"releaseCurve": "exponential",
		"sustain": 0.05
	},
	"filter": {
		"Q": 0,
		"detune": 10,
		"frequency": 0,
		"gain": 0,
		"rolloff": -12,
		"type": "lowpass"
	},
	"filterEnvelope": {
		"attack": 0.06,
		"attackCurve": "linear",
		"decay": 0.51,
		"decayCurve": "exponential",
		"release": 1.1,
		"releaseCurve": "exponential",
		"sustain": 0.11,
		"baseFrequency": 320,
		"exponent": 1,
		"octaves": -1
	},
	"oscillator": {
		"detune": -19,
		"frequency": 440,
		"partialCount": 4,
		"partials": [
			0.25,
			0.00,
			0.25,
			0.12
		],
		"phase": 0,
    // @ts-ignore
		"type": "custom"
	}
}).toDestination()
