import { ElMessage } from 'element-plus'
import { store, uprPlayer as player } from './store'
import * as Tone from 'tone'
import { Instrument } from 'tone/build/esm/instrument/Instrument'
import { ref, Ref, watch, WatchStopHandle } from 'vue'
// @ts-ignore  // 需要给 SampleLibrary 编写类型定义
import { SampleLibrary } from './Tonejs-Instruments'
import { instrumentMap } from './typings/ui'


export const pxPerSecond = 180  // 音符每秒对应的音符条长度(px)
export const maxVelocity = 127

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


export function noteBgColor(velocity: number): string {
  // note.style.backgroundColor = '#f09d63'
  // note.style.backgroundColor = 'rgba(180, 75, 0, 0.4)'

  // const colorNum = Math.round(((store.state.noteVelocity - 1) / 126) * 400)
  // const hue = Math.floor(colorNum / 11) + 20
  // const light = (colorNum % 11) * 3 + 50
  // 色相 20-60 乘 亮度 50-60 共400种变化

  let hue = (127 - velocity + 1) / 126
  // 将数字翻转后归一，使力度大的对应值小，方便后面对应深色
  hue = hue * 60 + 10
  return `hsl(${hue}, 100%, 60%)`
}

export class noteLengthSlider {
  sliders: HTMLElement[]
  // 左右滑块
  pianoroll: HTMLElement
  note?: HTMLElement
  // 被控制的音符条
  selectedId: number|null
  // true的时候阻止noteAdd
  originVelocity: number
  velocityWatchStop?: WatchStopHandle

  protected _prMousemoveBind: any
  protected _prMouseupBind: any
  // 绑定了this的鼠标事件处理函数

  constructor(ns: any) {
    this.sliders = Array.from(ns)
    this.selectedId = null
    this.pianoroll = document.querySelector('#pianoroll')!
    this._prMousemoveBind = this._prMousemove.bind(this)
    this._prMouseupBind = this._prMouseup.bind(this)
    this.sliders.forEach(x => x.addEventListener('mousedown', this._prMousedown.bind(this)))
    this.originVelocity = store.state.noteVelocity
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

    
    this.velocityWatchStop = watch(() => store.state.noteVelocity, (val, prev) => {
      if(!this.note) {
        return
      }
      this.note.style.backgroundColor = noteBgColor(val)
      this.note.dataset.velocity = val.toString()
    })
    this.originVelocity = store.state.noteVelocity
    store.state.noteVelocity = parseInt(note.dataset.velocity!)
  }

  hide() {
    this.sliders.forEach(x => x.style.display = 'none')
    if(!this.note) {
      return
    }
    this.velocityWatchStop!()
    store.state.noteVelocity = this.originVelocity
    // 解除绑定的音符并恢复全局力度
    player.remove(this.note)
    player.add(this.note)
    // 更新Tone.Transport中的音符长度和力度
    player.refreshLenth()
    // 这时note已经是更新后的状态，可能从最后一个变成中间也可能反过来，不如直接刷新
    this.note = undefined
    showMsg('note status changed', 'success')
  }
}

const uiSynth = new Tone.PolySynth(Tone.MonoSynth, {
	"volume": 16,
	"detune": -22,
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
		"attack": 0.05,
		"attackCurve": "linear",
		"decay": 0.49,
		"decayCurve": "exponential",
		"release": 0.99,
		"releaseCurve": "exponential",
		"sustain": 0.09,
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
})

export class uprPlayer {
  instrument: Instrument<any> = uiSynth
  readonly instrumentList: string[] = ['uiSynth', ...SampleLibrary.list]
  readonly instrumentChiMap: instrumentMap = {
    'bass-electric': '电贝司', 
    'bassoon': '大管', 
    'cello': '大提琴', 
    'clarinet': '单簧管', 
    'contrabass': '低音提琴',
    'flute': '长笛', 
    'french-horn': '圆号', 
    'guitar-acoustic': '木吉他', 
    'guitar-electric': '电吉他', 
    'guitar-nylon': '尼龙弦吉他', 
    'harmonium': '脚踏式风琴', 
    'harp': '竖琴', 
    'organ': '管风琴', 
    'piano': '钢琴', 
    'saxophone': '萨克斯', 
    'trombone': '长号', 
    'trumpet': '小号', 
    'tuba': '大号', 
    'violin': '小提琴', 
    'xylophone': '木琴',
  }
  private __currentInst: string = 'uiSynth'

  protected _paused: boolean
  protected _length: number
  // 钢琴卷帘最大长度，不可开启loop使用loopend
  protected _position: Ref<number>
  // 当前播放位置 (秒)，因为无法直接监听transport.seconds
  protected _timerId?: NodeJS.Timer

  constructor(instName: string='uiSynth') {
    SampleLibrary.setExt('.ogg')
    this._paused = false
    this._length = 0
    this._position = ref(0)
    this.setInstrument(instName)
  }

  public get position(): number {
    return this._position.value
  }
  
  public set position(v: number) {
    if(v < 0) {
      v = 0
      showMsg('设定的时间不能小于0', 'warning')
    } else if(v > this._length) {
      v = parseFloat(this._length.toFixed(4))
      showMsg(`设定的时间不能超过upr长度: ${v}s`, 'warning')
    }

    if(v === 0 && this._position.value === v) {
      this._position.value = -1
      setTimeout(() => {
        this._position.value = 0
      }, 0)
      // 数值无所谓，关键是先赋予一个不同的值使vue能监听到数据变动
      // 解决element-plus.input不根据v-model直接将输入显示出来的问题
      return
    }

    this._position.value = v
    Tone.Transport.seconds = v
  }
  
  public get length(): number {
    return this._length
  }

  public get currentInst() : string {
    return this.__currentInst
  }
  
  syncTime() {
    this._position.value = Tone.Transport.seconds
  }
  
  play() {
    if(this._length < 1e-4) {
      showMsg('点击添加音符后才能演奏', 'warning')
      return
    }
    if(!this._paused) {
      Tone.Transport.stop()
      // 再次播放前必须先停止
    }
    Tone.Transport.start()
    this._paused = false
    this._timerId = setInterval(() => {
      this.syncTime()
      if(this._position.value >= this._length) {
        this.stop()
        showMsg('all the notes have been played', 'info')
      }
      // 播放完所有音符就停止
    }, 10)
  }

  pause() {
    Tone.Transport.pause()
    this._paused = true
    clearInterval(Number(this._timerId))
    this.syncTime()
  }

  stop() {
    this.pause()
    // 先同步再停止才能在播放结束时使position留在最后的时刻
    Tone.Transport.stop()
  }

  cancel() {
    this.pause()
    Tone.Transport.cancel()
  }

  add(note: HTMLElement): string {
    // 将该音符添加到transport中等待播放

    const velocity = parseInt(note.dataset.velocity!) / maxVelocity
    const pitch = note.dataset.pitch!
    const duration = parseFloat(note.style.width) / pxPerSecond
    const startTime = parseFloat(note.style.left) / pxPerSecond

    const eventId = Tone.Transport.schedule((time) => {
      this.instrument.triggerAttackRelease(pitch, duration, time, velocity)
    }, startTime)
    // 这里用的都是绝对时间，不需要设置bpm、拍号等等

    this._length = Math.max(this._length, startTime + duration)
    return eventId.toString()
  }

  remove(note: HTMLElement) {
    const end = (parseFloat(note.style.left) + parseFloat(note.style.width)) / pxPerSecond
    if(Math.abs(this._length - end) < 1e-4) {
      this.refreshLenth()
    }
    Tone.Transport.clear(parseInt(note.dataset.eventId || ''))
  }

  refreshLenth() {
    // 通过比较每行最后一个音符找出整首曲子的最后一个音符，进而确定曲子长度
    let newLength: number = 0
    for(let cell of getPrRowArray()) {
      if(!cell.childElementCount) {
        continue
      }
      const {left, width} = (cell.lastElementChild as HTMLElement).style
      const end = (parseFloat(left) + parseFloat(width)) / pxPerSecond
      newLength = Math.max(end, newLength)
    }
    this._length = newLength
  }

  setInstrument(instName: string) {
    if(instName==='uiSynth') {
      this.instrument = uiSynth.toDestination()
      showMsg('uiSynth loaded successfully', 'success')
      return
    }
    this.instrument = SampleLibrary.load({
      instruments: instName,
      baseUrl: './samples/',
      // '/samples/'不行，最终打包时不能被vite转换成相对asar的路径
      onload: () => {showMsg(`${instName} samples loaded successfully`, 'success')}
    }).toDestination()
    this.__currentInst = instName
  }
}

// function notesScheduleTest() {
//   // 调用synth.sync/unsync能方便地接入transport
//   // 但是有延迟而且会有残存的嗡嗡声

//   Tone.Transport.schedule((time) => {
//     utils.pianoSynth.triggerAttackRelease('G5', '4n', time, 0.5)
//   }, 0)
//   Tone.Transport.schedule((time) => {
//     utils.pianoSynth.triggerAttackRelease('A5', '4n', time+0.2, 0.5)
//   }, 1)
//   Tone.Transport.schedule((time) => {
//     utils.pianoSynth.triggerAttackRelease(['B5', 'D#6', 'F#6'], ['2n', '4n', '4n'], time, 0.5)
//   }, 2)
// }



interface dragHooks {
  mousedown?: (...args: any[]) => void,
  mousemove?: (...args: any[]) => void,
  mouseup?: (...args: any[]) => void,
}
interface dragOptions {
  elemStyle?: object,  // CSSStyleDeclaration,
  hooks?: dragHooks,
}

export class draggableElement {
  elem: HTMLElement
  // 左右滑块
  parent: HTMLElement
  hooks: dragHooks

  protected _mousedownBind: any
  protected _mousemoveBind: any
  protected _mouseupBind: any
  // 绑定了this的鼠标事件处理函数

  constructor(tag: string, className: string, options: dragOptions = {}) {
    this.elem = document.createElement(tag)
    this.elem.className = className
    const {elemStyle, hooks} = options
    if(elemStyle) {
      Object.assign(this.elem.style, elemStyle)
    }
    this.hooks = hooks || {}
    this.parent = document.body
    // 仅是为了初始化，并非真的要挂到body上
    this._mousedownBind = this._mousedown.bind(this)
    this._mousemoveBind = this._mousemove.bind(this)
    this._mouseupBind = this._mouseup.bind(this)
    this.elem.addEventListener('mousedown', this._mousedownBind)
  }

  mount(parent: string) {
    this.parent = document.querySelector(parent) as HTMLElement
    this.parent.append(this.elem)
  }

  _mousedown(e: HTMLElement) {
    this.parent.addEventListener('mousemove', this._mousemoveBind)
    this.parent.addEventListener('mouseup', this._mouseupBind)
    this.elem.classList.add('active')
    this.hooks.mousedown && this.hooks.mousedown()
  }

  _mousemove(e: MouseEvent) {
    const newLeft = parseFloat(this.elem.style.left) + e.movementX
    this.elem.style.left = `${newLeft}px`
    this.hooks.mousemove && this.hooks.mousemove()
    // mousemove的钩子十分耗费性能
  }

  _mouseup(e: MouseEvent) {
    this.parent.removeEventListener('mousemove', this._mousemoveBind)
    this.parent.removeEventListener('mouseup', this._mouseupBind)
    this.elem.classList.remove('active')
    this.hooks.mouseup && this.hooks.mouseup()
  }
}
