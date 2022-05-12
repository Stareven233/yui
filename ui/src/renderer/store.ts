import { InjectionKey, reactive } from 'vue'
import { createStore, useStore as baseUseStore, Store } from 'vuex'
import { Upr } from './typings/ui'
import * as utils from './utils'


export const uprPlayer = new utils.uprPlayer()

export interface State {
  noteTimeRatio: number,
  noteVelocity: number,
  upr: Upr,
  uprLoading: boolean,
}

export const key: InjectionKey<Store<State>> = Symbol()

export const store = createStore<State>({
  state: reactive({
    noteTimeRatio: 1,
    noteVelocity: 64,
    upr: {fps: 62.5, pianoroll: [], updatedAt: 0, qpm: 120, timeSignature: [4, 4], keySignature: 0,},
    uprLoading: false,
  }),
  mutations: {
    noteTimeRatio (state: State, ratio: number) {
      state.noteTimeRatio = ratio
    },
    noteVelocity (state: State, velocity: number) {
      if (velocity > 127) 
        velocity = 127
      else if (velocity < 1) 
        velocity = 1
      state.noteVelocity = velocity
    },
    updateUPR (state: State, upr: Upr) {
      state.upr = upr
    },
    changeQPM (state: State, qpm: number) {
      state.upr.qpm = qpm
    },
    changeTimeSignature (state: State, ts: number[]) {
      state.upr.timeSignature = ts
    },
    changeKeySignature (state: State, ks: number) {
      state.upr.keySignature = ks
    },
    uprLoading (state: State, isLoading: boolean) {
      state.uprLoading = isLoading
    }
  }
})

export function useStore() {
  return baseUseStore(key)
}

// nextTick(() => {
//   console.log('useStore() :>> ', useStore())
// })
