import { InjectionKey } from 'vue'
import { createStore, Store } from 'vuex'

interface Upr {
  fps: number,
  pianoroll: string[],
  qpm: number,
  updatedAt: number,
}

export interface State {
  noteTimeRatio: number,
  noteVelocity: number,
  upr: Upr,
}

export const key: InjectionKey<Store<State>> = Symbol()

export const store = createStore<State>({
  state: {
    noteTimeRatio: 1,
    noteVelocity: 64,
    upr: {fps: 0, pianoroll: [], updatedAt: 0, qpm: 120},
  },
  mutations: {
    noteTimeRatio (state: State, ratio: number) {
      state.noteTimeRatio = ratio
    },
    noteVelocity (state: State, velocity: number) {
      if (velocity > 127) 
        {velocity = 127}
      else if (velocity < 1) 
        {velocity = 1}
      state.noteVelocity = velocity
    },
    updateUPR (stateL: State, upr: Upr) {
      stateL.upr = upr
    },
    changeQPM (stateL: State, qpm: number) {
      stateL.upr.qpm = qpm
    }
  }
})

