import { InjectionKey } from 'vue'
import { createStore, Store } from 'vuex'

export interface State {
  noteTimeRatio: number,
  noteVelocity: number,
}

export const key: InjectionKey<Store<State>> = Symbol()

export const store = createStore<State>({
  state: {
    noteTimeRatio: 1,
    noteVelocity: 64,
  },
  mutations: {
    noteTimeRatio (state: any, ratio: number) {
      state.noteTimeRatio = ratio
    },
    noteVelocity (state: any, velocity: number) {
      if (velocity > 127) 
        {velocity = 127}
      else if (velocity < 1) 
        {velocity = 1}
      state.noteVelocity = velocity
    },
  }
})

