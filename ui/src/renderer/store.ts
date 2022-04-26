import { InjectionKey } from 'vue'
import { createStore, Store } from 'vuex'

export interface State {
  noteTimeRatio: number
}

export const key: InjectionKey<Store<State>> = Symbol()

export const store = createStore<State>({
  state: {
    noteTimeRatio: 1
  },
  mutations: {
    noteTimeRatio (state: any, ratio: number) {
      state.noteTimeRatio = ratio
    }
  }
})

