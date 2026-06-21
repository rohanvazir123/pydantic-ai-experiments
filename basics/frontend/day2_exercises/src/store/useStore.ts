// src/store/useStore.ts
import { create } from 'zustand'

interface StoreState {
  count1: number
  count2: number
  increment1: () => void
  increment2: () => void
  reset1: () => void
  reset2: () => void
}

export const useStore = create<StoreState>((set) => ({
  count1: 0,
  count2: 0,
  increment1: () => set((state) => ({ count1: state.count1 + 1 })),
  increment2: () => set((state) => ({ count2: state.count2 + 1 })),
  reset1: () => set({ count1: 0 }),
  reset2: () => set({ count2: 0 }),
}))

