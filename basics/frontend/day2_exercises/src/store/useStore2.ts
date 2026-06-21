// src/store/useStore.ts
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

interface StoreState {
  // 1. Data/State properties
  count: number
  theme: 'light' | 'dark'
  
  // 2. Grouped Actions namespace
  actions: {
    increment: () => void
    decrement: () => void
    toggleTheme: () => void
    reset: () => void
  }
}

export const useStore2 = create<StoreState>()(
  persist(
    (set) => ({
      // Initial state values
      count: 0,
      theme: 'light',

      // Grouping actions under a single static object key
      actions: {
        increment: () => set((state) => ({ count: state.count + 1 })),
        decrement: () => set((state) => ({ count: state.count - 1 })),
        toggleTheme: () => set((state) => ({ 
          theme: state.theme === 'light' ? 'dark' : 'light' 
        })),
        reset: () => set({ count: 0, theme: 'light' }),
      },
    }),
    {
      name: 'app-storage', // The unique key used in localStorage
      storage: createJSONStorage(() => localStorage), // Tells Zustand to use localStorage
      
      // OPTIONAL BEST PRACTICE: Only save data, do NOT save the action functions to storage
      partialize: (state) => ({ 
        count: state.count, 
        theme: state.theme 
      }),
    }
  )
)
