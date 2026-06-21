import './App.css'
import { useStore } from './store/useStore'
import { useShallow } from 'zustand/react/shallow'
import { useStore2 } from './store/useStore2'

// ✅ Best Practice: select individual values — avoids re-renders when unrelated state changes
export function App1() {
  const count = useStore((state) => state.count1)
  const increment = useStore((state) => state.increment1)
  const reset = useStore((state) => state.reset1)

  return (
    <div className="panel panel-blue">
      <div className="panel-header">
        <h2>Pattern 1</h2>
        <h1>Individual selectors</h1>
      </div>
      <div className="count accent-blue">{count}</div>
      <div className="btn-row">
        <button className="btn-primary btn-blue" onClick={increment}>Increment</button>
        <button className="btn-ghost" onClick={reset}>Reset</button>
      </div>
    </div>
  )
}

// ✅ useShallow: select multiple values together without causing extra re-renders
export function App2() {
  const { count, increment, reset } = useStore(
    useShallow((state) => ({
      count: state.count2,
      increment: state.increment2,
      reset: state.reset2,
    }))
  )

  return (
    <div className="panel panel-violet">
      <div className="panel-header">
        <h2>Pattern 2</h2>
        <h1>useShallow selector</h1>
      </div>
      <div className="count accent-violet">{count}</div>
      <div className="btn-row">
        <button className="btn-primary btn-violet" onClick={increment}>Increment</button>
        <button className="btn-ghost" onClick={reset}>Reset</button>
      </div>
    </div>
  )
}

// ✅ Grouped actions object — the actions reference is stable, never causes re-renders
export function App3() {
  const { count, theme } = useStore2(
    useShallow((state) => ({ count: state.count, theme: state.theme }))
  )
  const { increment, decrement, toggleTheme, reset } = useStore2((state) => state.actions)

  const isDark = theme === 'dark'

  return (
    <div className="panel panel-emerald" style={{
      background: isDark ? '#021a12' : '#f0fdf4',
      color: isDark ? '#e2e8f0' : '#0f172a',
      transition: 'background 0.3s, color 0.3s',
    }}>
      <div className="panel-header" style={{ borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)' }}>
        <h2>Pattern 3</h2>
        <h1>Actions object + persist</h1>
      </div>
      <div className="count accent-emerald">{count}</div>
      <p>Theme: <strong>{theme}</strong> · refresh to see persist</p>
      <div className="btn-row">
        <button className="btn-primary btn-emerald" onClick={increment}>+</button>
        <button className="btn-primary btn-emerald" onClick={decrement}>−</button>
        <button className="btn-ghost" onClick={toggleTheme}>Toggle theme</button>
        <button className="btn-ghost" onClick={reset}>Reset</button>
      </div>
    </div>
  )
}
