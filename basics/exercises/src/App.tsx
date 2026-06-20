import './App.css'
import { Counter } from './components/Counter'

export default function App() {
  return (
    <>
      <h1 className="page-title">React Day 3</h1>
      <div className="counter-grid">
        <Counter label="Apples" />
        <Counter label="Oranges" startAt={5} />
      </div>
    </>
  )
}
