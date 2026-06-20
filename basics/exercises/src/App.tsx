import { Counter } from './components/Counter'

export default function App() {
  return (
    <>
      <h1>React Day 3</h1>
      <Counter label="Apples" />
      <Counter label="Oranges" startAt={5} />
    </>
  )
}
