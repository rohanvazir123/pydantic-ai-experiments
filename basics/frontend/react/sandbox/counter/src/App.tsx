import { useState } from 'react'
import './App.css';

import { Counter } from './components/Counter'
import { UserCard } from './components/UserCard'
import { UserList } from './components/UserList';
import { ScrollList } from './components/ScrollList';

export default function App() {

  const [query, setQuery] = useState('')
  const SAMPLE_USERS = [
    { id: 1, name: 'Ada Lovelace', email: 'ada@example.com', role: 'admin' as const },
    { id: 2, name: 'Grace Hopper', email: 'grace@example.com', role: 'viewer' as const },
    { id: 3, name: 'Alan Turing', email: 'alan@example.com', role: 'viewer' as const },
    { id: 4, name: 'Grace Kelly', email: 'grace.kelly@example.com', role: 'admin' as const },
    { id: 5, name: 'Alan Watts', email: 'alan.watts@example.com', role: 'viewer' as const },
    { id: 6, name: 'Ada Newsom', email: 'ada.newsom@example.com', role: 'admin' as const },
    { id: 7, name: 'Alan Goodman', email: 'alan.goodman@example.com', role: 'viewer' as const },
    { id: 8, name: 'Grace Schafer', email: 'grace.schafer@example.com', role: 'admin' as const },
    { id: 9, name: 'Alan Kingsley', email: 'alan.kingsley@example.com', role: 'viewer' as const },
    { id: 10, name: 'Ada Patel', email: 'ada.patel@example.com', role: 'admin' as const },
    { id: 11 , name: 'Alan Newman', email: 'alan.newman@example.com', role: 'viewer' as const }
  ];

  // derived value — filter inline, no separate useState
  const visibleUsers = SAMPLE_USERS.filter(u =>
    u.name.toLowerCase().includes(query.toLowerCase())
  )
 
  return (
    <>
      <h1 style={{color: 'rgb(88, 170, 160)'}}>React Day 1</h1>
      <Counter label="Bananas" color="rgb(252, 242, 40)" />
      <Counter label="Strawberries" startAt={5000000} color="rgb(214, 8, 66)" />
      {/* <UserCard user={SAMPLE_USERS[0]} highlighted={true} /> */}
      <UserList users={visibleUsers} highlightedId={SAMPLE_USERS[2].id} />
      <ScrollList users={SAMPLE_USERS} />
      <input
          type="text"
          placeholder="Filter users..."
          value={query}                           // controlled — value driven by state
          onChange={e => setQuery(e.target.value)} // update state on every keystroke
          style={{ marginBottom: '12px', padding: '8px', width: '100%' }}
        />
        <p>{visibleUsers.length} user(s) found</p>
        <ScrollList users={visibleUsers} />

    </>
  )
}
