import './App.css';

import {useState} from 'react';

import {Counter} from './components/Counter';
import {UserList} from './components/UserList';

const USERS = [
    {
        id: 1,
        name: 'Ada Lovelace',
        email: 'ada@example.com',
        role: 'admin' as const,
    },
    {
        id: 2,
        name: 'Grace Hopper',
        email: 'grace@example.com',
        role: 'viewer' as const,
    },
    {
        id: 3,
        name: 'Alan Turing',
        email: 'alan@example.com',
        role: 'viewer' as const,
    },
];

export default function App() {
    const [query, setQuery] = useState('');

    // derived value — filter inline, no separate useState
    const visibleUsers = USERS.filter((u) =>
        u.name.toLowerCase().includes(query.toLowerCase()),
    );
    return (
        <>
            <h1 className='page-title'>React Day 3</h1>
            <div className='counter-grid'>
                <Counter label='Apples' />
                <Counter label='Oranges' startAt={5} />
                <hr />
                <input
                    type='text'
                    placeholder='Filter users...'
                    value={query} // controlled — value driven by state
                    onChange={(e) => setQuery(e.target.value)} // update state on every keystroke
                    style={{
                        marginBottom: '12px',
                        padding: '8px',
                        width: '100%',
                    }}
                />
                <p>{visibleUsers.length} user(s) found</p>
                <UserList users={visibleUsers} />
            </div>
        </>
    );
}
