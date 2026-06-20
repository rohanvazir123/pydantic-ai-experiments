import './App.css';

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
    return (
        <>
            <h1 className='page-title'>React Day 3</h1>
            <div className='counter-grid'>
                <Counter label='Apples' />
                <Counter label='Oranges' startAt={5} />
                <hr />
                <UserList users={USERS} highlightedId={1} />
            </div>
        </>
    );
}
