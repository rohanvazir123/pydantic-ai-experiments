import {useEffect, useRef, useState} from 'react';

import {UserList} from './components/UserList';

// Shape of what the API returns
interface ApiUser {
    id: number;
    name: string;
    email: string;
}

// Convert to our internal User type
function toUser(api: ApiUser) {
    return {
        id: api.id,
        name: api.name,
        email: api.email,
        role: 'viewer' as const,
    };
}

export default function App() {
    const [users, setUsers] = useState<ReturnType<typeof toUser>[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [query, setQuery] = useState('');

    const inputRef = useRef<HTMLInputElement>(null); // DOM ref — always null until mount

    // Auto-focus the input once users have loaded
    useEffect(() => {
        if (!loading) {
            inputRef.current?.focus(); // ?. because current is null before mount
        }
    }, [loading]); // runs when loading changes from true to false

    useEffect(() => {
        const controller = new AbortController();

        async function fetchUsers() {
            try {
                setLoading(true);
                const res = await fetch(
                    'https://jsonplaceholder.typicode.com/users',
                    {
                        signal: controller.signal,
                    },
                );
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data: ApiUser[] = await res.json();
                setUsers(data.map(toUser));
            } catch (err) {
                if ((err as Error).name !== 'AbortError') {
                    setError(
                        err instanceof Error ? err.message : 'Unknown error',
                    );
                }
            } finally {
                setLoading(false);
            }
        }

        fetchUsers();

        return () => controller.abort(); // cleanup: cancel the request on unmount
    }, []); // empty array = run once on mount

    const visibleUsers = users.filter((u) =>
        u.name.toLowerCase().includes(query.toLowerCase()),
    );

    if (loading) return <p>Loading users...</p>;
    if (error) return <p style={{color: 'red'}}>Error: {error}</p>;

    return (
        <>
            <h1>React Day 3</h1>
            <input
                ref={inputRef} // attach the ref to the DOM node
                type='text'
                placeholder='Filter users...'
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{marginBottom: '12px', padding: '8px', width: '100%'}}
            />
            <p>
                {visibleUsers.length} of {users.length} users
            </p>
            <UserList users={visibleUsers} />
        </>
    );
}
