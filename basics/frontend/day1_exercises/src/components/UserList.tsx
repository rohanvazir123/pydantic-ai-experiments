import {UserCard} from './UserCard';

interface User {
    id: number;
    name: string;
    email: string;
    role: 'admin' | 'viewer';
}

interface Props {
    users: User[];
    highlightedId?: number;
}

export function UserList({users, highlightedId}: Props) {
    if (users.length === 0) {
        return <p>No users found.</p>;
    }

    return (
        <div>
            {users.map((user) => (
                <UserCard
                    key={user.id} // key must be stable and unique — never use array index here
                    user={user}
                    highlighted={user.id === highlightedId}
                />
            ))}
        </div>
    );
}
