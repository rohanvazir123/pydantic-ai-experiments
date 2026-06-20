interface User {
    id: number;
    name: string;
    email: string;
    role: 'admin' | 'viewer';
}

interface Props {
    user: User;
    highlighted?: boolean;
}

export function UserCard({user, highlighted = false}: Props) {
    // Early return — render nothing if name is somehow empty
    if (!user.name) return null;

    return (
        <div
            style={{
                border: `2px solid ${highlighted ? 'blue' : '#ccc'}`,
                padding: '12px',
                marginBottom: '8px',
                borderRadius: '8px',
            }}
        >
            <strong>{user.name}</strong>
            <p style={{margin: '4px 0', fontSize: '14px', color: '#666'}}>
                {user.email}
            </p>
            <span
                style={{
                    fontSize: '12px',
                    background: user.role === 'admin' ? '#fde68a' : '#e0e7ff',
                    padding: '2px 8px',
                    borderRadius: '4px',
                }}
            >
                {user.role}
            </span>
        </div>
    );
}
