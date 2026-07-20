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
                border: `2px solid ${highlighted ? 'rgb(242, 0, 0)' : 'rgba(0, 0, 0, 0.1)'}`,
                padding: '12px',
                marginBottom: '8px',
                borderRadius: '8px',
            }}
        >
            <strong>{user.name}</strong>
            <p style={{margin: '8px 0', fontSize: '14px', color: 'rgb(121, 180, 13)'}}>
                {user.email}
            </p>
            <span
                style={{
                    fontSize: '12px',
                    background: user.role === 'admin' ? 'rgb(138, 138, 253)' : 'rgb(210, 12, 81)',
                    padding: '2px 8px',
                    borderRadius: '4px',
                }}
            >
                {user.role}
            </span>
        </div>
    );
}
