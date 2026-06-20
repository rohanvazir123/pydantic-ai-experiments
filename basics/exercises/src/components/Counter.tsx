import {useState} from 'react';

interface Props {
    label: string;
    startAt?: number; // optional — defaults to 0
}

export function Counter({label, startAt = 0}: Props) {
    const [count, setCount] = useState(startAt);

    // derived value — do NOT put this in useState
    const isNegative = count < 0;

    return (
        <div
            style={{
                border: '1px solid #ccc',
                padding: '16px',
                marginBottom: '8px',
            }}
        >
            <h2>
                {label}: {count}
            </h2>
            <button onClick={() => setCount((prev) => prev + 1)}>+</button>
            <button
                onClick={() => setCount((prev) => prev - 1)}
                style={{margin: '0 8px'}}
            >
                −
            </button>
            <button onClick={() => setCount(0)}>Reset</button>
            {isNegative && <p style={{color: 'red'}}>Gone negative!</p>}
        </div>
    );
}
