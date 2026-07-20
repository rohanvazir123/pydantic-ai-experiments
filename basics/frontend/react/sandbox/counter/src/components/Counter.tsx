import {useState} from 'react';

interface Props {
    label: string;
    startAt?: number; // optional — defaults to 0
    color?: string; // optional — defaults to green below
}

export function Counter({label, startAt = 0, color = 'rgb(0, 165, 11)'}: Props) {
    const [count, setCount] = useState(startAt);
    const isNegative = count < 0;
    const isZero = count === 0;

    return (
        <div
            style={{
                border: '1px solid rgb(207, 204, 221)',
                padding: '16px',
                marginBottom: '8px',
                color,
            }}
        >
            <h2>
                { label}: {count}
            </h2>


            {/* Pattern 1: && — render something or nothing */}
            {isNegative && <p style={{color: 'red'}}>Gone negative!</p>}

            {/* Pattern 2: ternary — render one of two things */}
            <p style={{color: isZero ? 'rgb(127, 241, 13)' : 'rgb(1, 245, 17)'}}>
                {isZero
                    ? 'Counter is at zero'
                    : `${Math.abs(count)} away from zero`}
            </p>

            {/* Pattern 3: early return is for the whole component — demonstrated below */}
            <button onClick={() => setCount((prev) => prev + 1)}>+</button>
            <button
                onClick={() => setCount((prev) => prev - 1)}
                style={{margin: '0 8px'}}
            >−</button>
            <button onClick={() => setCount(0)}>Reset</button>
        </div>
    );
}
