import { useRef, useEffect } from 'react'
import { UserCard } from './UserCard'

interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'
}

interface Props {
  users: User[]
}

export function ScrollList({ users }: Props) {
  // DOM ref — attach to the last element so we can scroll to it
  const bottomRef = useRef<HTMLDivElement>(null)

  // Scroll into view whenever the users array changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [users])

  return (
    <div style={{ maxHeight: '300px', overflowY: 'auto', border: '1px solid #ccc', cursor: 'pointer', padding: '8px' }}>
      {users.map(u => <UserCard key={u.id} user={u} />)}
      <div ref={bottomRef} />  {/* invisible sentinel — we scroll to this */}
    </div>
  )
}