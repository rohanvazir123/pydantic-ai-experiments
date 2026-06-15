import { NavBar } from './NavBar'

export function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <NavBar />
      <div className="pl-14 min-h-screen">
        {children}
      </div>
    </>
  )
}
