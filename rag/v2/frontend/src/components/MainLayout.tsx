// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

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
