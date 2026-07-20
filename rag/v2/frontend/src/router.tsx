// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { PrivateRoute } from '@/components/PrivateRoute'
import { MainLayout }   from '@/components/MainLayout'
import { LoginPage }    from '@/pages/LoginPage'
import { ChatPage }     from '@/pages/ChatPage'
import { LogsPage }     from '@/pages/LogsPage'
import { IngestPage }   from '@/pages/IngestPage'
import { MemoriesPage } from '@/pages/MemoriesPage'

function Private({ children }: { children: React.ReactNode }) {
  return (
    <PrivateRoute>
      <MainLayout>{children}</MainLayout>
    </PrivateRoute>
  )
}

export function Router() {
  return (
    <BrowserRouter>
      <Toaster position="bottom-right" toastOptions={{ style: { background: '#ffffff', color: '#111827', border: '1px solid #e1e4eb' } }} />
      <Routes>
        <Route path="/login"     element={<LoginPage />} />
        <Route path="/chat"      element={<Private><ChatPage /></Private>} />
        <Route path="/logs"      element={<Private><LogsPage /></Private>} />
        <Route path="/ingest"    element={<Private><IngestPage /></Private>} />
        <Route path="/memories"  element={<Private><MemoriesPage /></Private>} />
        <Route path="*"          element={<Navigate to="/chat" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
