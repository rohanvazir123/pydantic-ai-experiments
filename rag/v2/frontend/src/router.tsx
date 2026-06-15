import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { PrivateRoute } from '@/components/PrivateRoute'
import { LoginPage }    from '@/pages/LoginPage'
import { ChatPage }     from '@/pages/ChatPage'
import { LogsPage }     from '@/pages/LogsPage'
import { IngestPage }   from '@/pages/IngestPage'
import { MemoriesPage } from '@/pages/MemoriesPage'

export function Router() {
  return (
    <BrowserRouter>
      <Toaster position="bottom-right" toastOptions={{ style: { background: '#ffffff', color: '#111827', border: '1px solid #e1e4eb' } }} />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/chat"  element={<PrivateRoute><ChatPage /></PrivateRoute>} />
        <Route path="/logs"  element={<PrivateRoute><LogsPage /></PrivateRoute>} />
        <Route path="/ingest"    element={<PrivateRoute><IngestPage /></PrivateRoute>} />
        <Route path="/memories"  element={<PrivateRoute><MemoriesPage /></PrivateRoute>} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
