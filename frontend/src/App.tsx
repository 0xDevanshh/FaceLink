import React, { useCallback, useEffect, useRef, useState } from 'react'
import UploadView from './components/UploadView'
import ProgressView from './components/ProgressView'
import ResultView from './components/ResultView'
import EvidenceView from './components/EvidenceView'
import SettingsView from './components/SettingsView'
import { api } from './api/client'
import type { CaseResult, SSEEvent } from './types/api'

export type View = 'upload' | 'progress' | 'result' | 'evidence' | 'settings'

export interface ScanState {
  caseId: string
  events: SSEEvent[]
  result: CaseResult | null
  done: boolean
  failed: boolean
}

export default function App() {
  const [view, setView]   = useState<View>('upload')
  const [scan, setScan]   = useState<ScanState | null>(null)

  // Keep SSE subscription at App level so it survives view changes
  const sseCloseRef  = useRef<(() => void) | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const deadRef      = useRef(false)

  // Clean up any active SSE/poll when a new scan starts or component unmounts
  function _cleanup() {
    deadRef.current = true
    if (pollTimerRef.current) { clearTimeout(pollTimerRef.current); pollTimerRef.current = null }
    if (sseCloseRef.current)  { sseCloseRef.current(); sseCloseRef.current = null }
  }

  useEffect(() => () => _cleanup(), []) // cleanup on unmount

  const fetchResult = useCallback(async (caseId: string) => {
    for (let i = 0; i < 20 && !deadRef.current; i++) {
      try {
        const result = await api.getResult(caseId)
        if (!deadRef.current) {
          setScan((s) => s ? { ...s, result, done: true } : s)
          setView('result')
        }
        return
      } catch {
        await new Promise((r) => setTimeout(r, 800))
      }
    }
    if (!deadRef.current) {
      setScan((s) => s ? { ...s, done: true, failed: true } : s)
      setView('result')
    }
  }, [])

  const pollUntilDone = useCallback((caseId: string) => {
    let tries = 0
    const tick = async () => {
      if (deadRef.current) return
      try {
        const st = await api.getStatus(caseId)
        if (st.status === 'done') { fetchResult(caseId); return }
        if (st.status === 'failed') {
          if (!deadRef.current) {
            setScan((s) => s ? { ...s, done: true, failed: true } : s)
            setView('result')
          }
          return
        }
      } catch { /* hiccup */ }
      if (++tries < 240 && !deadRef.current) {
        pollTimerRef.current = setTimeout(tick, 1500)
      }
    }
    pollTimerRef.current = setTimeout(tick, 1500)
  }, [fetchResult])

  function onScanStarted(caseId: string) {
    _cleanup()
    deadRef.current = false

    setScan({ caseId, events: [], result: null, done: false, failed: false })
    setView('progress')

    // Start SSE at App level — survives view unmounts
    sseCloseRef.current = api.subscribeEvents(
      caseId,
      (evt) => {
        if (deadRef.current) return
        setScan((s) => s ? { ...s, events: [...s.events, evt] } : s)
      },
      () => {
        // SSE stream ended cleanly (stage=done received)
        if (deadRef.current) return
        fetchResult(caseId)
      },
      () => {
        // SSE error — fall back to polling
        if (deadRef.current) return
        pollUntilDone(caseId)
      },
    )

    // Safety fallback: start polling after 8s in case SSE never fires onDone
    pollTimerRef.current = setTimeout(() => {
      if (!deadRef.current) {
        setScan((s) => {
          // Only kick off poll if scan isn't done yet
          if (s && !s.done) pollUntilDone(caseId)
          return s
        })
      }
    }, 8000)
  }

  function onReset() {
    _cleanup()
    setScan(null)
    setView('upload')
  }

  return (
    <div className="min-h-screen bg-surface font-mono">
      {/* Top nav */}
      <header className="border-b border-border bg-surface-1 px-6 py-3 flex items-center justify-between">
        <button
          onClick={onReset}
          className="flex items-center gap-2 text-accent font-bold text-lg hover:text-accent-dim transition-colors focus-visible:ring-2"
          aria-label="FaceChain home"
        >
          <span className="text-2xl">⛓</span>
          <span>FaceChain</span>
          <span className="text-xs text-muted ml-1 font-normal">v1.0.0</span>
        </button>
        <nav className="flex gap-4 text-sm" aria-label="Main navigation">
          {scan && (
            <>
              <NavBtn active={view === 'progress'} onClick={() => setView('progress')}>Progress</NavBtn>
              <NavBtn active={view === 'result'} onClick={() => setView('result')} disabled={!scan.done}>Result</NavBtn>
              <NavBtn active={view === 'evidence'} onClick={() => setView('evidence')} disabled={!scan.done}>Evidence</NavBtn>
            </>
          )}
          <NavBtn active={view === 'settings'} onClick={() => setView('settings')}>Settings</NavBtn>
        </nav>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        {view === 'upload' && (
          <UploadView onScanStarted={onScanStarted} />
        )}
        {view === 'progress' && scan && (
          <ProgressView
            caseId={scan.caseId}
            events={scan.events}
          />
        )}
        {view === 'result' && scan && (
          <ResultView
            scan={scan}
            onViewEvidence={() => setView('evidence')}
            onNewScan={onReset}
          />
        )}
        {view === 'evidence' && scan && (
          <EvidenceView caseId={scan.caseId} result={scan.result} />
        )}
        {view === 'settings' && <SettingsView />}
      </main>

      <footer className="border-t border-border mt-16 px-6 py-4 text-xs text-muted text-center">
        FaceChain — forensic face matching with blockchain attestation.
        <span className="mx-2">|</span>
        Authorized use only: your own images, public figures, or images you are authorized to investigate.
        <span className="mx-2">|</span>
        Not an identity system. Never outputs "this IS person X".
      </footer>
    </div>
  )
}

function NavBtn({
  children, active, onClick, disabled,
}: {
  children: React.ReactNode; active: boolean; onClick: () => void; disabled?: boolean
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-3 py-1 rounded text-sm transition-colors focus-visible:ring-2
        ${active ? 'bg-surface-3 text-accent' : 'text-muted hover:text-gray-100'}
        ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}`}
      aria-current={active ? 'page' : undefined}
    >
      {children}
    </button>
  )
}
