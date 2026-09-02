import React, { useState } from 'react'
import UploadView, { type ScanSettings } from './components/UploadView'
import FaceSelectView, { type FaceChoice } from './components/FaceSelectView'
import ProgressView from './components/ProgressView'
import ResultView from './components/ResultView'
import EvidenceView from './components/EvidenceView'
import SettingsView from './components/SettingsView'
import { api, ApiError } from './api/client'
import type { CaseResult, FaceDetectResponse, SSEEvent } from './types/api'

export type View = 'upload' | 'faces' | 'progress' | 'result' | 'evidence' | 'settings'

interface PendingSelection {
  previewUrl: string
  detection: FaceDetectResponse
  settings: ScanSettings
}

export interface ScanState {
  caseId: string
  events: SSEEvent[]
  result: CaseResult | null
  done: boolean
  failed: boolean
}

export default function App() {
  const [view, setView] = useState<View>('upload')
  const [scan, setScan] = useState<ScanState | null>(null)
  const [pending, setPending] = useState<PendingSelection | null>(null)
  const [selectError, setSelectError] = useState<string | null>(null)
  const [startingScan, setStartingScan] = useState(false)

  function onScanStarted(caseId: string) {
    releasePending()
    setScan({ caseId, events: [], result: null, done: false, failed: false })
    setView('progress')
  }

  function releasePending() {
    setPending((p) => {
      if (p) URL.revokeObjectURL(p.previewUrl)
      return null
    })
  }

  function onSelectFace(file: File, detection: FaceDetectResponse, settings: ScanSettings) {
    setSelectError(null)
    setPending({ previewUrl: URL.createObjectURL(file), detection, settings })
    setView('faces')
  }

  async function onFaceConfirmed(choice: FaceChoice) {
    if (!pending) return
    setStartingScan(true)
    setSelectError(null)
    try {
      const fd = new FormData()
      fd.append('upload_id', pending.detection.upload_id)
      fd.append('engines', pending.settings.engines.join(','))
      fd.append('no_chain', pending.settings.noChain ? 'true' : 'false')
      fd.append('chain_mode', pending.settings.noChain ? 'skip' : 'onchain')
      fd.append('user_declaration', 'true')
      fd.append('selection_mode', choice.mode)
      if (choice.faceIndex !== null) fd.append('face_index', String(choice.faceIndex))
      if (choice.crop) fd.append('crop', choice.crop.join(','))
      const res = await api.startScan(fd)
      onScanStarted(res.case_id)
    } catch (e) {
      setSelectError(
        e instanceof ApiError
          ? `Server error ${e.status}: ${e.message}`
          : 'Network error — is the backend running?',
      )
    } finally {
      setStartingScan(false)
    }
  }

  function onEvent(evt: SSEEvent) {
    setScan((s) => s ? { ...s, events: [...s.events, evt] } : s)
  }

  function onDone(result: CaseResult) {
    setScan((s) => s ? { ...s, result, done: true } : s)
    setView('result')
  }

  function onFailed() {
    setScan((s) => s ? { ...s, done: true, failed: true } : s)
    setView('result')
  }

  function onReset() {
    releasePending()
    setSelectError(null)
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
          <UploadView onScanStarted={onScanStarted} onSelectFace={onSelectFace} />
        )}
        {view === 'faces' && pending && (
          <>
            {selectError && (
              <div role="alert" className="mb-4 max-w-3xl mx-auto text-danger text-sm px-3 py-2
                bg-red-900/20 rounded border border-danger/30">
                {selectError}
              </div>
            )}
            <FaceSelectView
              previewUrl={pending.previewUrl}
              detection={pending.detection}
              onConfirm={onFaceConfirmed}
              onCancel={onReset}
              busy={startingScan}
            />
          </>
        )}
        {view === 'progress' && scan && (
          <ProgressView
            caseId={scan.caseId}
            events={scan.events}
            onEvent={onEvent}
            onDone={onDone}
            onFailed={onFailed}
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
  children,
  active,
  onClick,
  disabled,
}: {
  children: React.ReactNode
  active: boolean
  onClick: () => void
  disabled?: boolean
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
