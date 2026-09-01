import React, { useEffect, useRef } from 'react'
import type { SSEEvent } from '../types/api'

const STAGES = [
  { key: 'input',    label: '[01] Input validation' },
  { key: 'face',     label: '[02] Face detection + encoding' },
  { key: 'search',   label: '[03] Reverse image search' },
  { key: 'verify',   label: '[04] Candidate verification' },
  { key: 'evidence', label: '[05] Evidence generation' },
  { key: 'chain',    label: '[06] Blockchain attestation' },
  { key: 'readback', label: '[07] On-chain read-back' },
]

type StageStatus = 'idle' | 'running' | 'ok' | 'fail' | 'skip'

interface StageState {
  status: StageStatus
  details: string[]
}

function buildStageMap(events: SSEEvent[]): Record<string, StageState> {
  const map: Record<string, StageState> = {}
  for (const evt of events) {
    const key = evt.stage.split(':')[0]
    if (!map[key]) map[key] = { status: 'idle', details: [] }
    if (evt.status === 'start')      map[key].status = 'running'
    else if (evt.status === 'ok')    map[key].status = 'ok'
    else if (evt.status === 'fail')  map[key].status = 'fail'
    else if (evt.status === 'info' || evt.status === 'skip') map[key].status = 'skip'
    if (evt.detail) map[key].details.push(evt.detail)
  }
  return map
}

function engineChips(events: SSEEvent[]) {
  const engines: Record<string, { status: StageStatus; detail: string }> = {}
  for (const evt of events) {
    if (!evt.stage.startsWith('search:')) continue
    const name = evt.stage.slice(7)
    if (!engines[name]) engines[name] = { status: 'idle', detail: '' }
    if (evt.status === 'start')     engines[name].status = 'running'
    else if (evt.status === 'ok')   engines[name].status = 'ok'
    else if (evt.status === 'fail') engines[name].status = 'fail'
    if (evt.detail) engines[name].detail = evt.detail
  }
  return Object.entries(engines).map(([name, v]) => ({ name, ...v }))
}

const STATUS_ICON: Record<StageStatus, string> = {
  idle: '○', running: '◉', ok: '✓', fail: '✗', skip: '–',
}
const STATUS_COLOR: Record<StageStatus, string> = {
  idle:    'text-muted',
  running: 'text-accent animate-pulse',
  ok:      'text-success',
  fail:    'text-danger',
  skip:    'text-warn',
}

// Props: purely display — App handles all SSE/polling
interface Props {
  caseId: string
  events: SSEEvent[]
}

export default function ProgressView({ caseId, events }: Props) {
  const logRef = useRef<HTMLDivElement>(null)

  // Auto-scroll candidate log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [events.length])

  const stageMap   = buildStageMap(events)
  const chips      = engineChips(events)
  const candidates = events.filter((e) => e.stage === 'verify:candidate').map((e) => e.detail)
  const isDone     = events.some((e) => e.stage === 'done' || e.stage === 'error')
  const eventCount = events.length

  return (
    <div className="max-w-3xl mx-auto" aria-live="polite" aria-label="Scan progress">
      <div className="flex items-center justify-between mb-1">
        <h1 className="text-2xl font-bold text-accent">
          {isDone ? 'Scan complete' : 'Scanning…'}
        </h1>
        {isDone && <span className="text-success text-sm font-mono">● Done</span>}
        {!isDone && eventCount > 0 && (
          <span className="text-muted text-xs font-mono animate-pulse">
            {eventCount} event{eventCount !== 1 ? 's' : ''} received
          </span>
        )}
        {!isDone && eventCount === 0 && (
          <span className="text-warn text-xs font-mono animate-pulse">waiting for pipeline…</span>
        )}
      </div>
      <p className="text-muted text-xs font-mono mb-6">
        case: <span className="text-gray-400">{caseId}</span>
      </p>

      {/* Stage stepper */}
      <ol className="space-y-1.5 mb-8" aria-label="Pipeline stages">
        {STAGES.map((stage) => {
          const s = stageMap[stage.key] ?? { status: 'idle' as StageStatus, details: [] }
          return (
            <li
              key={stage.key}
              className={`flex flex-col px-4 py-2.5 rounded border transition-all
                ${s.status === 'running' ? 'border-accent/60 bg-surface-2 shadow-sm' :
                  s.status === 'ok'      ? 'border-success/40 bg-surface-1' :
                  s.status === 'fail'    ? 'border-danger/40 bg-surface-1' :
                                           'border-border bg-surface-1 opacity-60'}`}
            >
              <div className="flex items-center gap-3">
                <span
                  className={`font-mono text-base w-6 text-center shrink-0 ${STATUS_COLOR[s.status]}`}
                  aria-hidden="true"
                >
                  {STATUS_ICON[s.status]}
                </span>
                <span className={`font-mono text-sm ${s.status === 'idle' ? 'text-muted' : 'text-gray-100'}`}>
                  {stage.label}
                </span>
                {s.status === 'running' && (
                  <span className="ml-auto text-xs text-accent animate-pulse">running…</span>
                )}
              </div>
              {s.details.length > 0 && (
                <div className="ml-9 mt-1 space-y-0.5">
                  {s.details.slice(-3).map((d, i) => (
                    <div key={i} className="text-xs text-muted font-mono truncate" title={d}>↳ {d}</div>
                  ))}
                </div>
              )}
            </li>
          )
        })}
      </ol>

      {/* Engine status chips */}
      {chips.length > 0 && (
        <section className="mb-5" aria-labelledby="engine-heading">
          <h2 id="engine-heading" className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">
            Search Engines
          </h2>
          <div className="flex flex-wrap gap-2">
            {chips.map((chip) => (
              <div
                key={chip.name}
                title={chip.detail}
                className={`px-3 py-1 rounded-full text-xs font-mono border transition-colors
                  ${chip.status === 'ok'      ? 'border-success/60 text-success bg-green-900/10' :
                    chip.status === 'fail'    ? 'border-danger/60 text-danger bg-red-900/10'     :
                    chip.status === 'running' ? 'border-accent/60 text-accent bg-blue-900/10 animate-pulse' :
                                                'border-border text-muted'}`}
                aria-label={`${chip.name}: ${chip.status}`}
              >
                {chip.name}
                {chip.status !== 'idle' && (
                  <span className="ml-1.5 opacity-80">
                    {chip.status === 'ok'      ? `✓ ${chip.detail.slice(0, 20)}` :
                     chip.status === 'running' ? '…' : '✗'}
                  </span>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Candidate verification log */}
      {candidates.length > 0 && (
        <section aria-labelledby="candidates-heading">
          <h2 id="candidates-heading" className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">
            Candidates ({candidates.length})
          </h2>
          <div
            ref={logRef}
            className="bg-surface border border-border rounded p-3 h-44 overflow-y-auto scrollbar-thin font-mono text-xs space-y-0.5"
            aria-live="polite"
            aria-label="Candidate verification log"
          >
            {candidates.map((line, i) => {
              const verified = line.includes('VERIFIED')
              return (
                <div
                  key={i}
                  className={`truncate ${verified ? 'text-success' : 'text-gray-400'}`}
                  title={line}
                >
                  {line}
                </div>
              )
            })}
          </div>
        </section>
      )}
    </div>
  )
}
