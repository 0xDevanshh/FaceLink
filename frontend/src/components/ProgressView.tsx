import React, { useEffect, useRef } from 'react'
import { api } from '../api/client'
import type { CaseResult, SSEEvent } from '../types/api'

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

function parseStageKey(stage: string): string {
  return stage.split(':')[0]
}

function buildStageMap(events: SSEEvent[]): Record<string, StageState> {
  const map: Record<string, StageState> = {}
  for (const evt of events) {
    const key = parseStageKey(evt.stage)
    if (!map[key]) map[key] = { status: 'idle', details: [] }
    if (evt.status === 'start') map[key].status = 'running'
    else if (evt.status === 'ok') map[key].status = 'ok'
    else if (evt.status === 'fail') map[key].status = 'fail'
    else if (evt.status === 'info' || evt.status === 'skip') map[key].status = 'skip'
    if (evt.detail) map[key].details.push(evt.detail)
  }
  return map
}

function engineChips(events: SSEEvent[]): { name: string; status: StageStatus; detail: string }[] {
  const engines: Record<string, { status: StageStatus; detail: string }> = {}
  for (const evt of events) {
    if (!evt.stage.startsWith('search:')) continue
    const name = evt.stage.slice(7)
    if (!engines[name]) engines[name] = { status: 'idle', detail: '' }
    if (evt.status === 'start') engines[name].status = 'running'
    else if (evt.status === 'ok') engines[name].status = 'ok'
    else if (evt.status === 'fail') engines[name].status = 'fail'
    if (evt.detail) engines[name].detail = evt.detail
  }
  return Object.entries(engines).map(([name, v]) => ({ name, ...v }))
}

function candidateLines(events: SSEEvent[]): string[] {
  return events
    .filter((e) => e.stage === 'verify:candidate')
    .map((e) => e.detail)
}

const STATUS_ICON: Record<StageStatus, string> = {
  idle: '○',
  running: '◉',
  ok: '✓',
  fail: '✗',
  skip: '–',
}
const STATUS_COLOR: Record<StageStatus, string> = {
  idle: 'text-muted',
  running: 'text-accent animate-pulse',
  ok: 'text-success',
  fail: 'text-danger',
  skip: 'text-warn',
}

interface Props {
  caseId: string
  events: SSEEvent[]
  onEvent: (e: SSEEvent) => void
  onDone: (result: CaseResult) => void
  onFailed: () => void
}

export default function ProgressView({ caseId, events, onEvent, onDone, onFailed }: Props) {
  const logRef = useRef<HTMLDivElement>(null)
  const closerRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (closerRef.current) return // already subscribed
    let dead = false

    closerRef.current = api.subscribeEvents(
      caseId,
      (evt) => {
        if (dead) return
        onEvent(evt)
      },
      async () => {
        if (dead) return
        dead = true
        try {
          // poll until result is ready
          let tries = 0
          while (tries++ < 20) {
            await new Promise((r) => setTimeout(r, 500))
            const st = await api.getStatus(caseId)
            if (st.status === 'done') {
              const result = await api.getResult(caseId)
              onDone(result)
              return
            }
            if (st.status === 'failed') { onFailed(); return }
          }
          onFailed()
        } catch {
          onFailed()
        }
      },
      () => { if (!dead) { dead = true; onFailed() } },
    )

    return () => {
      dead = true
      closerRef.current?.()
    }
  }, [caseId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [events.length])

  const stageMap = buildStageMap(events)
  const chips = engineChips(events)
  const candidates = candidateLines(events)
  const isDone = events.some((e) => e.stage === 'done' || e.stage === 'error')

  return (
    <div className="max-w-3xl mx-auto" aria-live="polite" aria-label="Scan progress">
      <h1 className="text-2xl font-bold text-accent mb-1">Scanning…</h1>
      <p className="text-muted text-sm mb-6">
        Case <span className="font-mono text-gray-300">{caseId}</span>
        {isDone && <span className="ml-3 text-success">● Complete</span>}
      </p>

      {/* Stage stepper */}
      <ol className="space-y-2 mb-8" aria-label="Pipeline stages">
        {STAGES.map((stage) => {
          const s = stageMap[stage.key] ?? { status: 'idle', details: [] }
          return (
            <li key={stage.key}
                className={`flex flex-col px-4 py-2 rounded border
                  ${s.status === 'running' ? 'border-accent/60 bg-surface-2' : 'border-border bg-surface-1'}`}>
              <div className="flex items-center gap-3">
                <span className={`font-mono text-lg w-6 text-center ${STATUS_COLOR[s.status]}`}
                      aria-hidden="true">
                  {STATUS_ICON[s.status]}
                </span>
                <span className={`font-mono text-sm ${s.status === 'idle' ? 'text-muted' : 'text-gray-100'}`}>
                  {stage.label}
                </span>
                {s.status === 'running' && (
                  <span className="ml-auto text-xs text-accent animate-pulse" aria-label="Running">running</span>
                )}
              </div>
              {s.details.length > 0 && (
                <ul className="ml-9 mt-1 space-y-0.5" aria-label={`${stage.label} details`}>
                  {s.details.slice(-4).map((d, i) => (
                    <li key={i} className="text-xs text-muted font-mono truncate" title={d}>
                      {d}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          )
        })}
      </ol>

      {/* Engine chips */}
      {chips.length > 0 && (
        <section className="mb-6" aria-labelledby="engine-status-heading">
          <h2 id="engine-status-heading" className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">
            Search Engines
          </h2>
          <div className="flex flex-wrap gap-2" role="list" aria-label="Engine status chips">
            {chips.map((chip) => (
              <div
                key={chip.name}
                role="listitem"
                title={chip.detail}
                className={`px-3 py-1 rounded-full text-xs font-mono border
                  ${chip.status === 'ok' ? 'border-success/60 text-success bg-green-900/10' :
                    chip.status === 'fail' ? 'border-danger/60 text-danger bg-red-900/10' :
                    chip.status === 'running' ? 'border-accent/60 text-accent bg-blue-900/10 animate-pulse' :
                    'border-border text-muted'}`}
                aria-label={`${chip.name}: ${chip.status}`}
              >
                {chip.name} · {chip.status}
                {chip.detail && <span className="ml-1 opacity-60 text-[10px]">{chip.detail.slice(0, 20)}</span>}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Candidate stream */}
      {candidates.length > 0 && (
        <section aria-labelledby="candidates-heading">
          <h2 id="candidates-heading" className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">
            Candidates ({candidates.length})
          </h2>
          <div
            ref={logRef}
            className="bg-surface-1 border border-border rounded p-3 h-40 overflow-y-auto scrollbar-thin font-mono text-xs space-y-0.5"
            aria-label="Candidate verification log"
            aria-live="polite"
          >
            {candidates.map((line, i) => (
              <div key={i} className="text-gray-300 truncate" title={line}>{line}</div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
