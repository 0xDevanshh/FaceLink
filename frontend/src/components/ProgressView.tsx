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

// Provider states that are refusals or configuration gaps rather than errors:
// worth showing as a warning, not as a failure of the scan.
const SOFT_PROVIDER_STATES = ['CHALLENGED', 'RATE_LIMITED', 'TIMEOUT', 'NOT_CONFIGURED',
  'NO_RESULTS', 'CANCELLED']

/**
 * Provider chips.
 *
 * The backend prefixes every provider event with `search:<engine>` and puts the
 * terminal `ProviderStatus` at the head of the detail string, so the label a
 * user sees is the same word recorded in the evidence bundle — CHALLENGED and
 * TIMEOUT stay distinguishable instead of collapsing into "failed".
 */
function engineChips(events: SSEEvent[]): {
  name: string; status: StageStatus; detail: string; providerStatus: string
}[] {
  const engines: Record<string, { status: StageStatus; detail: string; providerStatus: string }> = {}
  for (const evt of events) {
    if (!evt.stage.startsWith('search:')) continue
    const name = evt.stage.slice(7)
    if (name === 'platform') continue // platform tallies are not providers
    if (name.startsWith('variant:')) continue // extra search-variant passes get their own section
    if (!engines[name]) engines[name] = { status: 'idle', detail: '', providerStatus: '' }
    const e = engines[name]
    if (evt.status === 'start') e.status = 'running'
    else if (evt.status === 'ok') e.status = 'ok'
    else if (evt.status === 'fail') e.status = 'fail'
    if (evt.detail) {
      e.detail = evt.detail
      const head = evt.detail.split(/[:·]/)[0].trim()
      if (/^[A-Z_]+$/.test(head)) {
        e.providerStatus = head
        if (SOFT_PROVIDER_STATES.includes(head)) e.status = 'skip'
      }
    }
  }
  return Object.entries(engines).map(([name, v]) => ({ name, ...v }))
}

/**
 * Search-variant chips — separate from engine chips since a variant pass is a
 * supplementary search over a crop of the same photo, not another provider.
 * See `search/variants.py`.
 */
function variantChips(events: SSEEvent[]): { name: string; status: StageStatus; detail: string }[] {
  const variants: Record<string, { status: StageStatus; detail: string }> = {}
  for (const evt of events) {
    if (!evt.stage.startsWith('search:variant:')) continue
    const rest = evt.stage.slice('search:variant:'.length)
    const variantType = rest.split(':')[0]
    if (!variants[variantType]) variants[variantType] = { status: 'idle', detail: '' }
    const v = variants[variantType]
    if (evt.status === 'start') v.status = 'running'
    else if (evt.status === 'ok') v.status = 'ok'
    else if (evt.status === 'fail') v.status = 'fail'
    if (evt.detail) v.detail = evt.detail
  }
  return Object.entries(variants).map(([name, v]) => ({ name, ...v }))
}

function platformTallies(events: SSEEvent[]): string[] {
  // Last snapshot wins; the backend emits one event per platform after search.
  const seen = new Map<string, string>()
  for (const evt of events) {
    if (evt.stage !== 'search:platform' || !evt.detail) continue
    const [name] = evt.detail.split(':')
    seen.set(name.trim(), evt.detail)
  }
  return [...seen.values()]
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
  onEvent?: (e: SSEEvent) => void
  onDone?: (result: CaseResult) => void
  onFailed?: () => void
}

export default function ProgressView({ caseId, events, onEvent, onDone, onFailed }: Props) {
  const logRef = useRef<HTMLDivElement>(null)
  const closerRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (!onEvent || !onDone || !onFailed) {
      return () => {}
    }

    // No "already subscribed" guard here, deliberately.
    //
    // React 18 StrictMode runs an effect mount → cleanup → mount. A guard that
    // returned early on the second mount left the EventSource closed by the
    // first cleanup and never reopened — the scan ran to completion on the
    // server while the UI sat on "Scanning…" forever with every stage idle.
    // Subscribing on every run and tearing down in the cleanup is the correct
    // shape; the per-run `dead` flag stops a stale run from touching state.
    let dead = false

    const close = api.subscribeEvents(
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

    closerRef.current = close
    return () => {
      dead = true
      close()
    }
  }, [caseId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [events.length])

  const stageMap = buildStageMap(events)
  const chips = engineChips(events)
  const variantChipList = variantChips(events)
  const platforms = platformTallies(events)
  const candidates = candidateLines(events)
  const isDone = events.some((e) => e.stage === 'done' || e.stage === 'error')
  const failedEvent = events.find((e) => e.stage === 'error')
  const eventCount = events.length

  return (
    <div className="max-w-3xl mx-auto" aria-live="polite" aria-label="Scan progress">
      <div className="flex items-center justify-between mb-1">
        <h1 className="text-2xl font-bold text-accent">
          {isDone ? 'Scan complete' : 'Scanning…'}
        </h1>
        {isDone && !failedEvent && <span className="text-success text-sm font-mono">● Done</span>}
        {failedEvent && <span className="text-danger text-sm font-mono">● Stopped</span>}
        {!isDone && eventCount > 0 && (
          <span className="text-muted text-xs font-mono animate-pulse">
            {eventCount} event{eventCount !== 1 ? 's' : ''} received
          </span>
        )}
        {!isDone && eventCount === 0 && (
          <span className="text-warn text-xs font-mono animate-pulse">waiting for pipeline…</span>
        )}
      </div>
      <p className="text-muted text-sm mb-6">
        Case <span className="font-mono text-gray-300">{caseId}</span>
      </p>

      {/* A terminal error is stated here rather than left implicit: the stream
          always reaches an end state, and the reason for it is shown. */}
      {failedEvent && (
        <div role="alert" className="mb-6 border border-danger/50 bg-red-900/10 rounded px-4 py-3 text-sm text-danger">
          {failedEvent.detail || 'The scan stopped before producing a result.'}
        </div>
      )}

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
                    chip.status === 'skip' ? 'border-warn/60 text-warn bg-yellow-900/10' :
                    chip.status === 'fail' ? 'border-danger/60 text-danger bg-red-900/10' :
                    chip.status === 'running' ? 'border-accent/60 text-accent bg-blue-900/10 animate-pulse' :
                    'border-border text-muted'}`}
                aria-label={`${chip.name}: ${chip.providerStatus || chip.status}`}
              >
                {chip.name} · {chip.providerStatus || chip.status}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Search-variant chips: extra crops re-searched beyond the original
          upload, budgeted per scan depth — see search/variants.py. */}
      {variantChipList.length > 0 && (
        <section className="mb-6" aria-labelledby="variant-status-heading">
          <h2 id="variant-status-heading" className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">
            Search Variants
          </h2>
          <div className="flex flex-wrap gap-2" role="list" aria-label="Search variant status chips">
            {variantChipList.map((v) => (
              <div
                key={v.name}
                role="listitem"
                title={v.detail}
                className={`px-3 py-1 rounded-full text-xs font-mono border
                  ${v.status === 'ok' ? 'border-success/60 text-success bg-green-900/10' :
                    v.status === 'fail' ? 'border-warn/60 text-warn bg-yellow-900/10' :
                    v.status === 'running' ? 'border-accent/60 text-accent bg-blue-900/10 animate-pulse' :
                    'border-border text-muted'}`}
              >
                {v.name}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Candidates discovered per platform, priority platforms included even
          at zero, so "looked and found none" reads differently from "never
          looked". */}
      {platforms.length > 0 && (
        <section className="mb-6" aria-labelledby="platform-heading">
          <h2 id="platform-heading" className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">
            Discovery by platform
          </h2>
          <div className="flex flex-wrap gap-2" data-testid="progress-platforms">
            {platforms.map((line) => (
              <span key={line} className="px-3 py-1 rounded-full text-xs font-mono border border-border text-gray-300">
                {line}
              </span>
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
              <div
                key={i}
                className={`truncate ${line.includes('VERIFIED') ? 'text-success' : 'text-gray-400'}`}
                title={line}
              >
                {line}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
