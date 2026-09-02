import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import ProgressView from '../components/ProgressView'
import type { SSEEvent } from '../types/api'

// Mock the API client
vi.mock('../api/client', () => ({
  api: {
    subscribeEvents: vi.fn().mockReturnValue(() => {}),
    getStatus: vi.fn().mockResolvedValue({ status: 'running', event_count: 0, error: null, case_id: 'x' }),
    getResult: vi.fn().mockResolvedValue({ verdict: 'UNVERIFIED', verification: [] }),
  },
}))

function makeEvt(stage: string, status: string, detail: string): SSEEvent {
  return { stage, status, detail, ts: new Date().toISOString() }
}

describe('ProgressView', () => {
  let onEvent: ReturnType<typeof vi.fn>
  let onDone: ReturnType<typeof vi.fn>
  let onFailed: ReturnType<typeof vi.fn>

  beforeEach(() => {
    onEvent = vi.fn()
    onDone = vi.fn()
    onFailed = vi.fn()
    vi.clearAllMocks()
  })

  it('renders all 7 stage labels', () => {
    render(
      <ProgressView
        caseId="case_test"
        events={[]}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    expect(screen.getByText(/\[01\].*Input/)).toBeInTheDocument()
    expect(screen.getByText(/\[02\].*Face/)).toBeInTheDocument()
    expect(screen.getByText(/\[03\].*Reverse/)).toBeInTheDocument()
    expect(screen.getByText(/\[04\].*Candidate/)).toBeInTheDocument()
    expect(screen.getByText(/\[05\].*Evidence/)).toBeInTheDocument()
    expect(screen.getByText(/\[06\].*Blockchain/)).toBeInTheDocument()
    expect(screen.getByText(/\[07\].*read-back/i)).toBeInTheDocument()
  })

  it('shows running indicator for active stage', () => {
    const events = [makeEvt('face', 'start', '')]
    render(
      <ProgressView
        caseId="case_test"
        events={events}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    expect(screen.getByText('running')).toBeInTheDocument()
  })

  it('shows ✓ for completed stages', () => {
    const events = [
      makeEvt('input', 'start', ''),
      makeEvt('input', 'ok', '640x480, sha256 abc…'),
    ]
    render(
      <ProgressView
        caseId="case_test"
        events={events}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    // ✓ icon appears for completed stage
    expect(screen.getAllByText('✓').length).toBeGreaterThan(0)
  })

  it('shows ✗ for failed stages', () => {
    const events = [
      makeEvt('search', 'start', ''),
      makeEvt('search', 'fail', 'no results'),
    ]
    render(
      <ProgressView
        caseId="case_test"
        events={events}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    expect(screen.getAllByText('✗').length).toBeGreaterThan(0)
    expect(screen.getByText('no results')).toBeInTheDocument()
  })

  it('renders engine chips from search:* events', () => {
    const events = [
      makeEvt('search:yandex', 'start', ''),
      makeEvt('search:yandex', 'ok', '60 candidates'),
      makeEvt('search:bing', 'fail', 'blocked'),
    ]
    render(
      <ProgressView
        caseId="case_test"
        events={events}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    expect(screen.getByLabelText('yandex: ok')).toBeInTheDocument()
    expect(screen.getByLabelText('bing: fail')).toBeInTheDocument()
  })

  it('renders candidate log lines', () => {
    const events = [
      makeEvt('verify:candidate', 'ok', 'instagram.com img 0.95 face 0.92 score 0.88 VERIFIED'),
    ]
    render(
      <ProgressView
        caseId="case_test"
        events={events}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    // The candidate line is split across nodes — use getAllByText with partial match
    expect(screen.getAllByTitle('instagram.com img 0.95 face 0.92 score 0.88 VERIFIED').length).toBeGreaterThan(0)
  })

  it('does not leak secrets in rendered events', () => {
    const events = [
      makeEvt('chain', 'ok', 'attester 0xABCDEF12 balance 0.05 ETH'),
    ]
    const { container } = render(
      <ProgressView
        caseId="case_test"
        events={events}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    // Private keys (64-char hex) must never appear
    expect(container.innerHTML).not.toMatch(/0x[0-9a-fA-F]{64}/)
  })

  it('SSE subscription is set up on mount', async () => {
    const { api } = await import('../api/client')
    render(
      <ProgressView
        caseId="case_xyz"
        events={[]}
        onEvent={onEvent}
        onDone={onDone}
        onFailed={onFailed}
      />,
    )
    expect(vi.mocked(api.subscribeEvents)).toHaveBeenCalledWith(
      'case_xyz',
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    )
  })

  // ---- SSE subscription lifecycle --------------------------------------

  it('re-subscribes after a StrictMode-style mount/unmount/mount', async () => {
    // The bug this pins: an "already subscribed" ref guard meant that React
    // 18 StrictMode's mount → cleanup → mount left the EventSource closed by
    // the first cleanup and never reopened. The scan completed on the server
    // while the UI sat on "Scanning…" with every stage idle — the exact state
    // the pipeline is supposed to make impossible.
    const { api } = await import('../api/client')
    const closers: ReturnType<typeof vi.fn>[] = []
    vi.mocked(api.subscribeEvents).mockImplementation(() => {
      const close = vi.fn()
      closers.push(close)
      return close
    })

    const { StrictMode } = await import('react')
    render(
      <StrictMode>
        <ProgressView caseId="case_strict" events={[]} onEvent={onEvent}
          onDone={onDone} onFailed={onFailed} />
      </StrictMode>,
    )

    // Two subscribes and exactly one teardown leaves one live subscription.
    expect(vi.mocked(api.subscribeEvents).mock.calls.length).toBeGreaterThanOrEqual(2)
    const closed = closers.filter((c) => c.mock.calls.length > 0).length
    expect(closers.length - closed).toBe(1)
  })

  it('delivers events received after the remount to onEvent', async () => {
    const { api } = await import('../api/client')
    let deliver: ((e: SSEEvent) => void) | null = null
    vi.mocked(api.subscribeEvents).mockImplementation((_id, onEvt) => {
      deliver = onEvt
      return () => { deliver = null }
    })

    const { StrictMode } = await import('react')
    render(
      <StrictMode>
        <ProgressView caseId="case_strict2" events={[]} onEvent={onEvent}
          onDone={onDone} onFailed={onFailed} />
      </StrictMode>,
    )

    expect(deliver).not.toBeNull()
    act(() => { deliver!(makeEvt('input', 'ok', '1564x2000')) })
    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ stage: 'input' }))
  })

  it('closes the stream when the case changes', async () => {
    const { api } = await import('../api/client')
    const closers: ReturnType<typeof vi.fn>[] = []
    vi.mocked(api.subscribeEvents).mockImplementation(() => {
      const close = vi.fn()
      closers.push(close)
      return close
    })

    const { rerender } = render(
      <ProgressView caseId="case_a" events={[]} onEvent={onEvent} onDone={onDone} onFailed={onFailed} />,
    )
    rerender(
      <ProgressView caseId="case_b" events={[]} onEvent={onEvent} onDone={onDone} onFailed={onFailed} />,
    )
    expect(closers[0]).toHaveBeenCalled()   // the old case's stream is closed
    expect(closers.length).toBe(2)          // and the new case has its own
  })

  it('shows a terminal error rather than staying on Scanning forever', () => {
    render(
      <ProgressView
        caseId="case_err"
        events={[makeEvt('error', 'fail', 'scan exceeded the 600s deadline')]}
        onEvent={onEvent} onDone={onDone} onFailed={onFailed} />,
    )
    expect(screen.getByRole('alert').textContent).toMatch(/exceeded the 600s deadline/)
    expect(screen.getByText(/● Stopped/)).toBeInTheDocument()
  })

  it('shows a challenged provider as a warning, not a scan failure', () => {
    render(
      <ProgressView
        caseId="case_chal"
        events={[
          makeEvt('search:bing', 'fail', 'CHALLENGED: engine refused to process the image'),
          makeEvt('search:yandex', 'ok', 'COMPLETED: 60 candidates'),
        ]}
        onEvent={onEvent} onDone={onDone} onFailed={onFailed} />,
    )
    const chips = screen.getByLabelText('Engine status chips')
    expect(chips.textContent).toMatch(/bing · CHALLENGED/)
    expect(chips.textContent).toMatch(/yandex · COMPLETED/)
    // A challenged provider is not a terminal error for the scan.
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('renders per-platform discovery tallies including zeros', () => {
    render(
      <ProgressView
        caseId="case_plat"
        events={[
          makeEvt('search:platform', 'info', 'LinkedIn: 0'),
          makeEvt('search:platform', 'info', 'GitHub: 3'),
          makeEvt('search:platform', 'info', 'Other Web: 56'),
        ]}
        onEvent={onEvent} onDone={onDone} onFailed={onFailed} />,
    )
    const tallies = screen.getByTestId('progress-platforms')
    expect(tallies.textContent).toMatch(/LinkedIn: 0/)
    expect(tallies.textContent).toMatch(/GitHub: 3/)
    // The platform tally must not be mistaken for a provider.
    expect(screen.queryByText(/platform · /)).toBeNull()
  })
})
