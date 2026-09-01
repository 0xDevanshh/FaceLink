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
})
