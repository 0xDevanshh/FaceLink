import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import ProgressView from '../components/ProgressView'
import type { SSEEvent } from '../types/api'

// ProgressView is now purely display — no SSE logic, no API calls
// All subscription is managed by App

function makeEvt(stage: string, status: string, detail: string): SSEEvent {
  return { stage, status, detail, ts: new Date().toISOString() }
}

describe('ProgressView', () => {
  it('renders all 7 stage labels', () => {
    render(<ProgressView caseId="case_test" events={[]} />)
    expect(screen.getByText(/\[01\].*Input/)).toBeInTheDocument()
    expect(screen.getByText(/\[02\].*Face/)).toBeInTheDocument()
    expect(screen.getByText(/\[03\].*Reverse/)).toBeInTheDocument()
    expect(screen.getByText(/\[04\].*Candidate/)).toBeInTheDocument()
    expect(screen.getByText(/\[05\].*Evidence/)).toBeInTheDocument()
    expect(screen.getByText(/\[06\].*Blockchain/)).toBeInTheDocument()
    expect(screen.getByText(/\[07\].*read-back/i)).toBeInTheDocument()
  })

  it('shows case ID', () => {
    render(<ProgressView caseId="case_20260901_123456" events={[]} />)
    expect(screen.getByText('case_20260901_123456')).toBeInTheDocument()
  })

  it('shows "waiting for pipeline" when no events yet', () => {
    render(<ProgressView caseId="case_test" events={[]} />)
    expect(screen.getByText(/waiting for pipeline/i)).toBeInTheDocument()
  })

  it('shows event count while running', () => {
    const events = [makeEvt('input', 'start', '')]
    render(<ProgressView caseId="case_test" events={events} />)
    expect(screen.getByText(/1 event received/i)).toBeInTheDocument()
  })

  it('shows running indicator for active stage', () => {
    const events = [makeEvt('face', 'start', '')]
    render(<ProgressView caseId="case_test" events={events} />)
    expect(screen.getByText(/running/i)).toBeInTheDocument()
  })

  it('shows ✓ for completed stages', () => {
    const events = [
      makeEvt('input', 'start', ''),
      makeEvt('input', 'ok', '640x480, sha256 abc…'),
    ]
    render(<ProgressView caseId="case_test" events={events} />)
    expect(screen.getAllByText('✓').length).toBeGreaterThan(0)
  })

  it('shows ✗ for failed stages', () => {
    const events = [
      makeEvt('search', 'start', ''),
      makeEvt('search', 'fail', 'no results'),
    ]
    render(<ProgressView caseId="case_test" events={events} />)
    expect(screen.getAllByText('✗').length).toBeGreaterThan(0)
    expect(screen.getByText(/no results/)).toBeInTheDocument()
  })

  it('renders engine chips from search:* events', () => {
    const events = [
      makeEvt('search:yandex', 'start', ''),
      makeEvt('search:yandex', 'ok', '60 candidates'),
      makeEvt('search:bing', 'fail', 'blocked'),
    ]
    render(<ProgressView caseId="case_test" events={events} />)
    expect(screen.getByLabelText('yandex: ok')).toBeInTheDocument()
    expect(screen.getByLabelText('bing: fail')).toBeInTheDocument()
  })

  it('renders candidate log with title attribute', () => {
    const events = [
      makeEvt('verify:candidate', 'ok', 'instagram.com img 0.95 face 0.92 score 0.88 VERIFIED'),
    ]
    render(<ProgressView caseId="case_test" events={events} />)
    expect(
      screen.getAllByTitle('instagram.com img 0.95 face 0.92 score 0.88 VERIFIED').length
    ).toBeGreaterThan(0)
  })

  it('shows "Scan complete" and Done badge when done event received', () => {
    const events = [
      makeEvt('input', 'ok', ''),
      makeEvt('done', 'ok', 'verdict=UNVERIFIED'),
    ]
    render(<ProgressView caseId="case_test" events={events} />)
    expect(screen.getByText(/Scan complete/i)).toBeInTheDocument()
    expect(screen.getByText(/Done/i)).toBeInTheDocument()
  })

  it('does not leak private key patterns in rendered output', () => {
    const events = [
      makeEvt('chain', 'ok', 'attester 0xABCDEF12 balance 0.05 ETH'),
    ]
    const { container } = render(<ProgressView caseId="case_test" events={events} />)
    // 64-char hex private key must never appear
    expect(container.innerHTML).not.toMatch(/0x[0-9a-fA-F]{64}/)
  })

  it('highlights VERIFIED candidates in green', () => {
    const events = [
      makeEvt('verify:candidate', 'ok', 'youtube.com img 0.75 face 0.92 VERIFIED'),
      makeEvt('verify:candidate', 'info', 'instagram.com img 0.30 face 0.10 none'),
    ]
    const { container } = render(<ProgressView caseId="case_test" events={events} />)
    const divs = container.querySelectorAll('[aria-label="Candidate verification log"] > div')
    expect(divs[0].className).toContain('text-success')
    expect(divs[1].className).toContain('text-gray-400')
  })

  it('detail lines are truncated with title attribute', () => {
    const longDetail = 'x'.repeat(200)
    const events = [makeEvt('input', 'ok', longDetail)]
    render(<ProgressView caseId="case_test" events={events} />)
    // The title carries the full text (with prefix character)
    const el = document.querySelector(`[title*="${longDetail.slice(0, 20)}"]`)
    expect(el).toBeTruthy()
  })
})
