import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import EvidenceView from '../components/EvidenceView'
import type { CaseResult } from '../types/api'

// Mock API — no real backend
vi.mock('../api/client', () => ({
  api: {
    getEvidenceUrl: vi.fn().mockReturnValue('/api/v1/scan/case_test/evidence'),
    verifyEvidence: vi.fn().mockResolvedValue({
      overall: 'PASS',
      checks: [
        { check: 'bundle integrity', passed: true },
        { check: 'evidenceHash recomputed', passed: true, detail: 'sha256:abc123…' },
        { check: 'matched_url hash consistent', passed: true },
      ],
    }),
  },
}))

const MOCK_RESULT: CaseResult = {
  case_id: 'case_20260901_000000',
  pipeline_version: '1.0.0',
  created_at: '2026-09-01T00:00:00Z',
  verdict: 'VERIFIED',
  failure_reason: null,
  evidence_sha256: 'aabbccdd' + 'ee'.repeat(28),
  face: {
    detected: true, backend: 'insightface', model: 'buffalo_l/SCRFD+ArcFace',
    faces_found: 1, bbox: [10, 20, 110, 140], det_score: 0.89,
    embedding_dimension: 512, embedding_sha256: 'bb'.repeat(32),
  },
  reverse_search: {
    engines_attempted: ['yandex', 'bing'],
    engines_succeeded: ['yandex'],
    engine_errors: { bing: 'blocked' },
    query_mode: { yandex: 'upload' },
    total_candidates: 10,
    social_candidates: 2,
  },
  verification: [],
  best_match: null,
  stages_passed: ['SEARCH_FOUND', 'SOCIAL_MATCH'],
  blockchain: null,
}

describe('EvidenceView', () => {
  beforeEach(() => vi.clearAllMocks())

  it('renders case ID and evidence hash', () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    expect(screen.getByText(/evidence\/case_20260901_000000/)).toBeInTheDocument()
    expect(screen.getByText(/aabbccdd/)).toBeInTheDocument()
  })

  it('renders download link with correct href', () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    const links = screen.getAllByRole('link', { name: /download/i })
    expect(links.length).toBeGreaterThan(0)
    expect(links[0]).toHaveAttribute('href', '/api/v1/scan/case_test/evidence')
  })

  it('renders file tree with expected file names', () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    // Use getAllByText and check at least one exists for repeated names
    expect(screen.getAllByText('case.json').length).toBeGreaterThan(0)
    expect(screen.getAllByText('attested_payload.json').length).toBeGreaterThan(0)
    expect(screen.getAllByText('blockchain.json').length).toBeGreaterThan(0)
  })

  it('expands file content on click', async () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    const caseBtn = screen.getByRole('button', { name: /case\.json/i })
    fireEvent.click(caseBtn)
    await waitFor(() => {
      // Should show the JSON content of case.json
      expect(screen.getByText(/VERIFIED/)).toBeInTheDocument()
    })
  })

  it('collapses file content on second click', async () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    const caseBtn = screen.getByRole('button', { name: /case\.json/i })
    fireEvent.click(caseBtn)  // expand
    fireEvent.click(caseBtn)  // collapse
    await waitFor(() => {
      expect(caseBtn).toHaveAttribute('aria-expanded', 'false')
    })
  })

  it('shows verify section with upload button', () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    // Button text is "Verify Evidence (.zip)" — match on partial text
    const btns = screen.getAllByRole('button')
    const verifyBtn = btns.find(b => b.textContent?.toLowerCase().includes('verify'))
    expect(verifyBtn).toBeTruthy()
  })

  it('calls verifyEvidence API and shows PASS result', async () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    const { api } = await import('../api/client')

    // Simulate file input change
    const zipBytes = new Uint8Array([0x50, 0x4b, 0x03, 0x04]) // ZIP magic
    const zipFile = new File([zipBytes], 'evidence.zip', { type: 'application/zip' })
    const input = document.querySelector('input[type=file][accept=".zip"]') as HTMLInputElement
    Object.defineProperty(input, 'files', { value: [zipFile] })
    fireEvent.change(input)

    await waitFor(() => {
      expect(api.verifyEvidence).toHaveBeenCalledWith(zipFile)
      expect(screen.getByText('Overall: PASS')).toBeInTheDocument()
    })
  })

  it('shows FAIL overall when verify fails', async () => {
    const { api } = await import('../api/client')
    vi.mocked(api.verifyEvidence).mockResolvedValueOnce({
      overall: 'FAIL',
      checks: [
        { check: 'evidenceHash recomputed', passed: false, detail: 'hash mismatch' },
      ],
    })

    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    const zipFile = new File([new Uint8Array(4)], 'e.zip', { type: 'application/zip' })
    const input = document.querySelector('input[type=file][accept=".zip"]') as HTMLInputElement
    Object.defineProperty(input, 'files', { value: [zipFile] })
    fireEvent.change(input)

    await waitFor(() => expect(screen.getByText('Overall: FAIL')).toBeInTheDocument())
    expect(screen.getByText(/hash mismatch/)).toBeInTheDocument()
  })

  it('shows tamper-demo section', () => {
    render(<EvidenceView caseId="case_20260901_000000" result={MOCK_RESULT} />)
    expect(screen.getByText(/Tamper-Evidence Demo/i)).toBeInTheDocument()
    expect(screen.getByText(/server-side evidence is never modified/i)).toBeInTheDocument()
  })
})
