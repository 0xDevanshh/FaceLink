import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ResultView from '../components/ResultView'
import type { ScanState } from '../App'
import type { CaseResult, VerifiedCandidate } from '../types/api'

vi.mock('../api/client', () => ({
  api: { getEvidenceUrl: vi.fn().mockReturnValue('/api/v1/scan/x/evidence') },
}))

function makeCandidate(verified: boolean, overrides: Partial<VerifiedCandidate> = {}): VerifiedCandidate {
  return {
    engine: 'yandex',
    url: 'https://instagram.com/p/ABC/',
    domain: 'instagram.com',
    platform: 'Instagram',
    is_social: true,
    canonical_url: 'https://instagram.com/p/ABC/',
    platform_priority: 2,
    candidate_type: verified ? 'EXACT_IMAGE' : 'SOCIAL_POST',
    fetched: true,
    fetch_note: '',
    candidate_image_url: null,
    candidate_image_source: 'og:image',
    candidate_image_sha256: 'dd'.repeat(32),
    candidate_image_phash: 'aabb',
    candidate_faces_found: 1,
    image_similarity: 0.95,
    face_detected: true,
    face_similarity: verified ? 0.92 : 0.10,
    metadata_consistency: 0.7,
    confidence_band: verified ? 'STRONG' : 'INSUFFICIENT',
    stages: verified
      ? ['SEARCH_FOUND', 'SOCIAL_MATCH', 'IMAGE_MATCH', 'FACE_MATCH', 'VERIFIED']
      : ['SEARCH_FOUND'],
    match_type: verified ? 'exact-image' : 'none',
    final_score: verified ? 0.88 : 0.12,
    verified,
    rejection_reason: verified ? '' : 'face similarity 0.10 below threshold 0.38',
    ...overrides,
  }
}

function makeScan(result: CaseResult | null, failed = false): ScanState {
  return {
    caseId: 'case_20260901_000000',
    events: [],
    result,
    done: true,
    failed,
  }
}

const BASE_RESULT: CaseResult = {
  case_id: 'case_20260901_000000',
  pipeline_version: '1.0.0',
  created_at: '2026-09-01T00:00:00Z',
  verdict: 'VERIFIED',
  failure_reason: null,
  evidence_sha256: 'aabb'.repeat(16),
  face: {
    detected: true, backend: 'insightface', model: 'buffalo_l/SCRFD+ArcFace',
    faces_found: 1, bbox: [10, 20, 110, 140], det_score: 0.89,
    embedding_dimension: 512, embedding_sha256: 'bb'.repeat(32),
    faces: [], quality: null,
  },
  face_selection: {
    mode: 'auto', face_index: 0, faces_offered: 1, bbox: [10, 20, 110, 140],
    crop_rect: null, crop_sha256: null,
    original_sha256: 'cc'.repeat(32), original_width: 800, original_height: 1200,
    selected_at: '2026-09-01T00:00:00Z',
  },
  reverse_search: {
    engines_attempted: ['yandex'],
    engines_succeeded: ['yandex'],
    engine_errors: {},
    query_mode: { yandex: 'upload' },
    total_candidates: 5,
    social_candidates: 2,
    providers: [
      { engine: 'yandex', status: 'COMPLETED', candidates: 5, duration_s: 12.5, query_mode: 'upload', error: '' },
    ],
    platform_counts: { LinkedIn: 0, Instagram: 2, 'X/Twitter': 0, GitHub: 0, YouTube: 0, 'Other Web': 3 },
    timed_out: false,
  },
  verification: [makeCandidate(true)],
  best_match: makeCandidate(true),
  stages_passed: ['SEARCH_FOUND', 'SOCIAL_MATCH', 'IMAGE_MATCH', 'FACE_MATCH', 'VERIFIED'],
  blockchain: null,
}

describe('ResultView', () => {
  it('renders VERIFIED verdict banner', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    // The h1 verdict banner specifically
    expect(screen.getByRole('heading', { name: 'VERIFIED' })).toBeInTheDocument()
  })

  it('renders UNVERIFIED verdict', () => {
    const result = { ...BASE_RESULT, verdict: 'UNVERIFIED', failure_reason: 'no face match',
                     verification: [makeCandidate(false)], best_match: null }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText('UNVERIFIED')).toBeInTheDocument()
    expect(screen.getByText('no face match')).toBeInTheDocument()
  })

  it('renders pipeline failure state', () => {
    render(<ResultView scan={makeScan(null, true)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText(/Pipeline Failed/i)).toBeInTheDocument()
  })

  it('shows evidence hash with copy button', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    // The evidence hash label span is unique
    expect(screen.getByText('evidenceHash')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument()
  })

  it('shows verified candidate count', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText(/Verified Matches \(1\)/i)).toBeInTheDocument()
  })

  it('shows rejected candidates section collapsed by default', () => {
    const result = {
      ...BASE_RESULT,
      verification: [makeCandidate(true), makeCandidate(false)],
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText(/Rejected candidates \(1\)/i)).toBeInTheDocument()
    // Collapsed by default — rejected card not visible
    expect(screen.getByRole('button', { name: /rejected/i })).toHaveAttribute('aria-expanded', 'false')
  })

  it('expands rejected section on click', () => {
    const result = {
      ...BASE_RESULT,
      verification: [makeCandidate(true), makeCandidate(false, { url: 'https://instagram.com/p/XYZ/' })],
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: /rejected/i }))
    expect(screen.getByRole('button', { name: /rejected/i })).toHaveAttribute('aria-expanded', 'true')
  })

  it('shows each provider with its own terminal status', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    const table = screen.getByTestId('provider-table')
    expect(table.textContent).toMatch(/yandex/)
    expect(table.textContent).toMatch(/COMPLETED/)
    expect(screen.getByText(/Total unique candidates/i)).toBeInTheDocument()
  })

  it('distinguishes a challenged provider from one that found nothing', () => {
    const result = {
      ...BASE_RESULT,
      reverse_search: {
        ...BASE_RESULT.reverse_search!,
        providers: [
          { engine: 'yandex', status: 'COMPLETED' as const, candidates: 5, duration_s: 12.5, query_mode: 'upload', error: '' },
          { engine: 'bing', status: 'CHALLENGED' as const, candidates: 0, duration_s: 11.8, query_mode: 'upload', error: 'engine refused to process the image' },
          { engine: 'google_lens', status: 'NO_RESULTS' as const, candidates: 0, duration_s: 8.0, query_mode: 'upload', error: 'no outbound result links found' },
          { engine: 'serpapi_yandex', status: 'NOT_CONFIGURED' as const, candidates: 0, duration_s: 0, query_mode: 'api', error: 'SERPAPI_KEY not set' },
        ],
      },
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    const table = screen.getByTestId('provider-table')
    for (const status of ['COMPLETED', 'CHALLENGED', 'NO_RESULTS', 'NOT_CONFIGURED']) {
      expect(table.textContent).toContain(status)
    }
  })

  it('reports zero counts for priority platforms that returned nothing', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    // "we looked and found none" must not render the same as "we never looked".
    const counts = screen.getByTestId('platform-counts')
    expect(counts.textContent).toMatch(/LinkedIn 0/)
    expect(counts.textContent).toMatch(/GitHub 0/)
    expect(counts.textContent).toMatch(/Instagram 2/)
  })

  it('calls onViewEvidence when button clicked', () => {
    const onViewEvidence = vi.fn()
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={onViewEvidence} onNewScan={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: /view evidence/i }))
    expect(onViewEvidence).toHaveBeenCalled()
  })

  it('calls onNewScan when button clicked', () => {
    const onNewScan = vi.fn()
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={onNewScan} />)
    fireEvent.click(screen.getByRole('button', { name: /new scan/i }))
    expect(onNewScan).toHaveBeenCalled()
  })

  it('shows scope disclaimer', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText(/not an identity claim/i)).toBeInTheDocument()
  })

  it('shows degraded state when 0 candidates', () => {
    const result = { ...BASE_RESULT, verification: [], best_match: null, verdict: 'UNVERIFIED' }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    // Should show empty/degraded message
    expect(screen.getByText(/No candidates were verified/i)).toBeInTheDocument()
  })

  it('embedding hash is shown with privacy note', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText(/vector stays local/i)).toBeInTheDocument()
  })
})
