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
    candidate_face_index: 0,
    candidate_face_quality: 0.85,
    candidate_face_bands: {},
    found_via_variant: '',
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
    rank_explanation: '',
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
    variants: [],
  },
  verification: [makeCandidate(true)],
  best_match: makeCandidate(true),
  stages_passed: ['SEARCH_FOUND', 'SOCIAL_MATCH', 'IMAGE_MATCH', 'FACE_MATCH', 'VERIFIED'],
  evidence_graph: null,
  threshold_snapshot: null,
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

  it('shows the recorded threshold snapshot and calibration status', () => {
    const result = {
      ...BASE_RESULT,
      threshold_snapshot: {
        face_match_threshold: 0.38, image_match_threshold: 0.80, verify_min_score: 0.70,
        weight_face: 0.5, weight_image: 0.4, weight_meta: 0.1,
        insightface_model: 'buffalo_l', face_backend: 'insightface',
        calibration_status: 'CALIBRATION_INSUFFICIENT',
        calibration_note: 'Only 1 genuine and 1 impostor pair supplied.',
      },
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText('CALIBRATION_INSUFFICIENT')).toBeInTheDocument()
    expect(screen.getByText(/Only 1 genuine and 1 impostor pair/)).toBeInTheDocument()
    // The footer must use the real recorded threshold, not a hard-coded 38%.
    expect(screen.getByText(/threshold 38%/)).toBeInTheDocument()
  })

  it('falls back to 38% in the footer when no threshold snapshot was recorded', () => {
    render(<ResultView scan={makeScan(BASE_RESULT)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText(/threshold 38%/)).toBeInTheDocument()
  })

  it('shows the independent evidence count from the evidence graph', () => {
    const result = {
      ...BASE_RESULT,
      evidence_graph: {
        nodes: [
          { id: 'image:0', type: 'image', label: 'image cluster (1 URL(s))' },
          { id: 'domain:linkedin.com', type: 'domain', label: 'linkedin.com' },
          { id: 'image:1', type: 'image', label: 'image cluster (1 URL(s))' },
          { id: 'domain:github.com', type: 'domain', label: 'github.com' },
        ],
        edges: [],
        independent_evidence_count: 2,
      },
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText('Independent evidence sources:').nextElementSibling).toHaveTextContent('2')
    expect(screen.getByText(/2 distinct image\(s\) across 2 domain\(s\)/)).toBeInTheDocument()
  })

  it('shows search variant chips with their candidate counts', () => {
    const result = {
      ...BASE_RESULT,
      reverse_search: {
        ...BASE_RESULT.reverse_search!,
        variants: [
          { variant_id: 'v0-original', variant_type: 'original', sha256: '', width: 0, height: 0,
            candidates_found: 0, skipped: false, skip_reason: '' },
          { variant_id: 'v1-tight_crop', variant_type: 'tight_crop', sha256: 'aa', width: 400, height: 400,
            candidates_found: 3, skipped: false, skip_reason: '' },
        ],
      },
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    const section = screen.getByTestId('search-variants')
    expect(section.textContent).toMatch(/tight_crop/)
    expect(section.textContent).toMatch(/\+3/)
  })

  it('marks a skipped search variant distinctly from one that ran', () => {
    const result = {
      ...BASE_RESULT,
      reverse_search: {
        ...BASE_RESULT.reverse_search!,
        variants: [
          { variant_id: 'v1-loose_crop', variant_type: 'loose_crop', sha256: 'bb', width: 500, height: 500,
            candidates_found: 0, skipped: true, skip_reason: 'overall search budget exhausted' },
        ],
      },
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText(/loose_crop \(skipped\)/)).toBeInTheDocument()
  })

  it('shows graded face quality bands', () => {
    const result = {
      ...BASE_RESULT,
      face: {
        ...BASE_RESULT.face!,
        quality: {
          passed: true, error: null, detail: '', blur_score: 120, face_px: 200, face_count: 1,
          det_score: 0.9, yaw_deg: 2.1, roll_deg: 1.0, brightness: 160,
          bands: { detection: 'PASS', resolution: 'GOOD', blur: 'GOOD' },
          overall_quality: 0.91,
        },
      },
    }
    render(<ResultView scan={makeScan(result)} onViewEvidence={vi.fn()} onNewScan={vi.fn()} />)
    expect(screen.getByText('Overall quality:').nextElementSibling).toHaveTextContent('91%')
    expect(screen.getAllByText('GOOD', { exact: true }).length).toBeGreaterThan(0)
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
