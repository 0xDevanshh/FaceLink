import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import CandidateCard from '../components/CandidateCard'
import type { VerifiedCandidate } from '../types/api'

function makeCandidate(overrides: Partial<VerifiedCandidate> = {}): VerifiedCandidate {
  return {
    engine: 'yandex',
    url: 'https://instagram.com/p/ABC123/',
    domain: 'instagram.com',
    platform: 'Instagram',
    is_social: true,
    canonical_url: 'https://instagram.com/p/ABC/',
    platform_priority: 2,
    candidate_type: 'EXACT_IMAGE',
    fetched: true,
    fetch_note: 'HTTP 200, 3 image refs',
    candidate_image_url: 'https://cdn.example/a.jpg',
    candidate_image_source: 'og:image',
    candidate_image_sha256: 'dd'.repeat(32),
    candidate_image_phash: 'fafc13e1a083a6d8',
    candidate_faces_found: 1,
    candidate_face_index: 0,
    candidate_face_quality: 0.85,
    candidate_face_bands: {},
    found_via_variant: '',
    image_similarity: 0.95,
    face_detected: true,
    face_similarity: 0.92,
    metadata_consistency: 0.7,
    confidence_band: 'STRONG',
    stages: ['SEARCH_FOUND', 'SOCIAL_MATCH', 'IMAGE_MATCH', 'FACE_MATCH', 'VERIFIED'],
    match_type: 'exact-image',
    final_score: 0.88,
    verified: true,
    rejection_reason: '',
    rank_explanation: '',
    ...overrides,
  }
}

describe('CandidateCard', () => {
  it('renders VERIFIED badge for verified candidate', () => {
    render(<CandidateCard candidate={makeCandidate()} />)
    // The badge specifically (not the ladder rung)
    expect(screen.getByRole('article')).toBeInTheDocument()
    const badge = screen.getAllByText('VERIFIED').find(
      el => el.className.includes('rounded-full')
    )
    expect(badge).toBeInTheDocument()
  })

  it('renders REJECTED badge for rejected candidate', () => {
    render(<CandidateCard candidate={makeCandidate({
      verified: false,
      stages: ['SEARCH_FOUND'],
      confidence_band: 'INSUFFICIENT',
      rejection_reason: 'face similarity 0.10 below threshold 0.38',
    })} />)
    expect(screen.getByText('REJECTED')).toBeInTheDocument()
  })

  it('shows rejection reason for rejected candidate', () => {
    render(<CandidateCard candidate={makeCandidate({
      verified: false,
      stages: ['SEARCH_FOUND'],
      confidence_band: 'INSUFFICIENT',
      rejection_reason: 'face similarity 0.10 below threshold 0.38',
    })} />)
    expect(screen.getByText(/face similarity.*below threshold/i)).toBeInTheDocument()
  })

  it('shows the rank explanation for a verified candidate', () => {
    render(<CandidateCard candidate={makeCandidate({
      rank_explanation: 'Strong face match (0.920); on the exact retrieved image; '
        + 'found on Instagram (a priority platform).',
    })} />)
    expect(screen.getByText(/Strong face match \(0\.920\)/)).toBeInTheDocument()
  })

  it('renders nothing extra when rank_explanation is absent (older evidence bundles)', () => {
    render(<CandidateCard candidate={makeCandidate({ rank_explanation: '' })} />)
    expect(screen.queryByText(/Strong face match/)).not.toBeInTheDocument()
  })

  it('shows STRONG confidence band', () => {
    render(<CandidateCard candidate={makeCandidate({ confidence_band: 'STRONG' })} />)
    expect(screen.getByText('STRONG')).toBeInTheDocument()
  })

  it('uses the recorded threshold snapshot instead of a hardcoded value', () => {
    // Regression: the score bar used to hard-code 0.38/0.80 regardless of
    // what the scan was actually configured with.
    render(
      <CandidateCard
        candidate={makeCandidate({ face_similarity: 0.5 })}
        thresholds={{
          face_match_threshold: 0.6, image_match_threshold: 0.9, verify_min_score: 0.7,
          weight_face: 0.5, weight_image: 0.4, weight_meta: 0.1,
          insightface_model: 'buffalo_l', face_backend: 'insightface',
          calibration_status: 'DEFAULT', calibration_note: '',
        }}
      />
    )
    // The marker for a 0.6 threshold sits at 60%, not the old hard-coded 38%.
    expect(screen.getByTestId('threshold-marker-Face sim')).toHaveStyle({ left: '60%' })
    expect(screen.getByTestId('threshold-marker-Image sim')).toHaveStyle({ left: '90%' })
  })

  it('falls back to the documented default thresholds when no snapshot is given', () => {
    render(<CandidateCard candidate={makeCandidate()} />)
    expect(screen.getByTestId('threshold-marker-Face sim')).toHaveStyle({ left: '38%' })
    expect(screen.getByTestId('threshold-marker-Image sim')).toHaveStyle({ left: '80%' })
  })

  it('shows which candidate face matched when the candidate has multiple faces', () => {
    render(<CandidateCard candidate={makeCandidate({
      candidate_faces_found: 3, candidate_face_index: 1, candidate_face_quality: 0.72,
    })} />)
    expect(screen.getByText(/matched face #2/i)).toBeInTheDocument()
    expect(screen.getByText('72%')).toBeInTheDocument()
  })

  it('renders Instagram platform icon', () => {
    const { container } = render(<CandidateCard candidate={makeCandidate()} />)
    expect(container.textContent).toContain('📸')
  })

  it('shows the candidate domain as a link', () => {
    render(<CandidateCard candidate={makeCandidate()} />)
    const link = screen.getByRole('link', { name: /instagram\.com/i })
    expect(link).toHaveAttribute('href', 'https://instagram.com/p/ABC123/')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    expect(link).toHaveAttribute('target', '_blank')
  })

  it('displays all ladder rungs', () => {
    render(<CandidateCard candidate={makeCandidate()} />)
    // Ladder rung titles are unique — use title attribute
    expect(screen.getByTitle('SEARCH_FOUND')).toBeInTheDocument()
    expect(screen.getByTitle('FACE_MATCH')).toBeInTheDocument()
    expect(screen.getByTitle('VERIFIED')).toBeInTheDocument()
  })

  it('does not show IMAGE_MATCH rung when not in stages', () => {
    const cand = makeCandidate({
      stages: ['SEARCH_FOUND', 'SOCIAL_MATCH', 'FACE_MATCH', 'VERIFIED'],
      match_type: 'face-only',
    })
    render(<CandidateCard candidate={cand} />)
    // IMAGE MATCH rung should be present but styled as unreached (opacity-40)
    const imageMatchEl = screen.getByTitle('IMAGE_MATCH')
    expect(imageMatchEl.className).toContain('opacity-40')
  })

  it('has progress bars with aria-valuenow', () => {
    render(<CandidateCard candidate={makeCandidate()} />)
    const bars = screen.getAllByRole('progressbar')
    expect(bars.length).toBeGreaterThan(0)
    bars.forEach((bar) => {
      expect(bar).toHaveAttribute('aria-valuenow')
      expect(bar).toHaveAttribute('aria-valuemin', '0')
      expect(bar).toHaveAttribute('aria-valuemax', '100')
    })
  })

  it('renders article with correct aria-label', () => {
    render(<CandidateCard candidate={makeCandidate()} />)
    expect(screen.getByRole('article')).toHaveAttribute(
      'aria-label',
      'Candidate: instagram.com — VERIFIED'
    )
  })
})
