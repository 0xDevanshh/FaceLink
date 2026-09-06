import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import IdentitySection from '../components/IdentitySection'
import type { CaseResult, CelebrityInfo, IdentityResult, SocialAccount } from '../types/api'

function makeIdentity(overrides: Partial<IdentityResult> = {}): IdentityResult {
  return {
    person_id: 'Q3503829',
    name: 'Sundar Pichai',
    aliases: [],
    occupation: 'CEO, Google and Alphabet',
    category: 'technology',
    confidence: 0.96,
    level: 'HIGH',
    face_similarity: 0.91,
    margin: 0.3,
    evidence_count: 2,
    supporting_evidence: [
      { kind: 'face_similarity', description: 'face similarity 0.910', weight: 0.6, value: 0.91 },
    ],
    candidate_identities: [],
    ...overrides,
  }
}

function makeAccount(overrides: Partial<SocialAccount> = {}): SocialAccount {
  return {
    platform: 'X/Twitter', username: 'sundarpichai', url: 'https://x.com/sundarpichai',
    display_name: '', profile_image: '', status: 'OFFICIAL', confidence: 0.98,
    source: 'wikidata+this-scan-evidence', evidence: [],
    ...overrides,
  }
}

function makeCelebrity(overrides: Partial<CelebrityInfo> = {}): CelebrityInfo {
  return {
    available: true, name: 'Sundar Pichai', nationality: 'in',
    occupations: ['businessperson', 'executive'], birthday: '1972-07-12',
    age: 54, gender: 'male', height: 1.73, net_worth: 1500000000, is_alive: true,
    source: 'api_ninjas',
    ...overrides,
  }
}

function makeResult(
  identity: IdentityResult | null,
  profiles: SocialAccount[] = [],
  celebrity: CelebrityInfo | null = null,
): CaseResult {
  return {
    case_id: 'case_test', pipeline_version: '1.0.0', created_at: 'now',
    verdict: 'VERIFIED_OFFCHAIN', failure_reason: null, evidence_sha256: null,
    face: null, face_selection: null, reverse_search: null, verification: [],
    best_match: null, evidence_graph: null, threshold_snapshot: null, stages_passed: [],
    blockchain: null, identity, official_profiles: profiles, celebrity,
  }
}

describe('IdentitySection', () => {
  it('renders nothing when identity is absent', () => {
    const { container } = render(<IdentitySection result={makeResult(null)} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when identity has no confident name (LOW/UNKNOWN)', () => {
    const result = makeResult(makeIdentity({ name: null, level: 'UNKNOWN', person_id: null }))
    const { container } = render(<IdentitySection result={result} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders name, confidence level, and occupation for a HIGH match', () => {
    render(<IdentitySection result={makeResult(makeIdentity())} />)
    expect(screen.getByText('Sundar Pichai')).toBeInTheDocument()
    expect(screen.getByText(/HIGH CONFIDENCE/)).toBeInTheDocument()
    expect(screen.getByText(/96%/)).toBeInTheDocument()
    expect(screen.getByText('CEO, Google and Alphabet')).toBeInTheDocument()
  })

  it('renders official profiles as clickable links with a ✓', () => {
    const result = makeResult(makeIdentity(), [makeAccount()])
    render(<IdentitySection result={result} />)
    const link = screen.getByRole('link', { name: /X\/Twitter/ })
    expect(link).toHaveAttribute('href', 'https://x.com/sundarpichai')
    expect(link.textContent).toMatch(/✓/)
  })

  it('renders a likely-official profile with a ? not a ✓', () => {
    const account = makeAccount({ status: 'LIKELY_OFFICIAL', platform: 'GitHub', url: 'https://github.com/example' })
    const result = makeResult(makeIdentity(), [account])
    render(<IdentitySection result={result} />)
    const link = screen.getByRole('link', { name: /GitHub/ })
    expect(link.textContent).toMatch(/\?/)
    expect(link.textContent).not.toMatch(/✓/)
  })

  it('shows supporting evidence only after expanding it', () => {
    render(<IdentitySection result={makeResult(makeIdentity())} />)
    expect(screen.queryByText(/face similarity 0.910/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByText(/Why this identification/))
    expect(screen.getByText(/face similarity 0.910/)).toBeInTheDocument()
  })

  it('never presents the identity as a definitive claim', () => {
    render(<IdentitySection result={makeResult(makeIdentity())} />)
    expect(screen.getByText(/not a definitive claim of identity/i)).toBeInTheDocument()
  })

  // ---- API Ninjas celebrity metadata enrichment (additive) --------------

  it('renders celebrity occupation, nationality, birthday, and age when available', () => {
    render(<IdentitySection result={makeResult(makeIdentity(), [], makeCelebrity())} />)
    expect(screen.getByText('businessperson / executive')).toBeInTheDocument()
    expect(screen.getByText(/India/)).toBeInTheDocument()
    expect(screen.getByText('July 12, 1972')).toBeInTheDocument()
    expect(screen.getByText('54')).toBeInTheDocument()
  })

  it('does not render a celebrity block when unavailable', () => {
    const unavailable = makeCelebrity({ available: false, name: null, occupations: [] })
    render(<IdentitySection result={makeResult(makeIdentity(), [], unavailable)} />)
    expect(screen.queryByLabelText('Celebrity information')).not.toBeInTheDocument()
  })

  it('does not render a celebrity block when celebrity data is absent entirely', () => {
    render(<IdentitySection result={makeResult(makeIdentity())} />)
    expect(screen.queryByLabelText('Celebrity information')).not.toBeInTheDocument()
  })

  it('falls back to the identity occupation when celebrity has none', () => {
    const celebrity = makeCelebrity({ occupations: [], birthday: null, age: null })
    render(<IdentitySection result={makeResult(makeIdentity(), [], celebrity)} />)
    expect(screen.getByText('CEO, Google and Alphabet')).toBeInTheDocument()
  })

  it('official profiles still come from the existing social-discovery system, not from celebrity data', () => {
    const account = makeAccount()
    render(<IdentitySection result={makeResult(makeIdentity(), [account], makeCelebrity())} />)
    // Only the one profile explicitly passed via official_profiles renders —
    // celebrity metadata contributes no profile links of its own.
    expect(screen.getAllByRole('link')).toHaveLength(1)
  })
})
