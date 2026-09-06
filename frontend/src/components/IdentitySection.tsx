import React, { useState } from 'react'
import type { CaseResult, SocialAccount, SocialAccountStatus } from '../types/api'

const LEVEL_STYLE: Record<string, string> = {
  HIGH: 'text-success border-success/60 bg-green-900/10',
  MEDIUM: 'text-accent border-accent/60 bg-blue-900/10',
  LOW: 'text-warn border-warn/60 bg-yellow-900/10',
  UNKNOWN: 'text-muted border-border',
}

const STATUS_ICON: Record<SocialAccountStatus, string> = {
  OFFICIAL: '✓',
  LIKELY_OFFICIAL: '?',
  UNVERIFIED: '?',
  REJECTED: '✗',
}

const STATUS_STYLE: Record<SocialAccountStatus, string> = {
  OFFICIAL: 'text-success border-success/50 bg-green-900/10',
  LIKELY_OFFICIAL: 'text-warn border-warn/50 bg-yellow-900/10',
  UNVERIFIED: 'text-muted border-border',
  REJECTED: 'text-danger border-danger/40 bg-red-900/10',
}

const STATUS_LABEL: Record<SocialAccountStatus, string> = {
  OFFICIAL: 'Official/public profile found',
  LIKELY_OFFICIAL: 'Likely official — not independently corroborated this scan',
  UNVERIFIED: 'Unverified',
  REJECTED: 'Rejected — not a genuine match',
}

interface Props {
  result: CaseResult
}

/**
 * Known-person identity recognition — additive section, rendered only when
 * `result.identity` names someone (HIGH/MEDIUM confidence). LOW/UNKNOWN
 * never renders a name here at all ("never force a name" — see
 * identity/scorer.py), so there is nothing uncertain to show.
 */
export default function IdentitySection({ result }: Props) {
  const [showEvidence, setShowEvidence] = useState(false)
  const identity = result.identity
  if (!identity || !identity.name) return null

  const levelStyle = LEVEL_STYLE[identity.level] ?? LEVEL_STYLE.UNKNOWN
  const profiles = result.official_profiles ?? []

  return (
    <section className="mb-6 rounded-lg border border-border bg-surface-1 p-5" aria-labelledby="identity-heading">
      <h2 id="identity-heading" className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
        Detected Identity
      </h2>

      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="text-xl font-bold text-gray-100">{identity.name}</div>
          {identity.occupation && (
            <p className="text-sm text-muted mt-0.5">{identity.occupation}</p>
          )}
        </div>
        <div className={`px-3 py-1 rounded border text-sm font-mono ${levelStyle}`}>
          {identity.level} CONFIDENCE · {Math.round(identity.confidence * 100)}%
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
        <div>
          <span className="text-muted">Face match</span>
          <div className="font-mono text-gray-300">{Math.round(identity.face_similarity * 100)}%</div>
        </div>
        <div>
          <span className="text-muted">Margin over next candidate</span>
          <div className="font-mono text-gray-300">{Math.round(identity.margin * 100)}%</div>
        </div>
        <div>
          <span className="text-muted">Independent evidence</span>
          <div className="font-mono text-gray-300">
            {identity.evidence_count} signal{identity.evidence_count === 1 ? '' : 's'}
          </div>
        </div>
      </div>

      {profiles.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border/50">
          <h3 className="text-xs text-muted uppercase tracking-wider mb-2">Official / Verified Profiles</h3>
          <div className="flex flex-wrap gap-2" role="list" aria-label="Official profiles">
            {profiles.map((p) => (
              <ProfileChip key={p.platform} account={p} />
            ))}
          </div>
        </div>
      )}

      {identity.supporting_evidence.length > 0 && (
        <div className="mt-3">
          <button
            onClick={() => setShowEvidence((v) => !v)}
            className="text-xs text-muted hover:text-gray-300 transition-colors"
            aria-expanded={showEvidence}
          >
            <span className={`inline-block transition-transform ${showEvidence ? 'rotate-90' : ''}`} aria-hidden>▶</span>
            {' '}Why this identification
          </button>
          {showEvidence && (
            <ul className="mt-2 space-y-1" role="list" aria-label="Supporting evidence">
              {identity.supporting_evidence.map((e, i) => (
                <li key={i} className="text-xs text-muted">
                  <span className="text-gray-300 font-mono">{e.kind}</span>: {e.description}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <p className="mt-3 text-xs text-muted">
        Evidence-based, not a definitive claim of identity — see supporting evidence above.
      </p>
    </section>
  )
}

function ProfileChip({ account }: { account: SocialAccount }) {
  const style = STATUS_STYLE[account.status] ?? STATUS_STYLE.UNVERIFIED
  const icon = STATUS_ICON[account.status] ?? '?'
  return (
    <a
      href={account.url}
      target="_blank"
      rel="noopener noreferrer"
      title={STATUS_LABEL[account.status]}
      onClick={(e) => {
        if (!window.confirm(`Opening external URL:\n${account.url}\n\nThis link opens in a new tab. Continue?`)) {
          e.preventDefault()
        }
      }}
      className={`px-3 py-1 rounded-full text-xs font-mono border hover:brightness-125 transition-all ${style}`}
      aria-label={`${account.platform}: ${STATUS_LABEL[account.status]}`}
    >
      {account.platform} {icon}
    </a>
  )
}
