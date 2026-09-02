import React from 'react'
import type { VerifiedCandidate } from '../types/api'

const LADDER = ['SEARCH_FOUND', 'SOCIAL_MATCH', 'IMAGE_MATCH', 'FACE_MATCH', 'VERIFIED']

const PLATFORM_ICONS: Record<string, string> = {
  Instagram: '📸',
  'X/Twitter': '🐦',
  YouTube: '▶️',
  LinkedIn: '💼',
  Facebook: '👥',
  TikTok: '🎵',
  Reddit: '🤖',
  Pinterest: '📌',
  Threads: '🧵',
  Bluesky: '🦋',
  Tumblr: '📝',
  Mastodon: '🐘',
  VK: '💬',
  Weibo: '🌐',
  Flickr: '📷',
  GitHub: '🐙',
}

/**
 * How the match was established. EXACT_IMAGE and SAME_FACE are claims about
 * measurements — the same picture versus a different picture of the same face —
 * and the distinction matters: a reposted photo and an independent photograph
 * are different kinds of evidence.
 */
const TYPE_LABEL: Record<string, string> = {
  EXACT_IMAGE: 'Exact image — the same picture',
  SAME_FACE: 'Same face — a different picture',
  SOCIAL_PROFILE: 'Social profile page',
  SOCIAL_POST: 'Social post',
  DEVELOPER_PROFILE: 'Developer profile',
  PUBLIC_ARTICLE: 'Public article',
  PUBLIC_WEB_PAGE: 'Public web page',
  OTHER: 'Other',
}

const TYPE_STYLE: Record<string, string> = {
  EXACT_IMAGE: 'text-accent border-accent/50',
  SAME_FACE: 'text-success border-success/50',
}

const BAND_STYLE: Record<string, string> = {
  STRONG: 'text-success border-success/50 bg-green-900/10',
  MODERATE: 'text-accent border-accent/50 bg-blue-900/10',
  WEAK: 'text-warn border-warn/50 bg-yellow-900/10',
  INSUFFICIENT: 'text-danger border-danger/50 bg-red-900/10',
}

interface Props {
  candidate: VerifiedCandidate
  rank?: number
}

export default function CandidateCard({ candidate: c, rank }: Props) {
  const isVerified = c.verified
  const icon = c.platform ? (PLATFORM_ICONS[c.platform] ?? '🌐') : '🌐'
  const bandStyle = BAND_STYLE[c.confidence_band] ?? BAND_STYLE.INSUFFICIENT

  return (
    <article
      className={`rounded-lg border p-4 transition-colors
        ${isVerified
          ? 'border-success/60 bg-surface-2'
          : 'border-border bg-surface-1 opacity-80'}`}
      aria-label={`Candidate: ${c.domain} — ${isVerified ? 'VERIFIED' : 'REJECTED'}`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          {rank !== undefined && (
            <span className="text-xs text-muted font-mono shrink-0">#{rank + 1}</span>
          )}
          <span className="text-xl shrink-0" aria-hidden>{icon}</span>
          <div className="min-w-0">
            <a
              href={c.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline text-sm font-mono truncate block max-w-xs"
              onClick={(e) => {
                // Interstitial: warn before opening external link
                if (!window.confirm(
                  `Opening external URL:\n${c.url}\n\nThis link opens in a new tab. Continue?`
                )) e.preventDefault()
              }}
              title={c.url}
            >
              {c.domain}
            </a>
            <span className="text-xs text-muted">
              {c.platform ?? 'Other Web'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span
            className={`px-2 py-0.5 rounded border text-xs font-mono ${TYPE_STYLE[c.candidate_type] ?? 'text-muted border-border'}`}
            title={TYPE_LABEL[c.candidate_type] ?? c.candidate_type}
            data-testid="candidate-type"
          >
            {c.candidate_type}
          </span>
          {isVerified ? (
            <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-success/20 text-success border border-success/50">
              VERIFIED
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-red-900/20 text-danger border border-danger/30">
              REJECTED
            </span>
          )}
          <span className={`px-2 py-0.5 rounded border text-xs ${bandStyle}`}>
            {c.confidence_band}
          </span>
        </div>
      </div>

      {/* Score bars */}
      <div className="grid grid-cols-3 gap-3 mb-3 text-xs">
        <ScoreBar label="Face sim" value={c.face_similarity} threshold={0.38} />
        <ScoreBar label="Image sim" value={c.image_similarity} threshold={0.80} />
        <ScoreBar label="Metadata" value={c.metadata_consistency} />
      </div>

      {/* Final score */}
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xs text-muted">Final score</span>
        <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${c.final_score >= 0.7 ? 'bg-success' : c.final_score >= 0.4 ? 'bg-warn' : 'bg-danger'}`}
            style={{ width: `${Math.min(c.final_score * 100, 100)}%` }}
            role="progressbar"
            aria-valuenow={Math.round(c.final_score * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Final score: ${Math.round(c.final_score * 100)}%`}
          />
        </div>
        <span className={`text-sm font-bold font-mono ${c.final_score >= 0.7 ? 'text-success' : 'text-warn'}`}>
          {Math.round(c.final_score * 100)}%
        </span>
      </div>

      {/* Verification ladder */}
      <div className="flex items-center gap-1 mb-3 flex-wrap" aria-label="Verification ladder">
        {LADDER.map((rung, i) => {
          const reached = c.stages.includes(rung)
          return (
            <React.Fragment key={rung}>
              <span
                className={`text-xs px-2 py-0.5 rounded border font-mono
                  ${reached
                    ? rung === 'VERIFIED' ? 'border-success/60 text-success bg-success/10'
                    : 'border-accent/50 text-accent'
                    : 'border-border text-muted opacity-40'}`}
                title={rung}
              >
                {rung.replace('_', ' ')}
              </span>
              {i < LADDER.length - 1 && (
                <span className={`text-xs ${reached ? 'text-accent' : 'text-border'}`} aria-hidden>→</span>
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* Meta row */}
      <div className="text-xs text-muted flex flex-wrap gap-x-4 gap-y-1">
        <span>via <span className="text-gray-300">{c.engine}</span></span>
        {c.candidate_image_source && (
          <span>img src: <span className="text-gray-300">{c.candidate_image_source}</span></span>
        )}
        {c.match_type !== 'none' && (
          <span>match: <span className={c.match_type === 'exact-image' ? 'text-success' : 'text-warn'}>{c.match_type}</span></span>
        )}
        {c.candidate_faces_found > 0 && (
          <span>{c.candidate_faces_found} face(s) in candidate</span>
        )}
      </div>

      {/* Rejection reason */}
      {!isVerified && c.rejection_reason && (
        <div className="mt-2 text-xs text-danger/80 bg-red-900/10 border border-danger/20 rounded px-2 py-1">
          ✗ {c.rejection_reason}
        </div>
      )}
    </article>
  )
}

function ScoreBar({ label, value, threshold }: { label: string; value: number; threshold?: number }) {
  const pct = Math.min(Math.max(value, 0), 1)
  const threshPct = threshold ? threshold * 100 : undefined
  const color = threshold
    ? value >= threshold ? 'bg-success' : 'bg-danger'
    : 'bg-accent'

  return (
    <div>
      <div className="flex justify-between mb-0.5">
        <span className="text-muted">{label}</span>
        <span className="font-mono text-gray-300">{Math.round(pct * 100)}%</span>
      </div>
      <div className="relative h-1.5 bg-surface rounded-full overflow-hidden">
        {threshPct && (
          <div
            className="absolute top-0 bottom-0 w-px bg-warn/60 z-10"
            style={{ left: `${threshPct}%` }}
            aria-hidden
          />
        )}
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct * 100}%` }}
          role="progressbar"
          aria-valuenow={Math.round(pct * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label}: ${Math.round(pct * 100)}%`}
        />
      </div>
    </div>
  )
}
