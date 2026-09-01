import React, { useState } from 'react'
import CandidateCard from './CandidateCard'
import type { ScanState } from '../App'
import type { VerifiedCandidate } from '../types/api'
import { api } from '../api/client'

const VERDICT_STYLE: Record<string, string> = {
  VERIFIED: 'text-success border-success/60 bg-green-900/10',
  VERIFIED_SIMULATED: 'text-warn border-warn/60 bg-yellow-900/10',
  VERIFIED_OFFCHAIN: 'text-warn border-warn/60 bg-yellow-900/10',
  UNVERIFIED: 'text-danger border-danger/60 bg-red-900/10',
  NO_FACE: 'text-danger border-danger/60 bg-red-900/10',
  NO_SEARCH_RESULTS: 'text-warn border-warn/60 bg-yellow-900/10',
  INCOMPLETE: 'text-muted border-border',
}

interface Props {
  scan: ScanState
  onViewEvidence: () => void
  onNewScan: () => void
}

export default function ResultView({ scan, onViewEvidence, onNewScan }: Props) {
  const [showRejected, setShowRejected] = useState(false)
  const { result, caseId, failed } = scan

  if (failed && !result) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="border border-danger/60 bg-red-900/10 rounded-lg p-6 text-center">
          <h1 className="text-2xl font-bold text-danger mb-2">Pipeline Failed</h1>
          <p className="text-muted text-sm mb-4">
            The scan encountered an unrecoverable error. Check the backend logs for details.
          </p>
          <button
            onClick={onNewScan}
            className="px-4 py-2 rounded bg-surface-3 border border-border text-sm hover:border-accent/60"
          >
            ← New Scan
          </button>
        </div>
      </div>
    )
  }

  if (!result) return null

  const verdict = result.verdict
  const verdictStyle = VERDICT_STYLE[verdict] ?? VERDICT_STYLE.INCOMPLETE
  const verified = result.verification.filter((c) => c.verified)
  const rejected = result.verification.filter((c) => !c.verified)
  const search = result.reverse_search
  const chain = result.blockchain

  return (
    <div className="max-w-3xl mx-auto">
      {/* Verdict banner */}
      <div className={`rounded-lg border p-5 mb-6 ${verdictStyle}`}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold font-mono">{verdict}</h1>
            {result.failure_reason && (
              <p className="text-sm mt-1 opacity-80">{result.failure_reason}</p>
            )}
          </div>
          <div className="text-right text-xs text-muted">
            <div>Case <span className="font-mono text-gray-300">{result.case_id}</span></div>
            <div>Pipeline <span className="font-mono">{result.pipeline_version}</span></div>
          </div>
        </div>
        {result.stages_passed.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1" aria-label="Stages reached">
            {result.stages_passed.map((s) => (
              <span key={s} className="text-xs px-2 py-0.5 rounded border border-current opacity-70 font-mono">
                {s}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Evidence hash */}
      {result.evidence_sha256 && (
        <div className="mb-6 flex items-center gap-2 bg-surface-1 border border-border rounded px-4 py-2">
          <span className="text-xs text-muted shrink-0">evidenceHash</span>
          <span className="font-mono text-xs text-gray-300 truncate flex-1" title={result.evidence_sha256}>
            sha256:{result.evidence_sha256}
          </span>
          <button
            onClick={() => navigator.clipboard.writeText(result.evidence_sha256!)}
            className="text-xs text-accent hover:text-accent-dim shrink-0"
            aria-label="Copy evidence hash"
          >
            copy
          </button>
        </div>
      )}

      {/* Search stats */}
      {search && (
        <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="search-stats">
          <h2 id="search-stats" className="text-sm font-semibold text-gray-300 mb-2">Search</h2>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <Kv k="Engines tried" v={search.engines_attempted.join(', ') || '—'} />
            <Kv k="Engines succeeded" v={search.engines_succeeded.join(', ') || 'none'} />
            <Kv k="Total candidates" v={String(search.total_candidates)} />
            <Kv k="Social candidates" v={String(search.social_candidates)} />
            {Object.entries(search.engine_errors).map(([eng, err]) => (
              <div key={eng} className="col-span-2">
                <span className="text-warn">{eng}: </span>
                <span className="text-muted">{err}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Face record */}
      {result.face && (
        <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="face-record">
          <h2 id="face-record" className="text-sm font-semibold text-gray-300 mb-2">Face</h2>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <Kv k="Detected" v={result.face.detected ? 'yes' : 'no'} />
            <Kv k="Model" v={result.face.model} />
            <Kv k="Faces found" v={String(result.face.faces_found)} />
            <Kv k="Embedding" v={result.face.embedding_dimension ? `${result.face.embedding_dimension}-D` : '—'} />
            {result.face.embedding_sha256 && (
              <div className="col-span-2 text-muted">
                embedding sha256: <span className="font-mono">{result.face.embedding_sha256.slice(0, 24)}…</span>
                <span className="ml-2 text-warn text-[10px]">(vector stays local — only hash recorded)</span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Verified candidates */}
      {verified.length > 0 && (
        <section className="mb-6" aria-labelledby="verified-heading">
          <h2 id="verified-heading" className="text-sm font-semibold text-success mb-3">
            ✓ Verified Matches ({verified.length})
          </h2>
          <div className="space-y-3">
            {verified.map((c, i) => (
              <CandidateCard key={c.url} candidate={c} rank={i} />
            ))}
          </div>
        </section>
      )}

      {/* Rejected candidates — collapsed by default */}
      {rejected.length > 0 && (
        <section className="mb-6" aria-labelledby="rejected-heading">
          <button
            id="rejected-heading"
            onClick={() => setShowRejected((p) => !p)}
            className="flex items-center gap-2 text-sm text-muted hover:text-gray-300 transition-colors mb-2"
            aria-expanded={showRejected}
          >
            <span className={`transition-transform ${showRejected ? 'rotate-90' : ''}`} aria-hidden>▶</span>
            Rejected candidates ({rejected.length})
          </button>
          {showRejected && (
            <div className="space-y-3" role="list" aria-label="Rejected candidates">
              {rejected.map((c) => (
                <CandidateCard key={c.url} candidate={c} />
              ))}
            </div>
          )}
        </section>
      )}

      {/* Degraded / empty state */}
      {result.verification.length === 0 && (
        <div className="mb-6 border border-warn/40 bg-yellow-900/10 rounded p-4 text-sm text-warn">
          No candidates were verified. {result.failure_reason ?? ''}
          {search?.engines_succeeded.length === 0 && (
            <span className="block mt-1 text-xs text-muted">
              0 engines succeeded — check API keys in .env or try again with --headful
            </span>
          )}
        </div>
      )}

      {/* Blockchain */}
      {chain && chain.mode !== 'skipped' && (
        <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="chain-heading">
          <h2 id="chain-heading" className="text-sm font-semibold text-gray-300 mb-2">Blockchain</h2>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <Kv k="Network" v={`${chain.network} (${chain.chain_id})`} />
            <Kv k="Mode" v={chain.mode} />
            {chain.tx_hash && <Kv k="Tx hash" v={chain.tx_hash.slice(0, 18) + '…'} />}
            {chain.attestation_uid && <Kv k="Attestation UID" v={chain.attestation_uid.slice(0, 18) + '…'} />}
            {chain.mode === 'onchain' && (
              <Kv k="Read-back" v={chain.readback_verified ? '✓ PASS' : '✗ FAIL'} />
            )}
          </div>
          {chain.explorer_attestation && (
            <a
              href={chain.explorer_attestation}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline mt-2 block"
              onClick={(e) => {
                if (!window.confirm(`Open EAS explorer?\n${chain.explorer_attestation}`)) e.preventDefault()
              }}
            >
              View on EAS explorer ↗
            </a>
          )}
          {chain.note && <p className="text-xs text-muted mt-1">{chain.note}</p>}
        </section>
      )}

      {/* Action row */}
      <div className="flex gap-3 flex-wrap">
        <button
          onClick={onViewEvidence}
          className="px-4 py-2 rounded border border-accent/60 text-accent text-sm hover:bg-surface-2 transition-colors focus-visible:ring-2"
        >
          View Evidence Bundle
        </button>
        <a
          href={api.getEvidenceUrl(caseId)}
          download={`${caseId}.zip`}
          className="px-4 py-2 rounded border border-border text-muted text-sm hover:border-accent/60 hover:text-accent transition-colors"
          aria-label="Download evidence ZIP"
        >
          Download .zip
        </a>
        <button
          onClick={onNewScan}
          className="px-4 py-2 rounded border border-border text-muted text-sm hover:border-accent/60 hover:text-accent transition-colors focus-visible:ring-2"
        >
          ← New Scan
        </button>
      </div>

      <p className="mt-6 text-xs text-muted">
        Scope: face similarity {result.best_match ? `${Math.round(result.best_match.face_similarity * 100)}%` : 'N/A'} at
        threshold 38% under recorded config. This record attests that the input image and its primary face
        match the retrieved public image. <strong className="text-gray-300">Not an identity claim.</strong>
      </p>
    </div>
  )
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <span className="text-muted">{k}: </span>
      <span className="text-gray-300 font-mono">{v}</span>
    </div>
  )
}
