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
  FACE_QUALITY_INSUFFICIENT: 'text-danger border-danger/60 bg-red-900/10',
  FACE_SELECTION_REQUIRED: 'text-warn border-warn/60 bg-yellow-900/10',
  INVALID_CROP: 'text-danger border-danger/60 bg-red-900/10',
  INVALID_FACE_SELECTION: 'text-danger border-danger/60 bg-red-900/10',
  NO_SEARCH_RESULTS: 'text-warn border-warn/60 bg-yellow-900/10',
  CHAIN_MISMATCH: 'text-danger border-danger/60 bg-red-900/10',
  INCOMPLETE: 'text-muted border-border',
}

/** Plain-language gloss for the pipeline's verdict codes. */
const VERDICT_LABEL: Record<string, string> = {
  VERIFIED: 'Face verified and attested on-chain',
  VERIFIED_OFFCHAIN: 'Face verified — evidence generated, but not attested on-chain',
  VERIFIED_SIMULATED: 'Face verified — attestation simulated, nothing written on-chain',
  UNVERIFIED: 'No candidate met the verification thresholds',
  NO_FACE: 'No usable face detected in the uploaded image',
  FACE_QUALITY_INSUFFICIENT: 'The selected face is not good enough to match reliably',
  FACE_SELECTION_REQUIRED: 'Which face to scan is ambiguous — a selection is needed',
  INVALID_CROP: 'The supplied crop could not be applied to this image',
  INVALID_FACE_SELECTION: 'The selected face does not exist in this image',
  NO_SEARCH_RESULTS: 'No search provider returned usable candidates',
  CHAIN_MISMATCH: 'The on-chain record does not match the local evidence',
}

// Terminal provider states, grouped by what a reader should conclude.
const PROVIDER_TONE: Record<string, string> = {
  COMPLETED: 'text-success',
  NO_RESULTS: 'text-muted',
  NOT_CONFIGURED: 'text-muted',
  CHALLENGED: 'text-warn',
  RATE_LIMITED: 'text-warn',
  TIMEOUT: 'text-warn',
  CANCELLED: 'text-warn',
  FAILED: 'text-danger',
}

// Discovery priority order, mirroring the backend's PLATFORM_PRIORITY.
const PRIORITY_PLATFORMS = ['LinkedIn', 'Instagram', 'X/Twitter', 'GitHub', 'YouTube']

function groupByPlatform(candidates: VerifiedCandidate[]): [string, VerifiedCandidate[]][] {
  const groups = new Map<string, VerifiedCandidate[]>()
  for (const c of candidates) {
    const key = c.platform ?? 'Other Web'
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(c)
  }
  // Priority platforms first in their canonical order, then anything else
  // alphabetically, with the unrecognised web last.
  const rest = [...groups.keys()]
    .filter((k) => !PRIORITY_PLATFORMS.includes(k) && k !== 'Other Web')
    .sort()
  const order = [...PRIORITY_PLATFORMS, ...rest, 'Other Web']
  return order.filter((k) => groups.has(k)).map((k) => [k, groups.get(k)!])
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
  const thresholds = result.threshold_snapshot
  const thresholdPct = thresholds ? Math.round(thresholds.face_match_threshold * 100) : 38
  const graph = result.evidence_graph

  return (
    <div className="max-w-3xl mx-auto">
      {/* Verdict banner */}
      <div className={`rounded-lg border p-5 mb-6 ${verdictStyle}`}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold font-mono">{verdict}</h1>
            {VERDICT_LABEL[verdict] && (
              <p className="text-sm mt-0.5 opacity-90">{VERDICT_LABEL[verdict]}</p>
            )}
            {result.failure_reason && (
              <p className="text-sm mt-1 opacity-80">{result.failure_reason}</p>
            )}
          </div>
          <div className="text-right text-xs text-muted">
            <div>Case <span className="font-mono text-gray-300">{result.case_id}</span></div>
            <div>Pipeline <span className="font-mono">{result.pipeline_version}</span></div>
          </div>
        </div>
        {/* The union of rungs *any* candidate reached, which is not the same as
            the best match's own ladder — that is on its card. Labelled so the
            two are not confused: a social rung here can come from a candidate
            that was not the one that verified. */}
        {result.stages_passed.length > 0 && (
          <div className="mt-3">
            <span className="text-[10px] uppercase tracking-wider opacity-60">
              Rungs reached across all candidates
            </span>
          </div>
        )}
        {result.stages_passed.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1" aria-label="Rungs reached across all candidates">
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
          {/* Every provider's terminal state. A challenged or unconfigured
              provider is reported as such rather than being folded into
              "found nothing" — they are different facts about the search. */}
          <table className="w-full text-xs font-mono mb-3">
            <caption className="sr-only">Search provider outcomes</caption>
            <thead>
              <tr className="text-muted text-left">
                <th scope="col" className="font-normal pb-1">Provider</th>
                <th scope="col" className="font-normal pb-1">Status</th>
                <th scope="col" className="font-normal pb-1 text-right">Candidates</th>
                <th scope="col" className="font-normal pb-1 text-right">Time</th>
              </tr>
            </thead>
            <tbody data-testid="provider-table">
              {search.providers.length === 0 && (
                <tr><td colSpan={4} className="text-muted">no provider ran</td></tr>
              )}
              {search.providers.map((p) => (
                <tr key={p.engine} className="border-t border-border/50">
                  <td className="py-1 text-gray-300">{p.engine}</td>
                  <td className={`py-1 ${PROVIDER_TONE[p.status] ?? 'text-muted'}`}>
                    {p.status}
                    {p.error && (
                      <span className="block text-[10px] text-muted font-sans" title={p.error}>
                        {p.error.slice(0, 90)}
                      </span>
                    )}
                  </td>
                  <td className="py-1 text-right text-gray-300">{p.candidates}</td>
                  <td className="py-1 text-right text-muted">{p.duration_s.toFixed(1)}s</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <Kv k="Total unique candidates" v={String(search.total_candidates)} />
            <Kv k="On named platforms" v={String(search.social_candidates)} />
          </div>

          {Object.keys(search.platform_counts).length > 0 && (
            <div className="mt-3">
              <h3 className="text-xs text-muted uppercase tracking-wider mb-1">
                Candidates discovered by platform
              </h3>
              {/* Zeros are shown deliberately: "we looked for LinkedIn and
                  found none" must not render the same as "we never looked". */}
              <div className="flex flex-wrap gap-2" data-testid="platform-counts">
                {Object.entries(search.platform_counts).map(([name, n]) => (
                  <span
                    key={name}
                    className={`px-2 py-0.5 rounded text-xs font-mono border
                      ${n > 0 ? 'border-accent/50 text-accent' : 'border-border text-muted'}`}
                  >
                    {name} {n}
                  </span>
                ))}
              </div>
            </div>
          )}

          {search.timed_out && (
            <p className="mt-2 text-xs text-warn">
              The search budget was exhausted before every provider finished; the providers
              marked TIMEOUT above were abandoned.
            </p>
          )}

          {search.variants.length > 0 && (
            <div className="mt-3 pt-2 border-t border-border/50">
              <h3 className="text-xs text-muted uppercase tracking-wider mb-1">
                Search variants (beyond the original upload)
              </h3>
              <div className="flex flex-wrap gap-2" data-testid="search-variants">
                {search.variants.map((v) => (
                  <span
                    key={v.variant_id}
                    className={`px-2 py-0.5 rounded text-xs font-mono border
                      ${v.skipped ? 'border-border text-muted' : 'border-accent/50 text-accent'}`}
                    title={v.skipped ? v.skip_reason : `${v.width}x${v.height}`}
                  >
                    {v.variant_type} {v.skipped ? '(skipped)' : `· +${v.candidates_found}`}
                  </span>
                ))}
              </div>
            </div>
          )}
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
            {result.face.quality && (
              <>
                <Kv k="Quality gate" v={result.face.quality.passed ? 'PASS' : (result.face.quality.error ?? 'FAIL')} />
                <Kv k="Overall quality" v={`${Math.round(result.face.quality.overall_quality * 100)}%`} />
                {Object.entries(result.face.quality.bands).map(([metric, band]) => (
                  <Kv key={metric} k={metric[0].toUpperCase() + metric.slice(1)} v={band} />
                ))}
                <Kv k="Pose (yaw / roll)" v={`${result.face.quality.yaw_deg.toFixed(1)}° / ${result.face.quality.roll_deg.toFixed(1)}°`} />
              </>
            )}
            {result.face_selection && (
              <>
                <Kv
                  k="Selection"
                  v={
                    result.face_selection.mode +
                    (result.face_selection.face_index !== null
                      ? ` (face #${result.face_selection.face_index + 1} of ${result.face_selection.faces_offered})`
                      : '')
                  }
                />
                <Kv
                  k="Crop"
                  v={result.face_selection.crop_rect
                    ? result.face_selection.crop_rect.join(',')
                    : 'none — original used unmodified'}
                />
                {/* The original's hash is kept whether or not a crop was taken,
                    so a crop can never stand in for the uploaded evidence. */}
                <div className="col-span-2 text-muted">
                  original sha256:{' '}
                  <span className="font-mono">{result.face_selection.original_sha256.slice(0, 24)}…</span>
                  {result.face_selection.crop_sha256 && (
                    <>
                      {' · crop sha256: '}
                      <span className="font-mono">{result.face_selection.crop_sha256.slice(0, 24)}…</span>
                    </>
                  )}
                </div>
              </>
            )}
            {result.face.embedding_sha256 && (
              <div className="col-span-2 text-muted">
                embedding sha256: <span className="font-mono">{result.face.embedding_sha256.slice(0, 24)}…</span>
                <span className="ml-2 text-warn text-[10px]">(vector stays local — only hash recorded)</span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Thresholds + corroboration. The exact config that decided this
          verdict, and how much genuinely independent evidence backs it —
          shown explicitly rather than left implicit in the final score. */}
      {(thresholds || graph) && (
        <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="calibration-heading">
          <h2 id="calibration-heading" className="text-sm font-semibold text-gray-300 mb-2">
            Thresholds &amp; Corroboration
          </h2>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {thresholds && (
              <>
                <Kv k="Face match threshold" v={`${Math.round(thresholds.face_match_threshold * 100)}%`} />
                <Kv k="Image match threshold" v={`${Math.round(thresholds.image_match_threshold * 100)}%`} />
                <Kv k="Verify min score" v={`${Math.round(thresholds.verify_min_score * 100)}%`} />
                <Kv k="Model" v={`${thresholds.insightface_model} / ${thresholds.face_backend}`} />
                <div className="col-span-2">
                  <span className="text-muted">Calibration: </span>
                  <span className={
                    thresholds.calibration_status === 'CALIBRATED' ? 'text-success font-mono'
                    : thresholds.calibration_status === 'CALIBRATION_INSUFFICIENT' ? 'text-warn font-mono'
                    : 'text-muted font-mono'
                  }>
                    {thresholds.calibration_status}
                  </span>
                  <span className="text-muted ml-2">{thresholds.calibration_note}</span>
                </div>
              </>
            )}
            {graph && (
              <div className="col-span-2 mt-1 pt-2 border-t border-border/50">
                <span className="text-muted">Independent evidence sources: </span>
                <span className={`font-mono font-bold ${graph.independent_evidence_count >= 2 ? 'text-success' : 'text-muted'}`}>
                  {graph.independent_evidence_count}
                </span>
                <span className="text-muted ml-2">
                  ({graph.nodes.filter((n) => n.type === 'image').length} distinct image(s) across{' '}
                  {graph.nodes.filter((n) => n.type === 'domain').length} domain(s) — reposts of the same
                  photo count once, not once per URL)
                </span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Best match, called out explicitly. Chosen by evidential strength, so
          a strong match on the wider web outranks a weaker one on a priority
          platform — the platform name never promotes a candidate. */}
      {result.best_match && (
        <section className="mb-6" aria-labelledby="best-heading">
          <h2 id="best-heading" className="text-sm font-semibold text-gray-300 mb-3">
            Best candidate
          </h2>
          <CandidateCard candidate={result.best_match} rank={0} thresholds={result.threshold_snapshot} />
        </section>
      )}

      {/* Verified candidates, grouped by platform in discovery-priority order */}
      {verified.length > 0 && (
        <section className="mb-6" aria-labelledby="verified-heading">
          <h2 id="verified-heading" className="text-sm font-semibold text-success mb-3">
            ✓ Verified Matches ({verified.length})
          </h2>
          <div className="space-y-5" data-testid="verified-groups">
            {groupByPlatform(verified).map(([platform, group]) => (
              <div key={platform}>
                <h3 className="text-xs uppercase tracking-wider text-muted mb-2">
                  {platform} <span className="text-gray-400">({group.length})</span>
                </h3>
                <div className="space-y-3">
                  {group.map((c, i) => (
                    <CandidateCard key={c.url} candidate={c} rank={i} thresholds={result.threshold_snapshot} />
                  ))}
                </div>
              </div>
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
                <CandidateCard key={c.url} candidate={c} thresholds={result.threshold_snapshot} />
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

      {/* Blockchain. Only real, public transaction data is ever shown; when
          attestation was skipped or failed the panel says so instead. */}
      {chain && (
        <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="chain-heading">
          <h2 id="chain-heading" className="text-sm font-semibold text-gray-300 mb-2">Blockchain</h2>

          {chain.mode === 'skipped' && (
            <p className="text-xs text-muted" data-testid="chain-skipped">
              On-chain attestation was skipped for this scan{chain.note ? ` — ${chain.note}` : ''}.
              The evidence bundle and its SHA-256 were still generated, so this case can be
              attested later without re-running the scan.
            </p>
          )}

          {chain.mode === 'failed' && (
            <p className="text-xs text-warn" data-testid="chain-failed">
              Attestation failed, so no transaction exists for this case. Face verification is
              unaffected — it happened locally and is fully recorded.
              {chain.note && <span className="block mt-1 text-muted">{chain.note}</span>}
            </p>
          )}

          {chain.mode !== 'skipped' && chain.mode !== 'failed' && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <Kv k="Network" v={`${chain.network} (chain ${chain.chain_id})`} />
              <Kv k="Mode" v={chain.mode} />
              {chain.attester && <Kv k="Attester" v={chain.attester} />}
              {chain.schema_uid && <Kv k="Schema UID" v={chain.schema_uid.slice(0, 18) + '…'} />}
              {chain.tx_hash && <Kv k="Tx hash" v={chain.tx_hash.slice(0, 18) + '…'} />}
              {chain.block_number !== null && <Kv k="Block" v={String(chain.block_number)} />}
              {chain.gas_used !== null && <Kv k="Gas used" v={String(chain.gas_used)} />}
              {chain.attestation_uid && <Kv k="Attestation UID" v={chain.attestation_uid.slice(0, 18) + '…'} />}
              {chain.mode === 'onchain' && (
                <Kv k="Read-back" v={chain.readback_verified ? '✓ PASS' : '✗ FAIL'} />
              )}
            </div>
          )}

          {chain.readback_mismatches?.length > 0 && (
            <ul className="mt-2 text-xs text-danger list-disc list-inside">
              {chain.readback_mismatches.map((m) => <li key={m}>{m}</li>)}
            </ul>
          )}

          {chain.explorer_tx && (
            <a
              href={chain.explorer_tx}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline mt-2 block"
            >
              View transaction on the block explorer ↗
            </a>
          )}
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
        threshold {thresholdPct}% under recorded config. This record attests that the input image and its primary face
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
