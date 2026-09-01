import React, { useRef, useState } from 'react'
import { api } from '../api/client'
import type { CaseResult, VerifyCheck, VerifyResponse } from '../types/api'

interface Props {
  caseId: string
  result: CaseResult | null
}

const FILES = [
  { name: 'case.json', desc: 'Full pipeline record' },
  { name: 'attested_payload.json', desc: 'Fields hashed into evidenceHash' },
  { name: 'attested_payload.sha256', desc: 'Canonical hash of the payload' },
  { name: 'input.sha256', desc: 'Input image hash (shasum -c compatible)' },
  { name: 'face_embedding.sha256', desc: 'Embedding hash (vector stays local)' },
  { name: 'matched_image.sha256', desc: 'Hash of retrieved candidate image' },
  { name: 'reverse_search.json', desc: 'All candidates from all engines' },
  { name: 'verification.json', desc: 'Per-candidate measurements' },
  { name: 'blockchain.json', desc: 'On-chain record (if attested)' },
  { name: 'attestation.txt', desc: 'Human-readable receipt' },
]

export default function EvidenceView({ caseId, result }: Props) {
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null)
  const [verifying, setVerifying] = useState(false)
  const [verifyError, setVerifyError] = useState<string | null>(null)
  const [expandedFile, setExpandedFile] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  async function onVerifyUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setVerifying(true)
    setVerifyError(null)
    setVerifyResult(null)
    try {
      const res = await api.verifyEvidence(f)
      setVerifyResult(res)
    } catch (err) {
      setVerifyError(err instanceof Error ? err.message : 'Verification failed')
    } finally {
      setVerifying(false)
      e.target.value = ''
    }
  }

  function renderJson(obj: unknown): string {
    return JSON.stringify(obj, null, 2)
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-accent">Evidence Bundle</h1>
          <p className="text-muted text-sm font-mono mt-1">evidence/{caseId}/</p>
        </div>
        <a
          href={api.getEvidenceUrl(caseId)}
          download={`${caseId}.zip`}
          className="px-4 py-2 rounded border border-accent/60 text-accent text-sm hover:bg-surface-2 transition-colors"
          aria-label="Download full evidence ZIP"
        >
          Download .zip
        </a>
      </div>

      {/* evidenceHash */}
      {result?.evidence_sha256 && (
        <div className="mb-6 bg-surface-1 border border-border rounded p-4">
          <h2 className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">evidenceHash</h2>
          <p className="font-mono text-xs text-gray-300 break-all">{result.evidence_sha256}</p>
          <p className="text-xs text-muted mt-1">
            SHA-256 of <code className="text-gray-300">attested_payload.json</code> (sorted keys, no whitespace).
            Re-hash this file on any machine to reproduce the value.
          </p>
        </div>
      )}

      {/* File tree */}
      <section className="mb-6" aria-labelledby="file-tree-heading">
        <h2 id="file-tree-heading" className="text-sm font-semibold text-gray-300 mb-3">Bundle Contents</h2>
        <div className="space-y-1" role="list" aria-label="Evidence files">
          {FILES.map((f) => (
            <div key={f.name} role="listitem">
              <button
                onClick={() => setExpandedFile((p) => p === f.name ? null : f.name)}
                className="w-full flex items-center gap-3 px-3 py-2 rounded hover:bg-surface-2 text-left transition-colors group"
                aria-expanded={expandedFile === f.name}
                aria-controls={`file-content-${f.name.replace('.', '-')}`}
              >
                <span className="font-mono text-accent text-xs shrink-0">{f.name}</span>
                <span className="text-muted text-xs">{f.desc}</span>
                <span
                  className={`ml-auto text-muted text-xs transition-transform group-hover:text-gray-300
                    ${expandedFile === f.name ? 'rotate-90' : ''}`}
                  aria-hidden
                >
                  ▶
                </span>
              </button>
              {expandedFile === f.name && (
                <div
                  id={`file-content-${f.name.replace('.', '-')}`}
                  className="mx-3 mb-2 bg-surface border border-border rounded overflow-x-auto"
                >
                  {f.name.endsWith('.json') && result ? (
                    <FileJson name={f.name} result={result} renderJson={renderJson} />
                  ) : (
                    <div className="p-4 text-xs text-muted font-mono">
                      Download the .zip to view this file.
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div role="listitem" className="px-3 py-1 text-xs text-muted">
            artifacts/input.jpg · artifacts/face_crop.png
          </div>
        </div>
      </section>

      {/* Verify section */}
      <section className="mb-6 border border-border rounded p-4 bg-surface-1" aria-labelledby="verify-heading">
        <h2 id="verify-heading" className="text-sm font-semibold text-gray-300 mb-1">
          Independent Verification
        </h2>
        <p className="text-xs text-muted mb-3">
          Upload your downloaded .zip to re-hash every file and check consistency
          (re-derives evidenceHash, validates URL↔hash pairs, does not trust blockchain.json).
        </p>

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={() => fileRef.current?.click()}
            disabled={verifying}
            className="px-4 py-2 rounded border border-accent/60 text-accent text-sm hover:bg-surface-2 transition-colors disabled:opacity-40"
            aria-label="Upload evidence ZIP to verify"
          >
            {verifying ? 'Verifying…' : 'Verify Evidence (.zip)'}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".zip"
            onChange={onVerifyUpload}
            className="hidden"
            aria-hidden="true"
          />
        </div>

        {verifyError && (
          <div role="alert" className="mt-3 text-xs text-danger bg-red-900/10 border border-danger/30 rounded px-3 py-2">
            {verifyError}
          </div>
        )}

        {verifyResult && (
          <VerifyReport result={verifyResult} />
        )}
      </section>

      {/* Tamper demo */}
      <section className="border border-warn/30 bg-yellow-900/5 rounded p-4" aria-labelledby="tamper-demo-heading">
        <h2 id="tamper-demo-heading" className="text-sm font-semibold text-warn mb-1">
          🔬 Tamper-Evidence Demo
        </h2>
        <p className="text-xs text-muted mb-2">
          To see tamper detection in action: download the .zip, open{' '}
          <code className="text-gray-300">attested_payload.json</code>, change any field value by one
          character, repack as .zip, then upload to "Verify Evidence" above. The verifier will fail
          with the exact field that was changed — this proves the evidence is tamper-evident.
        </p>
        <p className="text-xs text-warn">
          ⚠ Only modify a local copy — server-side evidence is never modified.
        </p>
      </section>
    </div>
  )
}

function FileJson({
  name,
  result,
  renderJson,
}: {
  name: string
  result: CaseResult
  renderJson: (o: unknown) => string
}) {
  let content: unknown = null
  if (name === 'case.json') content = result
  else if (name === 'reverse_search.json') content = result.reverse_search
  else if (name === 'verification.json') content = { candidates: result.verification }
  else if (name === 'blockchain.json') content = result.blockchain

  if (!content) {
    return (
      <div className="p-4 text-xs text-muted">
        Not available in browser — download .zip to view.
      </div>
    )
  }

  return (
    <pre className="p-4 text-xs text-gray-300 overflow-auto max-h-80 scrollbar-thin whitespace-pre-wrap">
      {renderJson(content)}
    </pre>
  )
}

function VerifyReport({ result }: { result: VerifyResponse }) {
  return (
    <div className="mt-4" aria-label="Verification report" role="region">
      <div className={`text-sm font-bold mb-2 ${result.overall === 'PASS' ? 'text-success' : 'text-danger'}`}>
        Overall: {result.overall}
      </div>
      <ul className="space-y-1" role="list" aria-label="Check results">
        {result.checks.map((c: VerifyCheck, i: number) => (
          <li key={i} className="flex items-start gap-2 text-xs">
            <span className={c.passed ? 'text-success' : 'text-danger'} aria-hidden>
              {c.passed ? '✓' : '✗'}
            </span>
            <span className={c.passed ? 'text-gray-300' : 'text-danger'}>
              {c.check}
              {c.detail && <span className="text-muted ml-1">— {c.detail}</span>}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
