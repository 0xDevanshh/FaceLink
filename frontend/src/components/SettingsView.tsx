import React, { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { HealthResponse } from '../types/api'

const THRESHOLDS = [
  { key: 'FACE_MATCH_THRESHOLD', default: '0.38', desc: 'Minimum ArcFace cosine similarity to pass FACE_MATCH rung' },
  { key: 'IMAGE_MATCH_THRESHOLD', default: '0.80', desc: 'Minimum pHash similarity to pass IMAGE_MATCH rung' },
  { key: 'VERIFY_MIN_SCORE', default: '0.70', desc: 'Minimum composite score for VERIFIED verdict' },
  { key: 'MAX_CANDIDATES_TO_VERIFY', default: '12', desc: 'Max candidates to fetch and re-measure per run' },
]

const WEIGHTS = [
  { key: 'WEIGHT_FACE', default: '0.50', desc: 'Face similarity weight in composite score' },
  { key: 'WEIGHT_IMAGE', default: '0.40', desc: 'Image similarity weight' },
  { key: 'WEIGHT_META', default: '0.10', desc: 'Metadata consistency weight (should sum to 1.0)' },
]

const BANDS = [
  { label: 'STRONG', range: '≥ 85%', color: 'text-success' },
  { label: 'MODERATE', range: '70–84%', color: 'text-accent' },
  { label: 'WEAK', range: '50–69%', color: 'text-warn' },
  { label: 'INSUFFICIENT', range: '< 50%', color: 'text-danger' },
]

export default function SettingsView() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch((e) => setHealthError(e?.message ?? 'Backend unreachable'))
  }, [])

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-accent mb-6">Settings & About</h1>

      {/* Backend health */}
      <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="health-heading">
        <h2 id="health-heading" className="text-sm font-semibold text-gray-300 mb-3">Backend Status</h2>
        {healthError ? (
          <div role="alert" className="text-danger text-sm">
            ✗ {healthError}
            <p className="text-muted text-xs mt-1">
              Start the backend: <code className="text-gray-300">uvicorn server:app --reload</code>
            </p>
          </div>
        ) : health ? (
          <div className="space-y-2 text-sm">
            <div className="flex gap-2">
              <span className="text-muted">Status</span>
              <span className="text-success">● {health.status}</span>
            </div>
            <div className="flex gap-2">
              <span className="text-muted">Version</span>
              <span className="font-mono text-gray-300">{health.version}</span>
            </div>
            <div className="flex gap-2">
              <span className="text-muted">Face backend</span>
              <span className="font-mono text-gray-300">{health.face_backend}</span>
            </div>
            <div className="mt-2">
              <span className="text-muted text-xs">Engines configured</span>
              <div className="flex flex-wrap gap-2 mt-1" role="list" aria-label="Configured engines">
                {Object.entries(health.engines_configured).map(([k, v]) => (
                  <span
                    key={k}
                    role="listitem"
                    className={`px-2 py-0.5 rounded border text-xs font-mono
                      ${v ? 'border-success/50 text-success' : 'border-border text-muted'}`}
                  >
                    {k}: {v ? 'yes' : 'no'}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-muted text-sm animate-pulse">Connecting to backend…</div>
        )}
      </section>

      {/* Thresholds */}
      <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="thresh-heading">
        <h2 id="thresh-heading" className="text-sm font-semibold text-gray-300 mb-3">
          Active Thresholds
          <span className="ml-2 text-xs text-muted font-normal">(override in .env)</span>
        </h2>
        <table className="w-full text-xs" aria-label="Threshold configuration">
          <thead>
            <tr className="text-muted border-b border-border">
              <th className="text-left pb-2 font-normal">Variable</th>
              <th className="text-left pb-2 font-normal">Default</th>
              <th className="text-left pb-2 font-normal">Description</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {[...THRESHOLDS, ...WEIGHTS].map((t) => (
              <tr key={t.key}>
                <td className="py-2 font-mono text-accent">{t.key}</td>
                <td className="py-2 font-mono text-gray-300">{t.default}</td>
                <td className="py-2 text-muted">{t.desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* Confidence bands */}
      <section className="mb-6 bg-surface-1 border border-border rounded p-4" aria-labelledby="bands-heading">
        <h2 id="bands-heading" className="text-sm font-semibold text-gray-300 mb-3">Confidence Bands</h2>
        <p className="text-xs text-muted mb-3">
          Plain-language labels for face similarity scores. Based on the input face vs. retrieved candidate face
          cosine similarity (ArcFace 512-D embeddings, L2-normalised).
        </p>
        <div className="space-y-2" role="list" aria-label="Confidence band definitions">
          {BANDS.map((b) => (
            <div key={b.label} role="listitem" className="flex items-center gap-4 text-sm">
              <span className={`font-bold font-mono w-24 shrink-0 ${b.color}`}>{b.label}</span>
              <span className="font-mono text-gray-300 w-20 shrink-0">{b.range}</span>
              <span className="text-muted text-xs">{bandMeaning(b.label)}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Scope statement */}
      <section className="border border-warn/30 bg-yellow-900/5 rounded p-4" aria-labelledby="scope-heading">
        <h2 id="scope-heading" className="text-sm font-semibold text-warn mb-2">Scope & Ethics</h2>
        <div className="text-xs text-muted space-y-2">
          <p>
            <strong className="text-gray-300">What a VERIFIED record claims:</strong>{' '}
            the supplied image and the face in it match an image retrieved from that public
            social-media post, under the thresholds recorded in the run's evidence bundle,
            where the post was found by genuine reverse-image search at that timestamp.
          </p>
          <p>
            <strong className="text-gray-300">What it does NOT claim:</strong>{' '}
            any person's real-world identity. This is not an identification system. It is not
            evidence about a person. It is evidence about images.
          </p>
          <p>
            <strong className="text-gray-300">Authorized use:</strong>{' '}
            your own photos, public figures for testing, or images you are authorized to investigate —
            with consent where consent is owed, and never to locate, profile, or harass a private individual.
          </p>
          <p>
            <strong className="text-gray-300">Privacy by design:</strong>{' '}
            face embeddings are never persisted in plaintext, never logged, never sent to third parties.
            Only the embedding SHA-256 is recorded. Raw biometric vectors never leave your machine.
          </p>
        </div>
      </section>
    </div>
  )
}

function bandMeaning(label: string): string {
  switch (label) {
    case 'STRONG': return 'Very high cosine similarity — strong evidence the face matches'
    case 'MODERATE': return 'Above threshold — consistent with a match, not definitive'
    case 'WEAK': return 'Borderline — images may be degraded or heavily edited'
    case 'INSUFFICIENT': return 'Below meaningful threshold — cannot support a match claim'
    default: return ''
  }
}
