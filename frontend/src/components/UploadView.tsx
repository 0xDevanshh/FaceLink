import React, { useCallback, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'

const MAX_MB = 10
const MAX_BYTES = MAX_MB * 1024 * 1024
const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp', 'image/avif'])

const ENGINES = [
  { id: 'yandex', label: 'Yandex Images', alwaysOn: true },
  { id: 'bing', label: 'Bing Visual Search', alwaysOn: true },
  { id: 'google_lens', label: 'Google Lens', alwaysOn: true },
  { id: 'tineye', label: 'TinEye', alwaysOn: false },
  { id: 'serpapi_google_lens', label: 'SerpAPI (Google Lens)', apiKey: 'serpapi' },
  { id: 'serpapi_yandex', label: 'SerpAPI (Yandex)', apiKey: 'serpapi' },
]

interface Props {
  onScanStarted: (caseId: string) => void
}

export default function UploadView({ onScanStarted }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [declaration, setDeclaration] = useState(false)
  const [selectedEngines, setSelectedEngines] = useState<Set<string>>(
    new Set(['yandex', 'bing', 'google_lens']),
  )
  const [noChain, setNoChain] = useState(true)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const validate = (f: File): string | null => {
    if (f.size > MAX_BYTES) return `File too large (${(f.size / 1024 / 1024).toFixed(1)}MB — max ${MAX_MB}MB)`
    if (!ALLOWED_TYPES.has(f.type)) return `Unsupported type: ${f.type}. Use JPEG, PNG, WebP, GIF, or BMP.`
    return null
  }

  const setFileWithPreview = (f: File) => {
    const err = validate(f)
    if (err) { setError(err); return }
    setError(null)
    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(f)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) setFileWithPreview(f)
  }, [])

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) setFileWithPreview(f)
  }

  const toggleEngine = (id: string) => {
    setSelectedEngines((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const submit = async () => {
    if (!file) { setError('Please select an image first.'); return }
    if (!declaration) { setError('Please confirm you are authorized to investigate this image.'); return }
    if (selectedEngines.size === 0) { setError('Select at least one search engine.'); return }

    setLoading(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('image', file)
      fd.append('engines', [...selectedEngines].join(','))
      fd.append('no_chain', noChain ? 'true' : 'false')
      fd.append('user_declaration', 'true')
      const res = await api.startScan(fd)
      onScanStarted(res.case_id)
    } catch (e) {
      setError(e instanceof ApiError ? `Server error ${e.status}: ${e.message}` : 'Network error — is the backend running?')
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-accent mb-2">Face Verification Pipeline</h1>
        <p className="text-muted text-sm">Upload a photo → find its social presence → verify locally → attest on-chain</p>
      </div>

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop zone for image upload"
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
          ${dragOver ? 'border-accent bg-surface-2' : 'border-border hover:border-accent/60 hover:bg-surface-1'}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif,image/bmp,image/avif"
          onChange={onFileChange}
          className="hidden"
          aria-hidden="true"
          data-testid="file-input"
        />
        {preview ? (
          <div className="flex flex-col items-center gap-3">
            <img
              src={preview}
              alt="Selected image preview"
              className="max-h-48 max-w-full rounded border border-border object-contain"
            />
            <span className="text-sm text-muted">{file?.name} ({((file?.size ?? 0) / 1024).toFixed(0)} KB)</span>
            <span className="text-xs text-accent">Click or drop to replace</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-muted">
            <span className="text-5xl" aria-hidden>📷</span>
            <span className="font-semibold text-gray-300">Drop image here or click to browse</span>
            <span className="text-xs">JPEG · PNG · WebP · GIF · BMP — max {MAX_MB}MB</span>
          </div>
        )}
      </div>

      {error && (
        <div role="alert" className="mt-3 text-danger text-sm px-3 py-2 bg-red-900/20 rounded border border-danger/30">
          {error}
        </div>
      )}

      {/* Engine selection */}
      <section className="mt-6" aria-labelledby="engine-heading">
        <h2 id="engine-heading" className="text-sm font-semibold text-gray-300 mb-2">Search Engines</h2>
        <div className="grid grid-cols-2 gap-2">
          {ENGINES.map((eng) => (
            <label
              key={eng.id}
              className={`flex items-center gap-2 px-3 py-2 rounded border text-sm cursor-pointer
                ${selectedEngines.has(eng.id) ? 'border-accent/60 bg-surface-2' : 'border-border bg-surface-1'}`}
            >
              <input
                type="checkbox"
                checked={selectedEngines.has(eng.id)}
                onChange={() => toggleEngine(eng.id)}
                className="accent-accent"
                aria-label={`Enable ${eng.label}`}
              />
              <span className={eng.apiKey ? 'text-muted' : ''}>{eng.label}</span>
              {eng.apiKey && (
                <span className="ml-auto text-xs text-warn" title={`Requires ${eng.apiKey.toUpperCase()}_KEY in .env`}>
                  API key
                </span>
              )}
            </label>
          ))}
        </div>
      </section>

      {/* Options */}
      <section className="mt-6 flex flex-col gap-3" aria-labelledby="options-heading">
        <h2 id="options-heading" className="text-sm font-semibold text-gray-300">Options</h2>
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={noChain}
            onChange={(e) => setNoChain(e.target.checked)}
            className="accent-accent"
            aria-label="Skip blockchain attestation"
          />
          <span>Skip blockchain attestation (--no-chain)</span>
          <span className="text-muted text-xs ml-1">— no wallet or gas needed</span>
        </label>
      </section>

      {/* Authorization declaration */}
      <section className="mt-6">
        <label
          className={`flex items-start gap-3 px-4 py-3 rounded border cursor-pointer transition-colors
            ${declaration ? 'border-success/60 bg-green-900/10' : 'border-border hover:border-success/30'}`}
        >
          <input
            type="checkbox"
            checked={declaration}
            onChange={(e) => setDeclaration(e.target.checked)}
            className="mt-0.5 accent-success"
            aria-label="Authorization declaration"
            data-testid="declaration-checkbox"
          />
          <span className="text-sm text-muted">
            <span className="text-gray-200 font-semibold">I confirm</span> this is my own photo, a public figure's photo,
            or an image I am authorized to investigate. I understand this tool makes no identity claims and is
            for evidence-based forensic matching only.
          </span>
        </label>
      </section>

      {/* Submit */}
      <button
        onClick={submit}
        disabled={!file || !declaration || loading}
        className="mt-6 w-full py-3 rounded bg-accent text-surface font-bold text-base
          hover:bg-accent-dim transition-colors disabled:opacity-40 disabled:cursor-not-allowed
          focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-accent"
        data-testid="start-scan-btn"
      >
        {loading ? 'Starting scan…' : 'Start Scan →'}
      </button>

      <p className="mt-4 text-xs text-muted text-center">
        Pipeline runs locally. No data leaves your machine except to configured search engine APIs.
      </p>
    </div>
  )
}
