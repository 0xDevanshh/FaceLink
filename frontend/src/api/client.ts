// Typed API client — all network calls go through here.
// Secrets never leave the backend; this client only handles public data.

/// <reference types="vite/client" />

import type {
  CaseResult,
  ChainStatusResponse,
  FaceDetectResponse,
  HealthResponse,
  ScanStartResponse,
  ScanStatusResponse,
  SSEEvent,
  VerifyResponse,
} from '../types/api'

const BASE = import.meta.env.VITE_API_BASE ?? ''

async function _fetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    ...init,
    headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText)
    throw new ApiError(res.status, msg)
  }
  return res.json() as Promise<T>
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export const api = {
  health(): Promise<HealthResponse> {
    return _fetch('/api/v1/health')
  },

  chainStatus(): Promise<ChainStatusResponse> {
    return _fetch('/api/v1/chain/status')
  },

  /**
   * Stage an upload and get back its detected faces.
   *
   * Returns an `upload_id` the scan reuses, so the photo crosses the wire once
   * and the bytes scanned are the bytes the boxes were computed from.
   */
  detectFaces(file: File): Promise<FaceDetectResponse> {
    const fd = new FormData()
    fd.append('image', file)
    return _fetch('/api/v1/faces', { method: 'POST', body: fd })
  },

  startScan(formData: FormData): Promise<ScanStartResponse> {
    return _fetch('/api/v1/scan', { method: 'POST', body: formData })
  },

  getStatus(caseId: string): Promise<ScanStatusResponse> {
    return _fetch(`/api/v1/scan/${encodeURIComponent(caseId)}/status`)
  },

  getResult(caseId: string): Promise<CaseResult> {
    return _fetch(`/api/v1/scan/${encodeURIComponent(caseId)}/result`)
  },

  getEvidenceUrl(caseId: string): string {
    return `${BASE}/api/v1/scan/${encodeURIComponent(caseId)}/evidence`
  },

  verifyEvidence(file: File): Promise<VerifyResponse> {
    const fd = new FormData()
    fd.append('evidence_zip', file)
    return _fetch('/api/v1/verify', { method: 'POST', body: fd })
  },

  /**
   * Open SSE stream. Returns a cleanup function.
   * The server tail-follows job.events — late connections get full replay.
   * onDone fires when stage==="done"|"error"; onError fires on connection failure.
   */
  subscribeEvents(
    caseId: string,
    onEvent: (e: SSEEvent) => void,
    onDone: () => void,
    onError: (err: Event) => void,
  ): () => void {
    const url = `${BASE}/api/v1/scan/${encodeURIComponent(caseId)}/events`
    const es  = new EventSource(url)

    es.addEventListener('open', () => {
      console.log('[SSE] connected', caseId)
    })

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as SSEEvent
        if (data.stage === 'ping') return   // keepalive — ignore
        onEvent(data)
        if (data.stage === 'done' || data.stage === 'error') {
          es.close()
          onDone()
        }
      } catch {
        /* ignore parse errors */
      }
    }

    es.onerror = (err) => {
      console.warn('[SSE] error', err)
      es.close()
      onError(err)
    }

    return () => es.close()
  },
}
