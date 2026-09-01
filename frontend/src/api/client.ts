// Typed API client — all network calls go through here.
// Secrets never leave the backend; this client only handles public data.

import type {
  CaseResult,
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

  /** Subscribe to SSE stream. Returns an EventSource and a cleanup fn. */
  subscribeEvents(
    caseId: string,
    onEvent: (e: SSEEvent) => void,
    onDone: () => void,
    onError: (err: Event) => void,
  ): () => void {
    const es = new EventSource(`${BASE}/api/v1/scan/${encodeURIComponent(caseId)}/events`)
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as SSEEvent
        onEvent(data)
        if (data.stage === 'done' || data.stage === 'error') {
          es.close()
          onDone()
        }
      } catch {
        // ignore malformed events
      }
    }
    es.onerror = (err) => {
      es.close()
      onError(err)
    }
    return () => es.close()
  },
}
