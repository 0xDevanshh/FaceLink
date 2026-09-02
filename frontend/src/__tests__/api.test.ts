import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { api, ApiError } from '../api/client'

// Mock global fetch
const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  })
}

describe('api client', () => {
  beforeEach(() => mockFetch.mockReset())

  it('health() parses response', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: 'ok', version: '1.0.0', engines_configured: {}, face_backend: 'auto' }))
    const h = await api.health()
    expect(h.status).toBe('ok')
    expect(h.version).toBe('1.0.0')
  })

  it('throws ApiError on non-2xx', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ detail: 'not found' }, 404))
    await expect(api.health()).rejects.toBeInstanceOf(ApiError)
  })

  it('ApiError carries status code', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({}, 413))
    try {
      await api.startScan(new FormData())
    } catch (e) {
      expect((e as ApiError).status).toBe(413)
    }
  })

  it('getEvidenceUrl returns correct path', () => {
    const url = api.getEvidenceUrl('case_20260901_000001')
    expect(url).toContain('case_20260901_000001')
    expect(url).toContain('/evidence')
  })

  it('getStatus calls correct endpoint', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ case_id: 'x', status: 'done', event_count: 5, error: null }))
    const s = await api.getStatus('case_x')
    expect(s.status).toBe('done')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('case_x'),
      expect.any(Object),
    )
  })

  it('subscribeEvents returns a cleanup function', () => {
    const mockClose = vi.fn()
    const mockES: any = {
      onmessage: null, onerror: null, close: mockClose,
      addEventListener: vi.fn(),
    }
    vi.stubGlobal('EventSource', vi.fn(() => mockES))
    const cleanup = api.subscribeEvents('case_x', vi.fn(), vi.fn(), vi.fn())
    expect(typeof cleanup).toBe('function')
    cleanup()
    expect(mockClose).toHaveBeenCalled()
    vi.unstubAllGlobals()
  })

  it('subscribeEvents fires onEvent for each message', () => {
    const onEvent = vi.fn()
    const mockES: any = { onmessage: null as any, onerror: null, close: vi.fn(), addEventListener: vi.fn() }
    vi.stubGlobal('EventSource', vi.fn(() => mockES))
    api.subscribeEvents('case_x', onEvent, vi.fn(), vi.fn())
    mockES.onmessage({ data: JSON.stringify({ stage: 'input', status: 'ok', detail: 'done', ts: '' }) })
    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ stage: 'input' }))
    vi.unstubAllGlobals()
  })

  it('subscribeEvents calls onDone when stage=done', () => {
    const onDone = vi.fn()
    const mockES: any = { onmessage: null as any, onerror: null, close: vi.fn(), addEventListener: vi.fn() }
    vi.stubGlobal('EventSource', vi.fn(() => mockES))
    api.subscribeEvents('case_x', vi.fn(), onDone, vi.fn())
    mockES.onmessage({ data: JSON.stringify({ stage: 'done', status: 'ok', detail: '', ts: '' }) })
    expect(onDone).toHaveBeenCalled()
    vi.unstubAllGlobals()
  })
})
