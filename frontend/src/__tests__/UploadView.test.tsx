import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import UploadView from '../components/UploadView'

// Mock the API client — no real backend needed.
vi.mock('../api/client', () => ({
  api: {
    startScan: vi.fn().mockResolvedValue({
      case_id: 'case_20260901_000001_aabbccdd',
      status_url: '/api/v1/scan/case_20260901_000001_aabbccdd/status',
      events_url: '/api/v1/scan/case_20260901_000001_aabbccdd/events',
      result_url: '/api/v1/scan/case_20260901_000001_aabbccdd/result',
    }),
    health: vi.fn().mockResolvedValue({ status: 'ok', version: '1.0.0', engines_configured: {} }),
  },
  ApiError: class ApiError extends Error {
    constructor(public status: number, msg: string) { super(msg) }
  },
}))

function makeJpeg(): File {
  const bytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, ...new Array(200).fill(0)])
  return new File([bytes], 'face.jpg', { type: 'image/jpeg' })
}

describe('UploadView', () => {
  let onScanStarted: ReturnType<typeof vi.fn>

  beforeEach(() => {
    onScanStarted = vi.fn()
  })

  it('renders drag-and-drop zone and start button', () => {
    render(<UploadView onScanStarted={onScanStarted} />)
    expect(screen.getByLabelText(/drop zone/i)).toBeInTheDocument()
    expect(screen.getByTestId('start-scan-btn')).toBeDisabled()
  })

  it('shows error when trying to start without a file', async () => {
    render(<UploadView onScanStarted={onScanStarted} />)
    const btn = screen.getByTestId('start-scan-btn')
    // Without declaration and file: disabled
    expect(btn).toBeDisabled()
    await userEvent.click(screen.getByTestId('declaration-checkbox'))
    // Without file: still disabled
    expect(btn).toBeDisabled()
    // Providing unsupported type triggers the validation error div
    const badFile = new File(['not-image'], 'x.pdf', { type: 'application/pdf' })
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [badFile] } })
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('alert').textContent).toMatch(/unsupported type/i)
  })

  it('disables start button until both file and declaration are set', async () => {
    render(<UploadView onScanStarted={onScanStarted} />)
    const btn = screen.getByTestId('start-scan-btn')
    expect(btn).toBeDisabled()

    // Check declaration only
    await userEvent.click(screen.getByTestId('declaration-checkbox'))
    expect(btn).toBeDisabled()
  })

  it('rejects files over 10MB', () => {
    render(<UploadView onScanStarted={onScanStarted} />)
    const big = new Uint8Array(11 * 1024 * 1024 + 10).fill(0xff)
    big[0] = 0xff; big[1] = 0xd8; big[2] = 0xff
    const bigFile = new File([big], 'huge.jpg', { type: 'image/jpeg' })

    const input = screen.getByTestId('file-input')
    fireEvent.change(input, { target: { files: [bigFile] } })
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByRole('alert').textContent).toMatch(/too large/i)
  })

  it('rejects non-image files', () => {
    render(<UploadView onScanStarted={onScanStarted} />)
    const txtFile = new File(['hello'], 'doc.txt', { type: 'text/plain' })
    const input = screen.getByTestId('file-input')
    fireEvent.change(input, { target: { files: [txtFile] } })
    expect(screen.getByRole('alert').textContent).toMatch(/unsupported type/i)
  })

  it('calls onScanStarted after valid upload + declaration', async () => {
    render(<UploadView onScanStarted={onScanStarted} />)
    const input = screen.getByTestId('file-input')
    fireEvent.change(input, { target: { files: [makeJpeg()] } })
    await userEvent.click(screen.getByTestId('declaration-checkbox'))
    await userEvent.click(screen.getByTestId('start-scan-btn'))
    await waitFor(() => expect(onScanStarted).toHaveBeenCalledWith('case_20260901_000001_aabbccdd'))
  })

  it('shows error message if API call fails', async () => {
    const { api } = await import('../api/client')
    vi.mocked(api.startScan).mockRejectedValueOnce(new Error('network error'))

    render(<UploadView onScanStarted={onScanStarted} />)
    const input = screen.getByTestId('file-input')
    fireEvent.change(input, { target: { files: [makeJpeg()] } })
    await userEvent.click(screen.getByTestId('declaration-checkbox'))
    await userEvent.click(screen.getByTestId('start-scan-btn'))
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
  })
})
