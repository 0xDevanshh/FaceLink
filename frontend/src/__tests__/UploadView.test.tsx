import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import UploadView from '../components/UploadView'

// Mock the API client — no real backend needed.
//
// `detectFaces` returns one clearly usable face, i.e. the unambiguous case, so
// these tests exercise the straight-through path. The ambiguous path (which
// hands off to face selection instead) is covered in FaceSelectView.test.tsx.
// `vi.mock` is hoisted above the imports, so everything it references has to
// live inside the factory.
vi.mock('../api/client', () => ({
  api: {
    startScan: vi.fn().mockResolvedValue({
      case_id: 'case_20260901_000001_aabbccdd',
      status_url: '/api/v1/scan/case_20260901_000001_aabbccdd/status',
      events_url: '/api/v1/scan/case_20260901_000001_aabbccdd/events',
      result_url: '/api/v1/scan/case_20260901_000001_aabbccdd/result',
    }),
    detectFaces: vi.fn().mockResolvedValue({
      upload_id: 'upl_deadbeefdeadbeef',
      sha256: 'ab'.repeat(32),
      image_width: 800,
      image_height: 1200,
      faces: [{ index: 0, bbox: [100, 150, 300, 400], det_score: 0.94, face_px: 200, area: 50000, usable: true, note: '' }],
      auto_index: 0,
      selection_required: false,
      reason: '',
      quality: { passed: true, error: null, detail: '', blur_score: 120.5, face_px: 200, face_count: 1 },
    }),
    health: vi.fn().mockResolvedValue({ status: 'ok', version: '1.0.0', engines_configured: {} }),
    chainStatus: vi.fn().mockResolvedValue({
      network: 'ethereum-sepolia', network_name: 'Ethereum Sepolia', chain_id: 11155111,
      eas_contract: '0x' + '11'.repeat(20), signer_configured: true, rpc_reachable: true,
      schema_registered: true, schema_uid: '0x' + 'aa'.repeat(32),
      attester: '0x' + '22'.repeat(20), balance_eth: 0.05, ready: true, note: '',
    }),
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
