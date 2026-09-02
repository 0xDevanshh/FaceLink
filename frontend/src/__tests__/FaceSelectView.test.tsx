import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import FaceSelectView from '../components/FaceSelectView'
import type { FaceDetectResponse } from '../types/api'

/**
 * The contract this view has to keep is coordinate fidelity: whatever the
 * operator picks must be handed back in the backend's *working* image space,
 * because that is the space the boxes arrived in and the space the crop will be
 * applied in. A scale mismatch here would crop the wrong part of the photo.
 */

function detection(overrides: Partial<FaceDetectResponse> = {}): FaceDetectResponse {
  return {
    upload_id: 'upl_abcdef0123456789',
    sha256: 'ab'.repeat(32),
    image_width: 600,
    image_height: 800,
    faces: [
      { index: 0, bbox: [50, 60, 250, 300], det_score: 0.93, face_px: 200, area: 48000, usable: true, note: '' },
      { index: 1, bbox: [320, 70, 500, 290], det_score: 0.88, face_px: 180, area: 39600, usable: true, note: '' },
    ],
    auto_index: null,
    selection_required: true,
    reason: '2 faces of comparable size were detected — choose which one the scan is about.',
    quality: { passed: true, error: null, detail: '', blur_score: 130.2, face_px: 200, face_count: 2 },
    ...overrides,
  }
}

/**
 * jsdom has no `PointerEvent`, so `fireEvent.pointerDown` falls back to a plain
 * Event and silently drops `clientX`/`clientY` — every coordinate would arrive
 * as NaN. Dispatching a MouseEvent under the pointer event's type keeps React's
 * onPointer* handlers wired up *and* carries real coordinates.
 */
function pointer(el: Element, type: string, x: number, y: number) {
  // Wrapped in act so the resulting state update is flushed before the next
  // query — otherwise assertions race the render.
  act(() => {
    el.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: x, clientY: y }))
  })
}

/** jsdom gives images zero width; pin a display size so `scale` is computable. */
function pinImageWidth(px: number) {
  Object.defineProperty(HTMLImageElement.prototype, 'clientWidth', {
    configurable: true,
    get() { return px },
  })
  HTMLElement.prototype.getBoundingClientRect = function () {
    return { left: 0, top: 0, width: px, height: px * (800 / 600), right: px, bottom: px, x: 0, y: 0, toJSON: () => ({}) } as DOMRect
  }
}

describe('FaceSelectView', () => {
  let onConfirm: ReturnType<typeof vi.fn>
  let onCancel: ReturnType<typeof vi.fn>

  beforeEach(() => {
    onConfirm = vi.fn()
    onCancel = vi.fn()
    pinImageWidth(600) // 1:1 with the working space, so scale === 1
  })

  const view = (d = detection()) =>
    render(
      <FaceSelectView previewUrl="blob:preview" detection={d} onConfirm={onConfirm} onCancel={onCancel} />,
    )

  it('explains why a selection is needed', () => {
    view()
    expect(screen.getByText(/comparable size/i)).toBeInTheDocument()
  })

  it('renders one clickable box per detected face', () => {
    view()
    expect(screen.getByRole('button', { name: /Face 1:/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Face 2:/ })).toBeInTheDocument()
  })

  it('cannot confirm until a face is chosen', async () => {
    view()
    expect(screen.getByTestId('confirm-face-btn')).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: /Face 2:/ }))
    expect(screen.getByTestId('confirm-face-btn')).toBeEnabled()
  })

  it('returns the chosen face index as a manual selection', async () => {
    view()
    await userEvent.click(screen.getByRole('button', { name: /Face 2:/ }))
    await userEvent.click(screen.getByTestId('confirm-face-btn'))
    expect(onConfirm).toHaveBeenCalledWith({ faceIndex: 1, crop: null, mode: 'manual-face' })
  })

  it('records an auto-selected face as auto, not as a manual choice', async () => {
    view(detection({ auto_index: 0, selection_required: false }))
    await userEvent.click(screen.getByTestId('confirm-face-btn'))
    // Evidence must not claim a human made a choice they did not make.
    expect(onConfirm).toHaveBeenCalledWith({ faceIndex: 0, crop: null, mode: 'auto' })
  })

  it('pre-selects the auto-chosen face when detection is unambiguous', () => {
    view(detection({ auto_index: 1, selection_required: false }))
    expect(screen.getByRole('button', { name: /Face 2:/ })).toHaveAttribute('aria-pressed', 'true')
  })

  it('converts a dragged crop into working-space coordinates', async () => {
    const { container } = view()
    const surface = container.querySelector('.cursor-crosshair')!

    pointer(surface, 'pointerdown', 100, 120)
    pointer(surface, 'pointermove', 340, 420)
    pointer(surface, 'pointerup', 0, 0)

    await userEvent.click(screen.getByTestId('confirm-face-btn'))
    expect(onConfirm).toHaveBeenCalledWith({
      faceIndex: null,
      crop: [100, 120, 240, 300],
      mode: 'manual-crop',
    })
  })

  it('scales a dragged crop back up when the image is displayed smaller', async () => {
    pinImageWidth(300) // half size: scale === 0.5
    const { container } = view()
    const surface = container.querySelector('.cursor-crosshair')!

    pointer(surface, 'pointerdown', 50, 60)
    pointer(surface, 'pointermove', 170, 210)
    pointer(surface, 'pointerup', 0, 0)

    await userEvent.click(screen.getByTestId('confirm-face-btn'))
    // On-screen 120x150 at half scale is 240x300 in the backend's space.
    expect(onConfirm).toHaveBeenCalledWith({
      faceIndex: null,
      crop: [100, 120, 240, 300],
      mode: 'manual-crop',
    })
  })

  it('clamps a crop dragged past the edge of the image', async () => {
    const { container } = view()
    const surface = container.querySelector('.cursor-crosshair')!

    pointer(surface, 'pointerdown', 500, 700)
    pointer(surface, 'pointermove', 2000, 3000)
    pointer(surface, 'pointerup', 0, 0)

    await userEvent.click(screen.getByTestId('confirm-face-btn'))
    const [{ crop }] = onConfirm.mock.calls[0]
    const [x, y, w, h] = crop!
    expect(x + w).toBeLessThanOrEqual(600)
    expect(y + h).toBeLessThanOrEqual(800)
  })

  it('lets a crop be cleared and a face picked instead', async () => {
    const { container } = view()
    const surface = container.querySelector('.cursor-crosshair')!
    pointer(surface, 'pointerdown', 100, 120)
    pointer(surface, 'pointermove', 300, 350)
    pointer(surface, 'pointerup', 0, 0)

    await userEvent.click(screen.getByRole('button', { name: /clear crop/i }))
    await userEvent.click(screen.getByRole('button', { name: /Face 1:/ }))
    await userEvent.click(screen.getByTestId('confirm-face-btn'))
    expect(onConfirm).toHaveBeenCalledWith({ faceIndex: 0, crop: null, mode: 'manual-face' })
  })

  it('moves the crop with the arrow keys and resizes it with shift', async () => {
    const { container } = view()
    const surface = container.querySelector('.cursor-crosshair')!
    pointer(surface, 'pointerdown', 100, 120)
    pointer(surface, 'pointermove', 300, 350)
    pointer(surface, 'pointerup', 0, 0)

    const canvas = screen.getByRole('application')
    fireEvent.keyDown(canvas, { key: 'ArrowRight' })
    fireEvent.keyDown(canvas, { key: 'ArrowDown', shiftKey: true })

    await userEvent.click(screen.getByTestId('confirm-face-btn'))
    const [{ crop }] = onConfirm.mock.calls[0]
    expect(crop![0]).toBe(108)  // moved right by one step
    expect(crop![3]).toBe(238)  // grew taller by one step
  })

  it('marks an unusable face and says why', () => {
    view(detection({
      faces: [{ index: 0, bbox: [10, 10, 50, 50], det_score: 0.71, face_px: 40, area: 1600, usable: false, note: '40px is below the 80px minimum' }],
      auto_index: null,
    }))
    const box = screen.getByRole('button', { name: /Face 1:/ })
    expect(box.getAttribute('aria-label')).toMatch(/not usable/i)
  })

  it('surfaces a failing quality gate', () => {
    view(detection({
      quality: { passed: false, error: 'BLURRY', detail: 'Laplacian variance 12.3 < threshold 40.0', blur_score: 12.3, face_px: 200, face_count: 2 },
    }))
    expect(screen.getByText('BLURRY')).toBeInTheDocument()
    expect(screen.getByText(/Laplacian variance/)).toBeInTheDocument()
  })

  it('lets the operator back out to a different photo', async () => {
    view()
    await userEvent.click(screen.getByTestId('cancel-face-btn'))
    expect(onCancel).toHaveBeenCalled()
  })

  it('disables both actions while a scan is starting', () => {
    render(
      <FaceSelectView previewUrl="blob:p" detection={detection({ auto_index: 0 })}
        onConfirm={onConfirm} onCancel={onCancel} busy />,
    )
    expect(screen.getByTestId('confirm-face-btn')).toBeDisabled()
    expect(screen.getByTestId('cancel-face-btn')).toBeDisabled()
  })
})
