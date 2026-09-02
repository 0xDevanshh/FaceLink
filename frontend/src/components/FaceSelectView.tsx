import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FaceDetectResponse } from '../types/api'

/**
 * Choose which face in the uploaded photo the scan is about.
 *
 * Coordinate spaces are the whole problem here. The backend reports boxes in
 * its *working* image space (EXIF-oriented, downscaled to a maximum edge),
 * while the browser renders the original file at whatever size the layout
 * allows. Everything on screen is therefore stored in working-space
 * coordinates and multiplied by a single `scale` on the way out, so a box the
 * operator clicks and the crop the backend receives always describe the same
 * pixels.
 */

const MIN_CROP = 24 // working-space px; the backend enforces the real floor

export interface FaceChoice {
  faceIndex: number | null
  crop: [number, number, number, number] | null
  mode: 'auto' | 'manual-face' | 'manual-crop'
}

interface Props {
  previewUrl: string
  detection: FaceDetectResponse
  onConfirm: (choice: FaceChoice) => void
  onCancel: () => void
  busy?: boolean
}

type Rect = { x: number; y: number; w: number; h: number }

export default function FaceSelectView({
  previewUrl,
  detection,
  onConfirm,
  onCancel,
  busy = false,
}: Props) {
  const [selected, setSelected] = useState<number | null>(detection.auto_index)
  const [crop, setCrop] = useState<Rect | null>(null)
  const [displayWidth, setDisplayWidth] = useState(0)
  const imgRef = useRef<HTMLImageElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const dragStart = useRef<{ x: number; y: number } | null>(null)

  // One scale factor maps working space to what is on screen.
  const scale = displayWidth > 0 ? displayWidth / detection.image_width : 0

  const measure = useCallback(() => {
    if (imgRef.current) setDisplayWidth(imgRef.current.clientWidth)
  }, [])

  useEffect(() => {
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [measure])

  const toWorking = (clientX: number, clientY: number): { x: number; y: number } => {
    const box = imgRef.current?.getBoundingClientRect()
    if (!box || scale === 0) return { x: 0, y: 0 }
    return {
      x: Math.round((clientX - box.left) / scale),
      y: Math.round((clientY - box.top) / scale),
    }
  }

  const clampRect = (r: Rect): Rect => {
    const w = detection.image_width
    const h = detection.image_height
    const width = Math.min(Math.max(MIN_CROP, r.w), w)
    const height = Math.min(Math.max(MIN_CROP, r.h), h)
    return {
      x: Math.min(Math.max(0, r.x), w - width),
      y: Math.min(Math.max(0, r.y), h - height),
      w: width,
      h: height,
    }
  }

  // ---- drawing a crop ----------------------------------------------------

  // Drag state lives in a ref, not in state. A pointermove can arrive in the
  // same task as its pointerdown, before React has re-rendered, and a
  // state-based guard would still read `false` and drop the whole drag.
  const onPointerDown = (e: React.PointerEvent) => {
    if (busy) return
    const p = toWorking(e.clientX, e.clientY)
    dragStart.current = p
    setCrop({ x: p.x, y: p.y, w: MIN_CROP, h: MIN_CROP })
    setSelected(null)
    ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragStart.current) return
    const p = toWorking(e.clientX, e.clientY)
    const start = dragStart.current
    setCrop(
      clampRect({
        x: Math.min(start.x, p.x),
        y: Math.min(start.y, p.y),
        w: Math.abs(p.x - start.x),
        h: Math.abs(p.y - start.y),
      }),
    )
  }

  const onPointerUp = () => {
    dragStart.current = null
  }

  // ---- keyboard: nudge and resize the crop -------------------------------

  const nudge = (dx: number, dy: number, resize: boolean) => {
    setCrop((c) => {
      if (!c) return c
      const step = 8
      return resize
        ? clampRect({ ...c, w: c.w + dx * step, h: c.h + dy * step })
        : clampRect({ ...c, x: c.x + dx * step, y: c.y + dy * step })
    })
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    const map: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    }
    const delta = map[e.key]
    if (!delta) return
    // Deliberately not gated on the `crop` closure value: that value can be a
    // render behind the state, and the updater inside `nudge` already no-ops
    // when there is no crop to move.
    e.preventDefault()
    nudge(delta[0], delta[1], e.shiftKey)
  }

  const usableFaces = useMemo(() => detection.faces.filter((f) => f.usable).length, [detection.faces])

  const confirm = () => {
    if (crop) {
      onConfirm({ faceIndex: null, crop: [crop.x, crop.y, crop.w, crop.h], mode: 'manual-crop' })
    } else if (selected !== null) {
      onConfirm({
        faceIndex: selected,
        crop: null,
        mode: selected === detection.auto_index ? 'auto' : 'manual-face',
      })
    }
  }

  const canConfirm = !busy && (crop !== null || selected !== null)

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold text-accent mb-1">Select a face</h1>
      <p className="text-muted text-sm mb-4">
        {detection.selection_required
          ? detection.reason
          : 'Detection is unambiguous. Confirm the highlighted face, or pick a different one.'}
      </p>

      <div
        ref={wrapRef}
        className="relative inline-block max-w-full rounded border border-border overflow-hidden select-none"
        onKeyDown={onKeyDown}
        tabIndex={0}
        role="application"
        aria-label="Face selection canvas. Click a detected face, or drag to draw a crop. Arrow keys move the crop; hold Shift to resize."
      >
        <img
          ref={imgRef}
          src={previewUrl}
          alt="Uploaded photo with detected faces marked"
          onLoad={measure}
          className="block max-h-[60vh] w-auto max-w-full"
          draggable={false}
        />

        {/* Crop-drawing surface. Sits above the image, below the face buttons. */}
        <div
          className="absolute inset-0 cursor-crosshair"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          aria-hidden="true"
        />

        {/* Detected faces */}
        {scale > 0 &&
          detection.faces.map((f) => {
            const [x1, y1, x2, y2] = f.bbox
            const isSel = selected === f.index && !crop
            return (
              <button
                key={f.index}
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  setCrop(null)
                  setSelected(f.index)
                }}
                onPointerDown={(e) => e.stopPropagation()}
                title={f.note || `Face ${f.index + 1}`}
                aria-pressed={isSel}
                aria-label={`Face ${f.index + 1}: ${f.face_px} pixels, confidence ${f.det_score.toFixed(2)}${f.usable ? '' : ' — not usable'}`}
                className={`absolute border-2 transition-colors focus-visible:ring-2 focus-visible:ring-accent
                  ${isSel ? 'border-accent bg-accent/10' : f.usable ? 'border-success/70 hover:border-accent' : 'border-danger/70 hover:border-danger'}`}
                style={{
                  left: x1 * scale,
                  top: y1 * scale,
                  width: (x2 - x1) * scale,
                  height: (y2 - y1) * scale,
                }}
              >
                <span
                  className={`absolute -top-5 left-0 px-1 text-[10px] font-mono rounded-t
                    ${isSel ? 'bg-accent text-surface' : f.usable ? 'bg-success/80 text-surface' : 'bg-danger/80 text-white'}`}
                >
                  #{f.index + 1} {f.det_score.toFixed(2)}
                </span>
              </button>
            )
          })}

        {/* Operator-drawn crop */}
        {crop && scale > 0 && (
          <div
            className="absolute border-2 border-accent bg-accent/10 pointer-events-none"
            style={{ left: crop.x * scale, top: crop.y * scale, width: crop.w * scale, height: crop.h * scale }}
            aria-hidden="true"
          >
            <span className="absolute -top-5 left-0 px-1 text-[10px] font-mono bg-accent text-surface rounded-t">
              crop {crop.w}×{crop.h}
            </span>
          </div>
        )}
      </div>

      {/* Detection summary */}
      <dl className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 text-xs font-mono">
        <Stat label="Faces found" value={String(detection.faces.length)} />
        <Stat label="Usable" value={String(usableFaces)} />
        <Stat label="Image" value={`${detection.image_width}×${detection.image_height}`} />
        <Stat
          label="Quality gate"
          value={detection.quality.passed ? 'PASS' : (detection.quality.error ?? 'FAIL')}
          bad={!detection.quality.passed}
        />
      </dl>
      {!detection.quality.passed && detection.quality.detail && (
        <p className="mt-2 text-xs text-warn font-mono">{detection.quality.detail}</p>
      )}

      <p className="mt-4 text-xs text-muted">
        Click a box to pick that face, or drag on the image to draw your own crop.
        Arrow keys move the crop; hold Shift to resize it. Your original photo and its
        hash are kept unchanged in the evidence bundle either way.
      </p>

      <div className="mt-5 flex flex-wrap gap-3">
        <button
          onClick={confirm}
          disabled={!canConfirm}
          data-testid="confirm-face-btn"
          className="px-5 py-2 rounded bg-accent text-surface font-bold hover:bg-accent-dim
            transition-colors disabled:opacity-40 disabled:cursor-not-allowed
            focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-accent"
        >
          {busy ? 'Starting scan…' : crop ? 'Scan this crop →' : 'Scan this face →'}
        </button>
        {crop && (
          <button
            onClick={() => { setCrop(null); setSelected(detection.auto_index) }}
            className="px-4 py-2 rounded border border-border text-sm text-muted hover:text-gray-100
              focus-visible:ring-2 focus-visible:ring-accent"
          >
            Clear crop
          </button>
        )}
        <button
          onClick={onCancel}
          disabled={busy}
          data-testid="cancel-face-btn"
          className="px-4 py-2 rounded border border-border text-sm text-muted hover:text-gray-100
            focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-40"
        >
          Choose a different photo
        </button>
      </div>
    </div>
  )
}

function Stat({ label, value, bad }: { label: string; value: string; bad?: boolean }) {
  return (
    <div>
      <dt className="text-muted uppercase tracking-wider text-[10px]">{label}</dt>
      <dd className={bad ? 'text-danger' : 'text-gray-100'}>{value}</dd>
    </div>
  )
}
