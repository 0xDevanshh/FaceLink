// Typed API contracts — mirrors the backend Pydantic models exactly.
// No dangerouslySetInnerHTML, no secrets, no embeddings.

export interface ScanStartResponse {
  case_id: string
  status_url: string
  events_url: string
  result_url: string
}

export interface ScanStatusResponse {
  case_id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  event_count: number
  error: string | null
}

export interface SSEEvent {
  stage: string
  status: string
  detail: string
  ts: string
}

/** Terminal lifecycle states a search provider can end in. */
export type ProviderStatus =
  | 'NOT_CONFIGURED'
  | 'READY'
  | 'STARTED'
  | 'COMPLETED'
  | 'NO_RESULTS'
  | 'TIMEOUT'
  | 'CHALLENGED'
  | 'RATE_LIMITED'
  | 'FAILED'
  | 'CANCELLED'

export type CandidateType =
  | 'EXACT_IMAGE'
  | 'SAME_FACE'
  | 'SOCIAL_PROFILE'
  | 'SOCIAL_POST'
  | 'DEVELOPER_PROFILE'
  | 'PUBLIC_ARTICLE'
  | 'PUBLIC_WEB_PAGE'
  | 'OTHER'

export interface ProviderReport {
  engine: string
  status: ProviderStatus
  candidates: number
  duration_s: number
  query_mode: string
  error: string
}

export interface DetectedFaceInfo {
  index: number
  bbox: number[] // x1, y1, x2, y2 in the working image's coordinate space
  det_score: number
  face_px: number
  area: number
  usable: boolean
  note: string
}

export interface FaceQuality {
  passed: boolean
  error: string | null
  detail: string
  blur_score: number
  face_px: number
  face_count: number
  // Graded metrics — informational, never a pass/fail gate on their own.
  det_score: number
  yaw_deg: number
  roll_deg: number
  brightness: number
  bands: Record<string, string> // metric -> GOOD | ACCEPTABLE | POOR | PASS | FAIL
  overall_quality: number // 0..1
}

export interface FaceDetectResponse {
  upload_id: string
  sha256: string
  image_width: number
  image_height: number
  faces: DetectedFaceInfo[]
  auto_index: number | null
  selection_required: boolean
  reason: string
  quality: FaceQuality
}

export interface FaceSelection {
  mode: string
  face_index: number | null
  faces_offered: number
  bbox: number[] | null
  crop_rect: number[] | null
  crop_sha256: string | null
  original_sha256: string
  original_width: number
  original_height: number
  selected_at: string
}

export interface ChainStatusResponse {
  network: string
  network_name: string
  chain_id: number
  eas_contract: string
  signer_configured: boolean
  rpc_reachable: boolean
  schema_registered: boolean
  schema_uid: string | null
  attester: string | null
  balance_eth: number | null
  ready: boolean
  note: string
}

export interface VerifiedCandidate {
  engine: string
  url: string
  domain: string
  platform: string | null
  is_social: boolean
  canonical_url: string
  platform_priority: number
  candidate_type: CandidateType
  fetched: boolean
  fetch_note: string
  candidate_image_url: string | null
  candidate_image_source: string
  candidate_image_sha256: string | null
  candidate_image_phash: string | null
  candidate_faces_found: number
  // Which of `candidate_faces_found` produced `face_similarity`, and how good
  // that face was — essential for group photos, where "a face matched" means
  // nothing without saying which one and how reliably.
  candidate_face_index: number
  candidate_face_quality: number // 0..1 graded quality of the matched face
  candidate_face_bands: Record<string, string>
  found_via_variant: string // "" for the historical single-image search path
  image_similarity: number
  face_detected: boolean
  face_similarity: number
  metadata_consistency: number
  confidence_band: string
  stages: string[]
  match_type: string
  final_score: number
  verified: boolean
  rejection_reason: string
}

export interface SearchVariantReport {
  variant_id: string
  variant_type: string // original | tight_crop | loose_crop
  sha256: string
  width: number
  height: number
  candidates_found: number
  skipped: boolean
  skip_reason: string
}

export interface EvidenceGraphNode {
  id: string
  type: string // candidate | image | domain | platform
  label: string
}

export interface EvidenceGraphEdge {
  source: string
  target: string
  type: string // same_image | same_domain | same_platform | same_face | independent_source
  note: string
}

export interface EvidenceGraphReport {
  nodes: EvidenceGraphNode[]
  edges: EvidenceGraphEdge[]
  independent_evidence_count: number
}

export interface ThresholdSnapshot {
  face_match_threshold: number
  image_match_threshold: number
  verify_min_score: number
  weight_face: number
  weight_image: number
  weight_meta: number
  insightface_model: string
  face_backend: string
  calibration_status: string // DEFAULT | CALIBRATED | CALIBRATION_INSUFFICIENT
  calibration_note: string
}

export interface FaceRecord {
  detected: boolean
  backend: string
  model: string
  faces_found: number
  bbox: number[] | null
  det_score: number | null
  embedding_dimension: number | null
  embedding_sha256: string | null
  faces: DetectedFaceInfo[]
  quality: FaceQuality | null
}

export interface SearchReport {
  engines_attempted: string[]
  engines_succeeded: string[]
  engine_errors: Record<string, string>
  query_mode: Record<string, string>
  providers: ProviderReport[]
  total_candidates: number
  social_candidates: number
  platform_counts: Record<string, number>
  timed_out: boolean
  variants: SearchVariantReport[] // empty for the historical single-image search path
}

export interface ChainRecord {
  network: string
  chain_id: number
  eas_contract: string
  schema_uid: string
  attester: string
  mode: string
  tx_hash: string | null
  block_number: number | null
  gas_used: number | null
  attestation_uid: string | null
  explorer_tx: string | null
  explorer_attestation: string | null
  readback_verified: boolean
  readback_mismatches: string[]
  note: string
}

export interface CaseResult {
  case_id: string
  pipeline_version: string
  created_at: string
  verdict: string
  failure_reason: string | null
  evidence_sha256: string | null
  face: FaceRecord | null
  face_selection: FaceSelection | null
  reverse_search: SearchReport | null
  verification: VerifiedCandidate[]
  best_match: VerifiedCandidate | null
  evidence_graph: EvidenceGraphReport | null
  threshold_snapshot: ThresholdSnapshot | null
  stages_passed: string[]
  blockchain: ChainRecord | null
}

export interface HealthResponse {
  status: string
  version: string
  engines_configured: Record<string, boolean>
  face_backend: string
  chain_configured: boolean
  network: string
  chain_id: number
  priority_platforms: string[]
}

export interface VerifyCheck {
  check: string
  passed: boolean
  detail?: string
}

export interface VerifyResponse {
  overall: 'PASS' | 'FAIL'
  checks: VerifyCheck[]
}
