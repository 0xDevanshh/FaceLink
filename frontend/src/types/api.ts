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

export interface VerifiedCandidate {
  engine: string
  url: string
  domain: string
  platform: string | null
  is_social: boolean
  fetched: boolean
  fetch_note: string
  candidate_image_url: string | null
  candidate_image_source: string
  candidate_image_sha256: string | null
  candidate_image_phash: string | null
  candidate_faces_found: number
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

export interface FaceRecord {
  detected: boolean
  backend: string
  model: string
  faces_found: number
  bbox: number[] | null
  det_score: number | null
  embedding_dimension: number | null
  embedding_sha256: string | null
}

export interface SearchReport {
  engines_attempted: string[]
  engines_succeeded: string[]
  engine_errors: Record<string, string>
  query_mode: Record<string, string>
  total_candidates: number
  social_candidates: number
}

export interface ChainRecord {
  network: string
  chain_id: number
  mode: string
  tx_hash: string | null
  attestation_uid: string | null
  explorer_attestation: string | null
  readback_verified: boolean
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
  reverse_search: SearchReport | null
  verification: VerifiedCandidate[]
  best_match: VerifiedCandidate | null
  stages_passed: string[]
  blockchain: ChainRecord | null
}

export interface HealthResponse {
  status: string
  version: string
  engines_configured: Record<string, boolean>
  face_backend: string
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
