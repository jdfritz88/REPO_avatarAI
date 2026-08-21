export interface Avatar {
  id: string
  name: string
  status: 'ready' | 'processing' | 'failed' | 'pending'
  thumbnail_url?: string
  image_url?: string
  idle_video_url?: string | null
  idle_playlist_urls?: string[] | null
  expression_photos?: ExpressionPhoto[] | null
  llm_credential_id?: string | null
  s3_key?: string
  voice_id?: string | null
  avatar_metadata?: {
    system_prompt?: string
    personality?: string
    background_color?: string
    animation_style?: string
  }
  created_at?: string
}

// Raw LivePortrait ExpressionEditor node inputs — ranges match
// backend/app/schemas.py's ExpressionParams exactly (transcribed from
// ComfyUI-AdvancedLivePortrait/nodes.py's INPUT_TYPES, not guessed).
export interface ExpressionParams {
  rotate_pitch: number  // -20..20
  rotate_yaw: number    // -20..20
  rotate_roll: number   // -20..20
  blink: number          // -20..5 (negative closes eyes)
  eyebrow: number        // -10..15 (negative furrows, positive raises)
  wink: number            // 0..25 (right-eye-only blink)
  pupil_x: number        // -15..15
  pupil_y: number        // -15..15
  aaa: number              // -30..120 ("ah" mouth-open amount; 0 = closed)
  eee: number              // -20..15 ("ee"-shape mouth pull)
  woo: number              // -20..15 ("oo"-shape mouth pucker)
  smile: number            // -0.3..1.3 (negative = displeasure/frown, positive = smile)
  src_ratio: number      // 0..1 (how much of the source's own expression is kept)
  crop_factor: number    // 1.5..2.5 (face-crop zoom factor)
}

export type LlmProvider = 'anthropic' | 'mistral' | 'openai' | 'kindroid' | 'custom'

export interface LlmCredential {
  id: string
  provider: LlmProvider | string
  label: string
  api_key_masked: string
  api_base_url?: string | null
  kindroid_ai_id?: string | null
  is_favorite: boolean
  sort_order: number
  last_verified_at?: string | null
  last_verify_ok?: boolean | null
  last_verify_message?: string | null
  created_at: string
}

export interface ExpressionPhoto {
  id: string
  label: string
  key: string
  url: string
  created_at: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  created_at: string
  // Set only for multi-agent turns — see SpeakerTag.
  participant_id?: string
  participant_name?: string
}

export interface SessionSummary {
  id: string
  user_id: string
  avatar_id: string
  status: 'active' | 'paused' | 'ended'
  started_at: string
  ended_at?: string | null
}

export type WsMessageType =
  | 'token'
  | 'transcription'
  | 'message'
  | 'video_chunk_start'
  | 'video_chunk'
  | 'video_chunk_end'
  | 'status'
  | 'error'
  | 'pong'
  | 'tts_fallback'
  | 'interrupted'
  | 'watchdog_alert'

// Speaker attribution — present only on events from a multi-agent turn
// (Session has active `participants` configured); absent/undefined on the
// original single-LLM pipeline, so existing single-avatar chats render
// exactly as before with no speaker label.
export interface SpeakerTag {
  participant_id?: string
  participant_name?: string
}

// Discriminated union — each WS event has a well-typed payload so the handler
// can rely on field presence without optional-chaining everywhere.
export type WsMessage =
  | { type: 'token'; token: string }
  | { type: 'transcription'; text: string }
  | ({ type: 'message'; role: 'assistant'; content: string } & SpeakerTag)
  | ({ type: 'video_chunk_start'; total_chunks: number } & SpeakerTag)
  | ({
      type: 'video_chunk'
      chunk_index: number
      total_chunks: number
      video_url: string
      text: string
    } & SpeakerTag)
  | ({ type: 'video_chunk_end'; sent_chunks: number } & SpeakerTag)
  | { type: 'status'; message: string; stage?: string }
  | ({ type: 'error'; message: string } & SpeakerTag)
  | { type: 'pong' }
  | { type: 'tts_fallback'; engine: string; voice_cloned: boolean; message: string }
  | { type: 'interrupted'; message: string }
  | { type: 'watchdog_alert'; severity: 'warning' | 'error'; message: string }

// A selectable multi-agent participant (GET /api/v1/llm/participants).
export interface Participant {
  id: string
  type: 'anthropic' | 'openai_compat' | 'kindroid'
  name: string
  model: string | null
}

export type TurnMode = 'round_robin' | 'human_directed' | 'free_form'

export interface VoiceApiResponse {
  id: string
  name: string
  language: string
  duration: number
  created_at?: string
}

export interface ApiError {
  response?: {
    data?: {
      detail?: string
    }
  }
  message?: string
}
