// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

export interface Citation {
  chunk_id:         string
  document_title:   string
  document_source:  string
  relevance_score:  number
  excerpt:          string
}

export type PipelineStatus =
  | 'answered'
  | 'abstained_retrieval'
  | 'abstained_citation'
  | 'abstained_judge'

export interface Message {
  id:                     string
  role:                   'user' | 'assistant'
  content:                string
  citations?:             Citation[]
  status?:                PipelineStatus
  confidence?:            number
  low_confidence_warning?: boolean
  estimated_cost_usd?:    number
  model_tier_used?:       string
  prompt_tokens?:         number
  completion_tokens?:     number
  latency_ms?:            Record<string, number>
  cache_hit?:             string | null
  request_id?:            string
  trace_url?:             string | null
  abstention_layer?:      number
  abstention_reason?:     string
  streaming?:             boolean   // true while tokens are arriving
}

export interface Conversation {
  id:           string
  session_id:   string
  title?:       string
  summary?:     string
  turn_count:   number
  last_turn_at?: string
  messages:     Message[]
}
