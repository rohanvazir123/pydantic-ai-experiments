export interface Citation {
  id: string
  title: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
  status?: 'answered' | 'abstained_retrieval' | 'abstained_citation' | 'abstained_judge'
  low_confidence_warning?: boolean
  citations?: Citation[]
  cost_usd?: number
}
