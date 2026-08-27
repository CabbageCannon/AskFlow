export type AiMessage = {
  role: "user" | "assistant" | "system",
  content: string
}

export type ChatRequest = {
  requestId: string,
  sessionId: string,
  messages: AiMessage[]
}

// SSE事件类型
export type DeltaEvent = {
  type: "delta",
  requestId: string,
  content: string
}

export type DoneEvent = {
  type: "done",
  requestId: string,
  finishReason: "stop" | "length"
}

export type ErrorEvent = {
  type: "error",
  requestId: string,
  code: string,
  message: string,
  retryable: boolean
}

export type ChatStreamEvent = DeltaEvent
  | DoneEvent
  | ErrorEvent