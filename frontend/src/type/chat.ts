export type MessageStatus = "pending"
  | "completed"
  | "generating"
  | "failed"
  | "aborted";

export type Message = {
  id: string,
  requestId?: string,
  role: "user" | "assistant",
  content: string,
  status: MessageStatus,
  createdAt: number,
  error?: string
}

export type Session = {
  id: string,
  title: string,
  messages: Message[],
  createdAt: string
}