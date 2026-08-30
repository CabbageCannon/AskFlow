export type MessageStatus = "pending"
  | "completed"
  | "generating"
  | "failed"
  | "aborted";

export type ResearchProgress={
  // 当前智能体运行状态
  stage:
    |"understanding"
    |"planning"
    |"researching"
    |"writing",
  // 运行状态对应图标
  label:string
}

export type Message = {
  id: string,
  requestId?: string,
  role: "user" | "assistant",
  content: string,
  status: MessageStatus,
  createdAt: number,
  error?: string,

  researchProgress?:ResearchProgress
}

export type Session = {
  id: string,
  title: string,
  messages: Message[],
  createdAt: string
}