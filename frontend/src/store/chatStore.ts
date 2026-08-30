import { create } from "zustand";
import type { Session, Message,ResearchProgress } from "@/type/chat.ts";
import { persist } from "zustand/middleware";

export type GenerationTask = {
  sessionId: string,
  messageId: string,
  requestId: string
}

type ChatStore = {
  sessions: Session[],
  currentSession: string,
  isAtBottom: boolean,

  ensureCurrentSession: () => string,
  createSession: () => void,
  selectSession: (sessionId: string) => void,
  deleteSession: (sessionIdL: string) => void,

  startGeneration: (sessionId: string, content: string) => GenerationTask | null,
  appendMessageChunk: (sessionId: string, messageId: string, chunk: string, requestId: string) => void,
  completeMessage: (sessionId: string, messageId: string, requestId: string) => void,
  abortMessage: (sessionId: string, messageId: string, requestId: string) => void,
  failMessage: (sessionId: string, messageId: string, requestId: string, error: string) => void,
  prepareRetryMessage: (sessionId: string, messageId: string) => GenerationTask | null,
  prepareRegenerateMessage: (sessionId: string, messageId: string) => GenerationTask | null,
  createMessage: (sessionId: string, newMessage: Message) => void,
  updateMessage: (sessionId: string, messageId: string, updateContent: string) => void,
  // 设置智能体运行状态
  setResearchProgress:(sessionId:string,messageId:string,requestId:string,progress:ResearchProgress | undefined)=>void

  setIsAtBottom: (bottom: boolean) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      sessions: [],
      currentSession: "" as string,
      isAtBottom: true,

      // 会话action
      ensureCurrentSession: () => {
        let newSessionId = crypto.randomUUID()
        let ensureCurrentSessionId = "";

        set(state => {
          const currentSessionExist = state.sessions.find(session => session.id === state.currentSession);

          // 当前指向的对话存在
          if (state.currentSession && currentSessionExist) {
            ensureCurrentSessionId = state.currentSession;
            return state;
          }

          // 当前指向的对话不存在，但对话历史不为0
          if (state.sessions.length > 0) {
            // 把第一个对话放到当前对话
            ensureCurrentSessionId = state.sessions[0].id;
            return {
              currentSession: ensureCurrentSessionId,
            }
          }

          // 历史对话也为0
          const newSession = {
            id: newSessionId,
            title: "新对话",
            messages: [],
            createdAt: Date.now().toString(),
          }

          ensureCurrentSessionId = newSessionId;

          return {
            sessions: [newSession],
            currentSession: ensureCurrentSessionId,
          }
        })

        return ensureCurrentSessionId;
      },
      createSession: () => {
        const newSessionId = crypto.randomUUID();
        const newSession: Session = {
          id: newSessionId,
          messages: [],
          title: "新对话",
          createdAt: Date.now().toString()
        }
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          currentSession: newSessionId,
        }))
      },
      deleteSession: (sessionId) => {
        set(state => {
          const newSessions = state.sessions.filter(session => session.id !== sessionId);
          return {
            sessions: newSessions,
            currentSession: state.currentSession === sessionId ? (newSessions.length === 0 ? "" : newSessions[0].id) : state.currentSession
          }
        })
      },
      selectSession: (sessionId) => {
        set(() => ({
          currentSession: sessionId,
        }))
      },

      // 消息action

      // 用户发送消息并生成相应generatingTask
      startGeneration: (sessionId: string, content: string) => {
        const normalizeContent = content.trim();
        if (!content) return null;

        const requestId = crypto.randomUUID();
        const userId = crypto.randomUUID();
        const assistantId = crypto.randomUUID();
        const createdAt = Date.now();

        let task: GenerationTask | null = null;

        set((state) => ({
          sessions: state.sessions.map((session) => {
            if (session.id !== sessionId) return session;
            const isGenerating = session.messages.some((message) => {
              return message.status === "generating" || message.status === "pending"
            });
            const hasUserMessage = session.messages.some(message => message.role === "user");
            if (isGenerating) return session;

            const userMessage: Message = {
              id: userId,
              requestId: requestId,
              role: "user",
              content: normalizeContent,
              status: "completed",
              createdAt: createdAt,
            };

            const assistantMessage: Message = {
              id: assistantId,
              requestId: requestId,
              role: "assistant",
              content: "",
              status: "generating",
              createdAt: createdAt,
              // 用户已发送就直接显示正在理解，不等后端反应
              researchProgress:{
                stage:"understanding",
                label:"正在理解你的问题"
              }
            }

            task = {
              sessionId: sessionId,
              messageId: assistantId,
              requestId: requestId
            };

            return {
              ...session,
              title: hasUserMessage ?
                session.title :
                normalizeContent,
              messages: [...session.messages, userMessage, assistantMessage]
            }
          })
        }));

        return task;
      },
      // 更新相应请求对应的ai回复的内容
      appendMessageChunk(sessionId: string, messageId: string, chunk: string, requestId: string) {
        if (!chunk) return;
        set(state => {
          return {
            sessions: state.sessions.map(session => {
              if (session.id !== sessionId) return session;
              return {
                ...session,
                messages: session.messages.map(message => {
                  if (message.id !== messageId
                    || message.requestId !== requestId
                    || message.status !== "generating"
                  ) return message;
                  return {
                    ...message,
                    content: message.content + chunk
                  }
                })
              }
            })
          }
        })
      },
      // 指定信息的内容输入完成，状态更新为completed
      completeMessage: (sessionId: string, messageId: string, requestId: string) => {
        set(state => ({
          sessions: state.sessions.map(session => {
            if (session.id !== sessionId) return session;
            return {
              ...session,
              messages: session.messages.map(message => {
                if (message.id !== messageId
                  || message.requestId !== requestId
                  || message.status !== "generating"
                ) return message;
                return {
                  ...message,
                  status: "completed",
                  error: undefined,
                  researchProgress:undefined
                }
              })
            }
          })
        }))
      },
      // 指定信息的内容并未输入完成，用户强制停止，状态更新为aborted
      abortMessage: (sessionId: string, messageId: string, requestId: string) => {
        set(state => ({
          sessions: state.sessions.map(session => {
            if (session.id !== sessionId) return session;
            return {
              ...session,
              messages: session.messages.map(message => {
                if (message.id !== messageId
                  || message.requestId !== requestId
                  || message.status !== "generating"
                ) return message;
                return {
                  ...message,
                  status: "aborted",
                  researchProgress:undefined
                }
              })
            }
          })
        }))
      },
      // 指定信息的内容输入失败，系统强制停止，状态更新为failed
      failMessage: (sessionId: string, messageId: string, requestId: string, error: string) => {
        set(state => ({
          sessions: state.sessions.map(session => {
            if (session.id !== sessionId) return session;
            return {
              ...session,
              messages: session.messages.map(message => {
                if (message.id !== messageId
                  || message.requestId !== requestId
                  || message.status !== "generating"
                ) return message;
                return {
                  ...message,
                  status: "failed",
                  error: error,
                  researchProgress:undefined
                }
              })
            }
          })
        }))
      },
      // 指定failed或aborted的信息重新发送请求，状态更新为generating
      prepareRetryMessage: (sessionId: string, messageId: string) => {
        let task: GenerationTask | null = null;
        const newRequestId = crypto.randomUUID();
        set(state => ({
          sessions: state.sessions.map(session => {
            if (session.id !== sessionId) return session;
            return {
              ...session,
              messages: session.messages.map(message => {
                if (message.id !== messageId
                  || message.role !== "assistant"
                  || !["failed", "aborted"].includes(message.status)
                ) return message;

                task = {
                  sessionId: sessionId,
                  messageId: messageId,
                  requestId: newRequestId
                }
                return {
                  ...message,
                  content: "",
                  requestId: newRequestId,
                  status: "generating",
                  error: undefined
                }
              })
            }
          })
        }))

        return task;
      },
      prepareRegenerateMessage: (sessionId: string, messageId: string) => {
        let task: GenerationTask | null = null;
        const newRequestId = crypto.randomUUID();
        const createdAt = Date.now();

        set(state => ({
          sessions: state.sessions.map(session => {
            if (session.id !== sessionId) return session;

            const isGenerating = session.messages.some(message => {
              return message.status === "generating" || message.status === "pending";
            });

            if (isGenerating) return session;

            const messageIndex = session.messages.findIndex(message => {
              return message.id === messageId;
            });

            if (messageIndex === -1) return session;

            const targetMessage = session.messages[messageIndex];

            if (
              targetMessage.role !== "assistant" ||
              targetMessage.status === "generating" ||
              targetMessage.status === "pending"
            ) {
              return session;
            }

            task = {
              sessionId,
              messageId,
              requestId: newRequestId,
            };

            const regenerateMessage: Message = {
              ...targetMessage,
              requestId: newRequestId,
              content: "",
              status: "generating",
              error: undefined,
              createdAt,
            };

            return {
              ...session,
              messages: [
                ...session.messages.slice(0, messageIndex),
                regenerateMessage,
              ],
            };
          })
        }))

        return task;
      },
      createMessage: (sessionId: string, newMessage: Message) => {
        set((state) => {
          const newSessions = state.sessions.map(session => {
            if (session.id !== sessionId) return session;
            return {
              ...session,
              messages: [...session.messages, newMessage]
            };
          })
          return {
            sessions: newSessions
          }
        })
      },
      updateMessage: (sessionId: string, messageId: string, updateContent: string) => {
        set(state => {
          const newSession = state.sessions.map(session => {
            if (session.id !== sessionId) return session;
            const newMessages = session.messages.map(message => {
              if (message.id !== messageId) return message;
              return {
                ...message,
                content: updateContent,
                status: "completed" as const
              }
            })
            return {
              ...session,
              messages: newMessages
            };
          })
          return {
            sessions: newSession
          }
        })
      },

      // 设置当前运行状态
      setResearchProgress:(sessionId:string,messageId:string,requestId:string,progress:ResearchProgress | undefined)=>{
        set((state)=>({
          sessions:state.sessions.map(session=>{
            if(session.id!==sessionId)return session

            return{
              ...session,
              messages:session.messages.map(message=>{
                if(message.id!==messageId || message.requestId!==requestId)return message
                
                return{
                  ...message,
                  researchProgress:progress
                }
              })
            }
          })
        }))
      },


      setIsAtBottom: (bottom: boolean) => {
        set(() => ({
          isAtBottom: bottom,
        }))
      }
    }), {
    name: "Sessions-storage"
  }
  )
)
