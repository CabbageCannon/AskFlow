import type { AiMessage, ChatRequest } from "@/type/api"
import { useChatStore, type GenerationTask } from "@/store/chatStore"
import { useCallback, useEffect, useRef } from "react"
import { streamDeepResearch } from "@/api/deepResearch"

type activeRequest = GenerationTask & {
  controller: AbortController
}

/**
 * 从 Store 中提取发给模型的会话上下文。
 *
 * 截取到当前 AI 占位消息之前，
 * 并且只发送已经完成的消息。
 */
const buildAiMessage = (sessionId: string, assistantMessageId: string): AiMessage[] => {
  const state = useChatStore.getState();
  const session = state.sessions.find(session => session.id === sessionId);

  if (!session) return [];

  const assistantMessageIndex = session.messages.findIndex(message => message.id === assistantMessageId);
  if (assistantMessageIndex === -1) return [];

  return session.messages
    .slice(0, assistantMessageIndex)
    .filter(message => message.status === "completed")
    .map(message => ({
      role: message.role,
      content: message.content
    }))
}

// 作为chatClient和chatStore的中间件
export const useSendMessage = () => {
  const appendMessageChunk = useChatStore(state => state.appendMessageChunk);
  const completeMessage = useChatStore(state => state.completeMessage);
  const failMessage = useChatStore(state => state.failMessage);
  const abortMessage = useChatStore(state => state.abortMessage);
  const currentSessionId = useChatStore(state => state.currentSession);
  const startGeneration = useChatStore(state => state.startGeneration);
  const prepareRetryMessage = useChatStore(state => state.prepareRetryMessage);
  const prepareRegenerateMessage = useChatStore(state => state.prepareRegenerateMessage);

  // 用于存储不同会话的请求信息
  const activeRequests = useRef(new Map<string, activeRequest>);

  // 当前对话是否有正在生成的消息
  const isGenerating = useChatStore(state => {
    const currentSession = state.sessions.find(session => session.id === state.currentSession);
    return currentSession?.messages.some(message => message.status === "generating" || message.status === "pending") ?? false;
  });

  /**
   * 执行已经创建好的生成任务。
   *
   * startGeneration 和 retryMessage
   * 都可以生成 GenerationTask。
   */
  const runGeneration = useCallback(async (task: GenerationTask): Promise<void> => {
    const {
      sessionId,
      messageId,
      requestId
    } = task;

    // 检测当前对话是否已经在请求
    if (activeRequests.current.has(sessionId)) return;

    const controller = new AbortController;

    activeRequests.current.set(sessionId, {
      ...task,
      controller
    })

    const aiMessages = buildAiMessage(sessionId, messageId);

    try {
      for await (const chunk of streamDeepResearch({
        messages:aiMessages,
        signal:controller.signal
      })) {
        if(chunk.event!="updates")continue

        const data=chunk.data

        console.log("LangGraph update:",data)
      }
    } catch (error) {
      const requestWasAborted = controller.signal.aborted || error.code === "ABORTED"
      if (requestWasAborted) {
        abortMessage(sessionId, messageId, requestId);
        return;
      }

      const errorMessage = error instanceof Error
        ? error.message
        : "聊天请求失败";

      failMessage(sessionId, messageId, requestId, errorMessage);
    } finally {
      const activeRequest = activeRequests.current.get(sessionId);

      // 防止旧请求由于执行慢，导致删除了新请求
      if (activeRequest?.requestId === requestId) {
        activeRequests.current.delete(sessionId);
      }
    }
  }, [appendMessageChunk, completeMessage, failMessage, abortMessage])

  // 发送一条信息
  const sendMessage = useCallback(async (
    content: string,
    sessionId = currentSessionId
  ): Promise<void> => {
    const nomalizedContent = content.trim();

    // 传入数据有误
    if (!sessionId || !nomalizedContent) return;

    // 当ai生成信息时，不允许用户发送信息
    if (activeRequests.current.has(sessionId)) return;

    const generationTask = startGeneration(sessionId, nomalizedContent);

    if (!generationTask) return;

    await runGeneration(generationTask);
  }, [currentSessionId, startGeneration, runGeneration])

  // 主动停止指定会话的生成
  const stopMessage = useCallback((sessionId = currentSessionId): void => {
    if (!sessionId) return;

    const activerequest = activeRequests.current.get(sessionId);

    if (!activerequest) return;

    activerequest.controller.abort();

    // 立即更新UI，无需等待后端处理完，后端同样也会调用abortMessage，但Store中会检查generating状态，因此理论上会忽略第二次
    abortMessage(activerequest.sessionId, activerequest.messageId, activerequest.requestId);
  }, [currentSessionId, abortMessage])

  // 重新生成失败的或被取消的AI消息
  const retryMessage = useCallback(async (messageId: string, sessionId = currentSessionId): Promise<void> => {
    if (!sessionId) return;

    if (activeRequests.current.has(sessionId)) return;

    /*
    * Store 会为这条 AI 消息创建新的 requestId，
    * 并重新设置为 generating。
    */

    const generationTask = prepareRetryMessage(sessionId, messageId);

    if (!generationTask) return;

    await runGeneration(generationTask);
  }, [currentSessionId, prepareRetryMessage, runGeneration])

  const regenerateMessage = useCallback(async (messageId: string, sessionId = currentSessionId): Promise<void> => {
    if (!sessionId) return;

    if (activeRequests.current.has(sessionId)) return;

    const generationTask = prepareRegenerateMessage(sessionId, messageId);

    if (!generationTask) return;

    await runGeneration(generationTask);
  }, [currentSessionId, prepareRegenerateMessage, runGeneration])

  // 使用该Hook的组件卸载时，取消所有仍未完成的请求
  useEffect(() => {
    const requests = activeRequests.current;

    return () => {
      for (const request of requests.values()) {
        request.controller.abort();

        abortMessage(
          request.sessionId,
          request.messageId,
          request.requestId
        )
      }

      requests.clear();
    }
  }, [abortMessage]);

  return {
    isGenerating,
    sendMessage,
    stopMessage,
    retryMessage,
    regenerateMessage
  }
}

