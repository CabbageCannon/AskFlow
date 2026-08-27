import parseSSE, { SSEParseError } from "./parseSSE";

import type {
  ChatRequest,
  ChatStreamEvent
} from "@/type/api";

type streamChatOptions = {
  request: ChatRequest,
  signal: AbortSignal
};

// 定义chatClientError类型
export type ChatClientErrorCode =
  | "ABORTED" //用户终止
  | "NETWORK_ERROR" //网络层出错，一般是没有拿到响应
  | "HTTP_ERROR" //后端返回非2xx的状态码
  | "EMPTY_RESPONSE_BODY" //响应的body为空
  | "STREAM_PARSE_ERROR" //SSE 或 Json 解析出错
  | "STREAM_READ_ERROR" //流读取过程中出错
  | "REQUEST_ID_MISMATCH" //后端返回的请求id和当前不一致
  | "INCOMPLETE_STREAM"; //后端最后没有返回done或者error的ChatStreamEvent

// 错误的额外信息
type ChatClientErrorOptions = {
  status?: string,
  retryable?: boolean
}

export class ChatClientError extends Error {
  readonly code: ChatClientErrorCode;
  readonly status?: string;
  readonly retryable: boolean;

  constructor(
    errorCode: ChatClientErrorCode,
    message: string,
    errorOptions: ChatClientErrorOptions = {}
  ) {
    super(message)

    this.name = "ChatClientError";
    this.status = errorOptions.status;
    this.code = errorCode;
    this.retryable = errorOptions.retryable ?? false;
  }
}

const readErrorMessage = async (
  response: Response
): Promise<string> => {
  const cloneResponse = response.clone();

  try {
    const data: unknown = await cloneResponse.json();

    if (typeof data === "object" &&
      data !== null
    ) {
      const errorData = data as Record<string, unknown>;
      if (typeof errorData.message === "string") return errorData.message;
      if (typeof errorData.error === "string") return errorData.error;
    }
  } catch {
    //错误信息可能不是json格式，尝试直接转换为字符串
  }

  // 有可能后端返回的body不是json格式
  try {
    // 直接把body当成字符串解析
    const errorData = await response.text();
    if (errorData.trim()) return errorData;
  } catch {
    //错误信息转换字符串也失败，尝试返回默认错误信息
  }

  // 仍然拿不到，就返回默认信息(兜底)
  return `聊天请求失败，状态码是 ${response.status}`;
}

// 请求聊天接口并逐个返回聊天事件(发送网络层请求并处理返回结果)
export async function* streamChat({
  request,
  signal
}: streamChatOptions): AsyncGenerator<ChatStreamEvent> {
  let response: Response;

  try {
    response = await fetch("/api/chat", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream"
      },

      body: JSON.stringify(request),
      signal
    })
  } catch {
    if (signal.aborted) {
      throw new ChatClientError("ABORTED", "生成已取消");
    }

    throw new ChatClientError("NETWORK_ERROR", "无法连接到聊天服务", {
      retryable: true
    })
  };

  if (!response.ok) {
    const errorMessage = await readErrorMessage(response);

    throw new ChatClientError("HTTP_ERROR",
      errorMessage,
      {
        status: String(response.status),
        // status大于500一般可以重试，400-499一般是请求有问题
        retryable: response.status >= 500
      }
    )
  }

  if (!response.body) {
    throw new ChatClientError("EMPTY_RESPONSE_BODY", "服务端没有返回响应流", {
      retryable: true
    });
  }

  let receivedTerminalEvent = false;

  try {
    for await (const event of parseSSE(response.body)) {
      if (event.requestId !== request.requestId) {
        throw new ChatClientError("REQUEST_ID_MISMATCH", "响应与当前请求不匹配");
      }

      if (event.type === "done" || event.type === "error") {
        receivedTerminalEvent = true;
      }

      if (event) yield event;
    }
  } catch (error) {
    if (signal.aborted) throw new ChatClientError("ABORTED", "生成已取消");

    if (error instanceof ChatClientError) throw error;
    if (error instanceof SSEParseError) {
      // 这里大概率是后端返回的数据有问题导致解析失败,没有必要重试
      throw new ChatClientError("STREAM_PARSE_ERROR", error.message);
    }

    // 这里认为可能是网络读取出错了，因此允许重试
    throw new ChatClientError("STREAM_READ_ERROR", "读取聊天响应时发生错误", {
      retryable: true
    });
  }

  if (!receivedTerminalEvent) {
    throw new ChatClientError("INCOMPLETE_STREAM", "聊天响应未完整结束", {
      retryable: true
    })
  }
} 