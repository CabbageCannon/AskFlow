import type { ChatStreamEvent } from "@/type/api";

type JsonObject = Record<string, unknown>;

// 自定义错误类：该错误类表示SSEParse过程出错
export class SSEParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SSEParseError"
  }
}

// 判断是否是json对象
const isJsonObject = (value: unknown): value is JsonObject => {
  return (
    typeof value === "object"
    && value !== null
    && !Array.isArray(value)
  )
}

// 获取json对象指定键的值
const requireString = (data: JsonObject, key: string): string => {
  const value = data[key];

  if (typeof value !== "string") {
    throw new SSEParseError(`SSE 事件字段 ${key} 必须是字符串`);
  }

  return value;
}

// 根据字符串块生成SSE事件对象
const parseEventBlock = (block: string): ChatStreamEvent | null => {
  let eventName = '';
  const dataLines: string[] = [];
  const lines = block.split(/\r?\n/);

  for (const line of lines) {
    // SSE中:可以用于注释
    if (line.startsWith(":")) continue;

    const separatorIndex = line.indexOf(":");

    const field = separatorIndex === -1
      ? ""
      : line.slice(0, separatorIndex);

    let value = separatorIndex === -1
      ? ""
      : line.slice(separatorIndex + 1);

    value = value.startsWith(" ") ? value.slice(1) : value;

    if (field === "event") eventName = value.trim();

    if (field === "data") dataLines.push(value);
  }

  // 空事件对象直接返回
  if (dataLines.length === 0) return null;

  // 若是其它事件暂时先返回null，以后再扩展
  if (eventName !== 'delta'
    && eventName !== 'done'
    && eventName !== 'error'
  ) return null;

  const dataText = dataLines.join("\n");

  let data: unknown;

  // 尝试将json格式字符串解析为js对象
  try {
    data = JSON.parse(dataText);
  } catch {
    throw new SSEParseError(`
      无法解析 ${eventName} 事件的 JSON:${dataText}
      `)
  }

  // 检查data是否真的是json格式
  if (!isJsonObject(data)) throw new SSEParseError(`
      ${eventName} 事件的data必须是 JSON
    `);

  const requestId = requireString(data, "requestId");
  switch (eventName) {
    case "delta": {
      return {
        type: "delta",
        requestId: requestId,
        content: requireString(data, "content")
      }
    };

    case "done": {
      const finishReason = requireString(data, "finishReason");

      if (finishReason !== "stop"
        && finishReason !== "length"
      ) throw new SSEParseError(`
        未知的finishReason ${finishReason}
        `)

      return {
        type: "done",
        requestId: requestId,
        finishReason: finishReason
      }
    };

    case "error": {
      const retryable = data.retryable;
      if (typeof retryable !== "boolean") throw new SSEParseError(`
        error 事件的 retryable 必须是布尔值
        `);

      return {
        type: "error",
        requestId: requestId,
        code: requireString(data, "code"),
        message: requireString(data, "message"),
        retryable
      }
    }
  }
}

export default async function* parseSSE(
  stream: ReadableStream<Uint8Array>
): AsyncGenerator<ChatStreamEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder('utf-8')

  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }

      /*
        * SSE 使用空行分隔事件。
        * 同时兼容：
        * \n\n
        * \r\n\r\n
        */
      buffer += decoder.decode(value, {
        stream: true
      });

      const blocks = buffer.split(/\r?\n\r?\n/);

      buffer = blocks.pop() ?? "";

      for (const block of blocks) {
        // 多个空行时可能出现空的block
        if (!block.trim()) continue;
        const event = parseEventBlock(block);

        if (event)
          yield event;
      }
    }
    if (buffer.trim()) {
      const event = parseEventBlock(buffer);

      if (event) {
        yield event;
      }
    }

  } finally {
    reader.releaseLock();
  }
}