import { Client } from "@langchain/langgraph-sdk"
import type { AiMessage } from "@/type/api"
import type { ResearchProgress } from "@/type/chat"

// 定义askFlow中认识的事件
export type DeepResearchEvent =
  | {
    type: "progress",
    stage: ResearchProgress["stage"],
    label: string
  }
  | {
    type: "content",
    content: string
  }
  | {
    type: "done"
  }


// langgraph的updates是某个节点已经执行完成后返回的状态更新，因此对应的label应该是当前update的下个状态的label
const NODE_PROGRESS: Record<string, ResearchProgress> = {
  clarify_with_user: {
    stage: "planning",
    label: "正在整理研究需求..."
  },
  write_research_brief: {
    stage: "researching",
    label: "正在进行深度研究..."
  },
  research_supervisor: {
    stage: "writing",
    label: "正在生成最终研究报告..."
  }
}

type JsonObject = Record<string, unknown>

// 外部数据永远先当unknown，验证后再使用
const isObject = (value: unknown): value is JsonObject => {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  )
}

// 解析clarify_with_user节点的update
const getClarificationContent = (data: JsonObject): string | null => {
  const clarifyNode = data["clarify_with_user"];

  if (!isObject(clarifyNode)) return null

  const messages = clarifyNode["messages"]

  if (!Array.isArray(messages) || messages.length === 0) return null

  const lastMessage = messages[messages.length - 1]

  if (!isObject(lastMessage)) return null

  const content = lastMessage["content"]

  return typeof content === "string" ? content : null
}

// 解析final_report_generation节点的update
const getFinalReport = (data: JsonObject): string | null => {
  const finalNode = data["final_report_generation"];

  if (!isObject(finalNode)) return null

  const finalReport = finalNode["final_report"]

  return typeof finalReport === "string" ? finalReport : null
}

// 解析final_report_generation节点的messages-tuple
const getFinalReportToken = (data: unknown): string | null => {
  if (!Array.isArray(data) || data.length < 2) {
    return null
  }

  const messageChunk = data[0]
  const metadata = data[1]

  if (!isObject(messageChunk) || !isObject(metadata)) return null

  if (metadata["langgraph_node"] !== "final_report_generation") return null

  const content = messageChunk["content"]
  return typeof content === "string" ? content : null
}

const API_URL = import.meta.env.VITE_LANGGRAPH_API_URL ?? "http://localhost:2024"

export const langgraphClient = new Client({
  apiUrl: API_URL
})

type DeepResearchOptions = {
  messages: AiMessage[],
  signal?: AbortSignal
}

export async function* streamDeepResearch({
  messages,
  signal,
}: DeepResearchOptions) {
  const stream = langgraphClient.runs.stream(
    null,
    "Deep Researcher",
    {
      input: {
        messages
      },
      // 订阅哪些运行事件，通过拿到的chunk的event这个字段来区分，
      // 对于updates类型事件，该字段值为updates，对于messages-tuple
      // 类型事件，该字段值为messages
      streamMode: ["updates", "messages-tuple"],
      signal
    }
  )

  // 缓存clarify节点数据，用于在agent需要用户澄清问题时显示
  let pendingClarification: string | null = null
  // 保存是否获得最终报告的messages-tuple输出，如果没有，就从update中直接拿
  let hasStreamedFinalReport = false
  // 标记最终报告是否已完成
  let completed = false

  for await (const chunk of stream) {
    // 解析两种事件：messages-tuple和updates

    // 1.LLM token(messages-tuple)
    if (chunk.event === "messages") {
      const token = getFinalReportToken(chunk.data)
      console.log("messages-tuple:",chunk.data)

      if (token) {
        hasStreamedFinalReport = true

        yield {
          type: "content",
          content: token
        }
      }

      continue
    }

    // 2.Graph state update
    if (chunk.event === "updates") {
      if (!isObject(chunk.data)) continue

      const data = chunk.data

      let nodeName = Object.keys(data)[0]

      if (nodeName) {
        const progress = NODE_PROGRESS[nodeName]

        if (progress)
          yield {
            type: "progress",
            stage: progress.stage,
            label: progress.label
          }
      }

      // 额外处理clarify_with_user的信息

      // 缓存clarify_with_user的用户可见信息
      const clarification = getClarificationContent(data)

      if (clarification) pendingClarification = clarification

      // 如果进入了write_research_brief阶段，就说明不需要澄清了，可以清空缓存信息
      if ("write_research_brief" in data) {
        pendingClarification = null
      }

      // 额外处理final_report_generation的信息
      const finalReport = getFinalReport(data)

      if (finalReport) {
        pendingClarification = null

        if (!hasStreamedFinalReport) {
          yield {
            type: "content",
            content: finalReport
          }
        }
        yield {
          type: "done"
        }
        completed = true
      }

    }
  }

  // 如果整个stream结束但只经过了clarify_with_user则说明graph正在等待用户澄清
  if (!completed && pendingClarification) {
    yield {
      type: "content",
      content: pendingClarification
    }
    yield{
      type:"done"
    }
  }
}