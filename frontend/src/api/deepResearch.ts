import {Client, type AIMessage} from "@langchain/langgraph-sdk"
import type { AiMessage } from "@/type/api"

const API_URL=import.meta.env.VITE_LANGGRAPH_API_URL ?? "http://localhost:2024"

export const langgraphClient=new Client({
  apiUrl:API_URL
})

type DeepResearchOptions={
  messages:AiMessage[],
  signal?:AbortSignal
}

export async function* streamDeepResearch({
  messages,
  signal,
}:DeepResearchOptions){
  const stream=langgraphClient.runs.stream(
    null,
    "Deep Researcher",
    {
      input:{
        messages
      },
      streamMode:"updates",
      signal
    }
  )

  for await (const chunk of stream){
    yield chunk
  }
}