import {Client} from "@langchain/langgraph-sdk"

export const langgraphClient=new Client({
  apiUrl:"http://localhost:2024"
})

export async function runRearch(question:string){
  const stream=langgraphClient.runs.stream(
    null,
    "Deep Researcher",
    {
      input:{
        messages:[
          {
            role:"user",
            content:question
          }
        ]
      },
      streamMode:"updates"
    }
  )

  for await (const chunk of stream){
    console.log(chunk.event)
    console.log(chunk.data)
  }
}