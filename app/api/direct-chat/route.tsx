import { NextResponse } from "next/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

// ===== Config =====
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || ""
const genAI = GOOGLE_API_KEY ? new GoogleGenerativeAI(GOOGLE_API_KEY) : null
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || ""

// ===== Helper: Tavily search =====
async function searchWithTavily(query: string) {
  try {
    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${TAVILY_API_KEY}`,
      },
      body: JSON.stringify({
        query,
        search_depth: "advanced",
        max_results: 5,
      }),
    })

    if (!response.ok) throw new Error(`Tavily API error: ${response.statusText}`)
    return await response.json()
  } catch (error) {
    console.error("Tavily search error:", error)
    return { results: [], answer: "" }
  }
}

// ===== Main handler =====
export async function POST(req: Request) {
  if (!GOOGLE_API_KEY || !genAI) {
    return NextResponse.json({ error: "Google API key not configured" }, { status: 500 })
  }

  try {
    const { messages, model = "gemini-1.5-flash", stream = false, files = [] } = await req.json()

    // Step 1: Extract user question
    const lastMessage = messages[messages.length - 1]
    const userQuestion =
      typeof lastMessage.content === "string"
        ? lastMessage.content
        : lastMessage.content?.find((c: any) => c.type === "text")?.text || ""

    if (!userQuestion.trim()) {
      return NextResponse.json({ error: "No question provided" }, { status: 400 })
    }

    let conversationContext = ""
    if (messages.length > 1) {
      const previousMessages = messages.slice(0, -1)
      conversationContext = previousMessages
        .map((msg: any) => {
          const role = msg.role === "user" ? "Người dùng" : "Trợ lý"
          const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
          return `${role}: ${content}`
        })
        .join("\n")
    }

    // Step 2: Tavily search
    console.log("[Tavily-to-Gemini] Step 1: Searching with Tavily")
    const searchResults = await searchWithTavily(userQuestion)

    // Step 3: Build enhanced prompt
    let enhancedPrompt = userQuestion

    if (searchResults?.answer || searchResults?.results?.length > 0) {
      const searchContent =
        searchResults.answer ||
        searchResults.results.map((r: any, i: number) => `[Nguồn ${i + 1}]: ${r.content}`).join("\n\n")

      enhancedPrompt = `Bạn là một AI assistant thông minh và hữu ích. Dựa trên thông tin tìm kiếm mới nhất và ngữ cảnh cuộc trò chuyện, hãy trả lời câu hỏi một cách chi tiết, chính xác và có cấu trúc.

${conversationContext ? `NGỮ CẢNH CUỘC TRÒ CHUYỆN:\n${conversationContext}\n\n` : ""}THÔNG TIN TÌM KIẾM:
${searchContent}

CÂU HỎI HIỆN TẠI: ${userQuestion}

YÊU CẦU TRẢ LỜI:
- Phân tích và tổng hợp thông tin từ các nguồn đáng tin cậy
- Kết hợp với ngữ cảnh cuộc trò chuyện trước đó nếu có liên quan
- Trình bày theo cấu trúc rõ ràng với các điểm chính
- Sử dụng danh sách đánh số khi cần thiết
- Trích dẫn nguồn cụ thể khi có thể
- Đưa ra nhận xét hoặc phân tích sâu hơn nếu phù hợp
- Trả lời bằng tiếng Việt với ngôn ngữ chuyên nghiệp
- Không sử dụng ký hiệu đặc biệt hoặc markdown formatting`
    }

    const processedMessages = messages.map((msg: any, index: number) => {
      if (index === messages.length - 1) {
        return { ...msg, content: enhancedPrompt }
      }
      return msg
    })

    // Step 4: Call Gemini
    console.log("[Tavily-to-Gemini] Step 2: Calling Gemini")
    const geminiModel = genAI.getGenerativeModel({ model })

    const result = await geminiModel.generateContent({
      contents: processedMessages.map((msg: any) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content) }],
      })),
      generationConfig: { temperature: 0.7, maxOutputTokens: 2000 },
    })

    const response = await result.response
    const text = response.text()

    console.log("[Tavily-to-Gemini] Completed:", {
      questionLength: userQuestion.length,
      answerLength: text.length,
      searchResultsCount: searchResults?.results?.length || 0,
    })

    return NextResponse.json({
      candidates: [
        {
          content: { parts: [{ text }], role: "model" },
          finishReason: "STOP",
        },
      ],
      choices: [{ message: { role: "assistant", content: text } }],
      searchResults,
      workflow: "tavily-to-gemini",
      step1: {
        type: "tavily_search",
        query: userQuestion,
        resultsCount: searchResults?.results?.length || 0,
        prompt: `Tìm kiếm thông tin về: "${userQuestion}"`,
      },
      step2: {
        type: "gemini_response",
        promptLength: enhancedPrompt.length,
        content: text.substring(0, 200) + "...",
      },
    })
  } catch (error: any) {
    console.error("Tavily-to-Gemini error:", error)
    return NextResponse.json(
      {
        error: error?.message || "Failed to process Tavily-to-Gemini workflow",
        errorType: "workflow_error",
      },
      { status: 500 },
    )
  }
}
