import { type NextRequest, NextResponse } from "next/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY
const TAVILY_API_KEY = process.env.TAVILY_API_KEY
const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY
const OPENAI_BASE_URL = "https://api.openai.com/v1"
const GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

// Model mapping to determine which API to use
const MODEL_MAPPING = {
  // OpenAI models
  "gpt-4o": { provider: "openai", model: "gpt-4o" },
  "gpt-4o-mini": { provider: "openai", model: "gpt-4o-mini" },
  "gpt-4-turbo": { provider: "openai", model: "gpt-4-turbo" },
  "gpt-3.5-turbo": { provider: "openai", model: "gpt-3.5-turbo" },

  // Google models - updated with newer versions
  "gemini-2.5-pro": { provider: "google", model: "gemini-2.5-pro" },
  "gemini-2.5-flash": { provider: "google", model: "gemini-2.5-flash" },
  "gemini-2.5-flash-lite": { provider: "google", model: "gemini-2.5-flash-lite" },
  "gemini-2.0-flash": { provider: "google", model: "gemini-2.0-flash" },
  "gemini-2.0-flash-lite": { provider: "google", model: "gemini-2.0-flash-lite" },
  "gemini-2.0-pro": { provider: "google", model: "gemini-2.0-pro" },
  "gemini-1.5-flash": { provider: "google", model: "gemini-1.5-flash" },
  "gemini-1.5-flash-lite": { provider: "google", model: "gemini-1.5-flash-lite" },
  "gemini-1.5-pro": { provider: "google", model: "gemini-1.5-pro" },
  "gemini-1.0-pro": { provider: "google", model: "gemini-1.0-pro" },
  "gemini-1.0-ultra": { provider: "google", model: "gemini-1.0-ultra" },
  "gemini-1.0-nano": { provider: "google", model: "gemini-1.0-nano" },
  "gemini-embedding": { provider: "google", model: "gemini-embedding" },
  "gemini-2.5-flash-preview-tts": { provider: "google", model: "gemini-2.5-flash-preview-tts" },
  "gemini-2.5-flash-preview-image-generation": {
    provider: "google",
    model: "gemini-2.5-flash-preview-image-generation",
  },
  "gemini-2.5-flash-live-preview-04-09": { provider: "google", model: "gemini-2.5-flash-live-preview-04-09" },
}

const genAI = GOOGLE_API_KEY ? new GoogleGenerativeAI(GOOGLE_API_KEY) : null

// Tavily search function
async function searchWithTavily(query: string) {
  if (!TAVILY_API_KEY) {
    console.warn("Tavily API key not configured, skipping search")
    return null
  }

  try {
    console.log("[Tavily] Searching for:", query.substring(0, 100) + "...")

    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        api_key: TAVILY_API_KEY,
        query: query,
        search_depth: "advanced",
        include_answer: true,
        include_raw_content: false,
        max_results: 5,
        include_domains: [],
        exclude_domains: [],
      }),
    })

    if (!response.ok) {
      console.error("Tavily API error:", response.status, response.statusText)
      return null
    }

    const data = await response.json()
    console.log("[Tavily] Search completed:", {
      resultsCount: data.results?.length || 0,
      hasAnswer: !!data.answer,
    })

    return data
  } catch (error) {
    console.error("Tavily search error:", error)
    return null
  }
}

// Perplexity API search function
async function searchWithPerplexity(query: string) {
  if (!PERPLEXITY_API_KEY) {
    console.warn("Perplexity API key not configured, skipping search")
    return null
  }

  try {
    console.log("[Perplexity] Searching for:", query.substring(0, 100) + "...")

    const response = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${PERPLEXITY_API_KEY}`
      },
      body: JSON.stringify({
        model: "sonar-medium-online",
        messages: [
          {
            role: "system",
            content: "Be precise and concise. Provide accurate and relevant information based on web search results."
          },
          {
            role: "user",
            content: query
          }
        ],
        max_tokens: 1000,
        temperature: 0.7,
      })
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("Perplexity API error:", response.status, response.statusText, errorText)
      return null
    }

    const data = await response.json()
    console.log("[Perplexity] Search completed:", {
      responseLength: data.choices?.[0]?.message?.content?.length || 0
    })

    return {
      answer: data.choices?.[0]?.message?.content,
      search_metadata: {
        total_results: 1,
        query: query
      },
      results: [
        {
          content: data.choices?.[0]?.message?.content,
          title: "Perplexity AI Response",
          url: "https://www.perplexity.ai"
        }
      ]
    }
  } catch (error) {
    console.error("Perplexity search error:", error)
    return null
  }
}

export async function POST(request: NextRequest) {
  try {
    const { messages, model, stream = false, files = [], workflow = "single" } = await request.json()

    const modelConfig = MODEL_MAPPING[model as keyof typeof MODEL_MAPPING]
    if (!modelConfig) {
      return NextResponse.json({ error: `Unsupported model: ${model}` }, { status: 400 })
    }

    const { provider, model: actualModel } = modelConfig

    // Workflow: tavily-to-gemini (search with Tavily, then ask Gemini)
    if (workflow === "tavily-to-gemini") {
      return await handleTavilyToGemini(messages, actualModel, stream, files)
    }

    // Workflow: chatgpt-to-gemini (process with ChatGPT, then enhance with Gemini)
    if (workflow === "chatgpt-to-gemini") {
      return await handleChatGPTToGemini(messages, actualModel, stream, files)
    }

    // Workflow: perplexity-to-gemini (search with Perplexity, then process with Gemini)
    if (workflow === "perplexity-to-gemini") {
      return await handlePerplexityToGemini(messages, actualModel, stream, files)
    }

    // Default: direct Gemini processing
    return await handleGoogle(messages, actualModel, stream, files)
  } catch (error) {
    console.error("Direct Chat API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// Perplexity-to-Gemini workflow handler
async function handlePerplexityToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  if (!PERPLEXITY_API_KEY || !GOOGLE_API_KEY || !genAI) {
    return NextResponse.json(
      { error: "Perplexity or Google API key not configured" }, 
      { status: 500 }
    )
  }

  try {
    // Step 1: Extract user question
    const lastMessage = messages[messages.length - 1]
    const userQuestion = 
      typeof lastMessage.content === "string"
        ? lastMessage.content
        : lastMessage.content?.find((c: any) => c.type === "text")?.text || ""

    if (!userQuestion.trim()) {
      return NextResponse.json({ error: "No question provided" }, { status: 400 })
    }

    // Step 2: Search with Perplexity
    console.log("[Perplexity-to-Gemini] Step 1: Searching with Perplexity")
    const searchResults = await searchWithPerplexity(userQuestion)

    // Step 3: Build enhanced prompt with search results
    let enhancedPrompt = userQuestion

    if (searchResults?.answer) {
      enhancedPrompt = `Bạn là một AI assistant chuyên nghiệp. Dựa trên thông tin tìm kiếm từ Perplexity AI dưới đây, hãy trả lời câu hỏi một cách chi tiết và chính xác:

THÔNG TIN TỪ PERPLEXITY AI:
${searchResults.answer}

CÂU HỎI: ${userQuestion}

YÊU CẦU:
1. Phân tích và tóm tắt thông tin từ kết quả tìm kiếm
2. Bổ sung kiến thức chuyên sâu nếu cần thiết
3. Trình bày theo cấu trúc rõ ràng, dễ hiểu
4. Đưa ra kết luận hoặc khuyến nghị nếu phù hợp
5. Trả lời bằng tiếng Việt với ngôn ngữ chuyên nghiệp
6. KHÔNG sử dụng markdown, chỉ dùng văn bản thuần`
    }

    // Step 4: Process messages for Gemini with enhanced prompt
    const processedMessages = [...messages]
    processedMessages[processedMessages.length - 1] = {
      ...processedMessages[processedMessages.length - 1],
      content: enhancedPrompt,
    }

    // Step 5: Call Gemini
    console.log("[Perplexity-to-Gemini] Step 2: Processing with Gemini")
    const geminiModel = genAI.getGenerativeModel({ 
      model,
      systemInstruction: "Bạn là một AI assistant chuyên nghiệp. Hãy phân tích thông tin từ Perplexity AI và cung cấp câu trả lời chi tiết, chính xác. Sử dụng văn bản thuần, không markdown, không emoji. Định dạng rõ ràng với các gạch đầu dòng và đánh số khi cần thiết."
    })

    const result = await geminiModel.generateContent({
      contents: processedMessages.map((msg) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content) }],
      })),
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 2000,
      },
    })

    const response = await result.response
    let text = response.text()

    // Post-process: Clean up markdown and special characters
    text = text
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/#{1,6}\s*/g, '')
      .replace(/```[\s\S]*?```/g, '')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, '')
      .replace(/^\s*[-*+]\s+/gm, '• ')
      .replace(/^\s*\d+\.\s+/gm, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim()

    console.log("[Perplexity-to-Gemini] Completed:", {
      questionLength: userQuestion.length,
      answerLength: text.length,
      searchResults: !!searchResults
    })

    return NextResponse.json({
      candidates: [{
        content: {
          parts: [{ text }],
          role: "model",
        },
        finishReason: "STOP",
      }],
      choices: [{
        message: {
          role: "assistant",
          content: text,
        },
      }],
      searchResults,
      workflow: "perplexity-to-gemini",
      step1: {
        type: "perplexity_search",
        query: userQuestion,
        resultsCount: searchResults?.results?.length || 0,
        prompt: `Tìm kiếm thông tin: "${userQuestion}"`,
      },
      step2: {
        type: "gemini_processing",
        promptLength: enhancedPrompt.length,
        content: text.substring(0, 200) + "...",
      },
    })
  } catch (error: any) {
    console.error("Perplexity-to-Gemini error:", error)
    return NextResponse.json(
      {
        error: error?.message || "Failed to process Perplexity-to-Gemini workflow",
        errorType: "workflow_error",
      },
      { status: 500 },
    )
  }
}

// ChatGPT-to-Gemini workflow handler
async function handleChatGPTToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  if (!OPENAI_API_KEY || !GOOGLE_API_KEY) {
    return NextResponse.json(
      { error: "OpenAI or Google API key not configured" }, 
      { status: 500 }
    )
  }

  try {
    // Step 1: Extract user question
    const lastMessage = messages[messages.length - 1]
    const userQuestion = 
      typeof lastMessage.content === "string"
        ? lastMessage.content
        : lastMessage.content?.find((c: any) => c.type === "text")?.text || ""

    if (!userQuestion.trim()) {
      return NextResponse.json({ error: "No question provided" }, { status: 400 })
    }

    // Step 2: Call ChatGPT to get initial response
    console.log("[ChatGPT-to-Gemini] Step 1: Getting response from ChatGPT")
    const chatGPTResponse = await handleOpenAI(
      [...messages], // Pass all messages for context
      "gpt-3.5-turbo", // Use GPT-3.5 for cost efficiency
      false, // Don't stream
      files
    )

    if (!chatGPTResponse.ok) {
      const errorData = await chatGPTResponse.json()
      throw new Error(errorData.error || "Failed to get response from ChatGPT")
    }

    const chatGPTData = await chatGPTResponse.json()
    const chatGPTAnswer = chatGPTData.choices?.[0]?.message?.content || ""

    // Step 3: Enhance the prompt for Gemini
    const enhancedPrompt = `Bạn là một AI assistant chuyên nghiệp. Dưới đây là câu hỏi của người dùng và câu trả lời từ ChatGPT. 
Hãy phân tích, đánh giá và cải thiện câu trả lời này một cách chi tiết hơn:

CÂU HỎI: ${userQuestion}

CÂU TRẢ LỜI TỪ CHATGPT:
${chatGPTAnswer}

YÊU CẦU CẢI THIỆN:
1. Đánh giá tính chính xác của câu trả lời
2. Bổ sung thông tin chi tiết nếu cần thiết
3. Đưa ra các ví dụ minh họa cụ thể
4. Trình bày lại theo cấu trúc rõ ràng, dễ hiểu
5. Trả lời bằng tiếng Việt với ngôn ngữ chuyên nghiệp
6. KHÔNG sử dụng markdown, chỉ dùng văn bản thuần`

    // Step 4: Prepare messages for Gemini
    const geminiMessages = [
      {
        role: "user",
        content: enhancedPrompt,
      },
    ]

    // Step 5: Call Gemini with enhanced prompt
    console.log("[ChatGPT-to-Gemini] Step 2: Enhancing response with Gemini")
    const geminiResponse = await handleGoogle(
      geminiMessages,
      model,
      stream,
      files
    )

    if (!geminiResponse.ok) {
      const errorData = await geminiResponse.json()
      throw new Error(errorData.error || "Failed to get enhanced response from Gemini")
    }

    const geminiData = await geminiResponse.json()
    const enhancedAnswer = geminiData.choices?.[0]?.message?.content || ""

    console.log("[ChatGPT-to-Gemini] Completed:", {
      originalAnswerLength: chatGPTAnswer.length,
      enhancedAnswerLength: enhancedAnswer.length,
    })

    return NextResponse.json({
      candidates: geminiData.candidates,
      choices: geminiData.choices,
      workflow: "chatgpt-to-gemini",
      step1: {
        type: "chatgpt_response",
        query: userQuestion,
        responseLength: chatGPTAnswer.length,
        preview: chatGPTAnswer.substring(0, 200) + "...",
      },
      step2: {
        type: "gemini_enhancement",
        promptLength: enhancedPrompt.length,
        content: enhancedAnswer.substring(0, 200) + "...",
      },
    })
  } catch (error: any) {
    console.error("ChatGPT-to-Gemini error:", error)
    return NextResponse.json(
      {
        error: error?.message || "Failed to process ChatGPT-to-Gemini workflow",
        errorType: "workflow_error",
      },
      { status: 500 },
    )
  }
}

// Tavily-to-Gemini workflow handler
async function handleTavilyToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  if (!GOOGLE_API_KEY || !genAI) {
    return NextResponse.json({ error: "Google API key not configured" }, { status: 500 })
  }

  try {
    // Step 1: Extract user question
    const lastMessage = messages[messages.length - 1]
    const userQuestion =
      typeof lastMessage.content === "string"
        ? lastMessage.content
        : lastMessage.content?.find((c: any) => c.type === "text")?.text || ""

    if (!userQuestion.trim()) {
      return NextResponse.json({ error: "No question provided" }, { status: 400 })
    }

    // Step 2: Search with Tavily
    console.log("[Tavily-to-Gemini] Step 1: Searching with Tavily")
    const searchResults = await searchWithTavily(userQuestion)

    // Step 3: Build enhanced prompt with search results
    let enhancedPrompt = userQuestion

    if (searchResults && searchResults.results && searchResults.results.length > 0) {
      const searchContext = searchResults.results
        .map((result: any, index: number) => `${index + 1}. ${result.title}\n${result.content}\nNguồn: ${result.url}`)
        .join("\n\n")

      enhancedPrompt = `Bạn là một AI assistant chuyên nghiệp. Dựa trên thông tin tìm kiếm sau đây, hãy trả lời câu hỏi một cách chi tiết và chính xác:

THÔNG TIN TÌM KIẾM:
${searchContext}

CÂU HỎI: ${userQuestion}

HƯỚNG DẪN TRẢ LỜI:
- Phân tích và tổng hợp thông tin từ các nguồn đáng tin cậy
- Trình bày theo cấu trúc rõ ràng với các điểm chính
- Sử dụng danh sách đánh số (1. 2. 3.) khi liệt kê
- Đưa ra nhận xét hoặc phân tích sâu hơn nếu phù hợp
- Trả lời bằng tiếng Việt với ngôn ngữ chuyên nghiệp
- KHÔNG sử dụng ký hiệu **, ##, ###, hoặc bất kỳ markdown nào
- KHÔNG sử dụng emoji hoặc ký hiệu đặc biệt
- Chỉ sử dụng văn bản thuần túy với định dạng đơn giản`
    }

    // Step 4: Process messages for Gemini with enhanced prompt
    const processedMessages = [...messages]
    processedMessages[processedMessages.length - 1] = {
      ...processedMessages[processedMessages.length - 1],
      content: enhancedPrompt,
    }

    // Step 5: Call Gemini
    console.log("[Tavily-to-Gemini] Step 2: Calling Gemini")
    const geminiModel = genAI.getGenerativeModel({ 
      model,
      systemInstruction: "Bạn là một AI assistant chuyên nghiệp. Trả lời bằng tiếng Việt, sử dụng văn bản thuần túy không có markdown, không có ký hiệu **, ##, ###, không có emoji. Chỉ sử dụng định dạng đơn giản với danh sách đánh số (1. 2. 3.) khi liệt kê và bullet points (•) khi liệt kê danh sách."
    })

    const result = await geminiModel.generateContent({
      contents: processedMessages.map((msg) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content) }],
      })),
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 2000,
      },
    })

    const response = await result.response
    let text = response.text()

    // Post-process: Clean up markdown and special characters
    text = text
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove **bold**
      .replace(/\*(.*?)\*/g, '$1') // Remove *italic*
      .replace(/#{1,6}\s*/g, '') // Remove headers ###
      .replace(/```[\s\S]*?```/g, '') // Remove code blocks
      .replace(/`([^`]+)`/g, '$1') // Remove inline code
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove markdown links, keep text
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, '') // Remove images
      .replace(/^\s*[-*+]\s+/gm, '• ') // Convert markdown lists to bullet points
      .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered list markers
      .replace(/\n{3,}/g, '\n\n') // Reduce multiple newlines to double
      .replace(/^\s+|\s+$/g, '') // Trim whitespace
      .replace(/[^\u0000-\u007F\u00C0-\u017F\u1EA0-\u1EF9\u0102\u0103\u00C2\u00CA\u00D4\u00E2\u00EA\u00F4\u00C1\u00C9\u00CD\u00D3\u00DA\u00DD\u00E1\u00E9\u00ED\u00F3\u00FA\u00FD\u00C3\u00E3\u00C4\u00E4\u00C5\u00E5\u00C6\u00E6\u00C7\u00E7\u00C8\u00E8\u00CB\u00EB\u00CE\u00EE\u00CF\u00EF\u00D1\u00F1\u00D2\u00F2\u00D5\u00F5\u00D6\u00F6\u00D8\u00F8\u00D9\u00F9\u00DC\u00FC\u00DF]/g, '') // Keep only basic Latin and Vietnamese characters
      .trim()

    console.log("[Tavily-to-Gemini] Completed:", {
      textLength: text.length,
      searchResultsCount: searchResults?.results?.length || 0,
      cleanedText: text.substring(0, 200) + "...",
    })

    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      choices: [
        {
          message: {
            role: "assistant",
            content: text,
          },
        },
      ],
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

async function handleOpenAI(messages: any[], model: string, stream: boolean, files: any[]) {
  if (!OPENAI_API_KEY) {
    return NextResponse.json({ error: "OpenAI API key not configured" }, { status: 500 })
  }

  // Process files for OpenAI (supports images, not PDFs directly)
  const processedMessages = await processMessagesForOpenAI(messages, files)

  const requestBody = {
    model,
    messages: processedMessages,
    stream,
    temperature: 0.7,
    max_tokens: 2000,
  }

  console.log("Sending request to OpenAI:", {
    model,
    messagesCount: processedMessages.length,
    filesCount: files.length,
  })

  try {
    const response = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      console.error("OpenAI API error:", errorData)

      if (errorData.error?.code === "insufficient_quota") {
        return NextResponse.json(
          {
            error: "OpenAI quota exceeded. Please check your billing or try using Gemini instead.",
            errorType: "quota_exceeded",
            fallbackSuggestion: "gemini",
          },
          { status: 402 },
        )
      }

      if (errorData.error?.code === "invalid_api_key") {
        return NextResponse.json(
          {
            error: "Invalid OpenAI API key. Please check your configuration.",
            errorType: "invalid_key",
          },
          { status: 401 },
        )
      }

      return NextResponse.json(
        {
          error: errorData.error?.message || "Failed to get response from OpenAI",
          errorType: "api_error",
        },
        { status: response.status },
      )
    }

    if (stream) {
      return new Response(response.body, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      })
    } else {
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    console.error("OpenAI request failed:", error)
    return NextResponse.json(
      {
        error: "Network error when calling OpenAI API",
        errorType: "network_error",
      },
      { status: 500 },
    )
  }
}

async function handleGoogle(messages: any[], model: string, stream: boolean, files: any[]) {
  if (!GOOGLE_API_KEY || !genAI) {
    return NextResponse.json({ error: "Google API key not configured" }, { status: 500 })
  }

  try {
    console.log("messages 👉", messages, files)
    const uploadedFiles = await uploadFilesToGeminiSDK(files)

    const { processedContents, prompts } = await processMessagesForGoogleSDK(messages, files, uploadedFiles)

    console.log("Sending request to Google Gemini SDK:", {
      model,
      contentsCount: processedContents.length,
      filesCount: files.length,
      uploadedFilesCount: uploadedFiles.length,
    })

    // 1. Khai báo công cụ bạn muốn sử dụng
    const tools = [
      {
        googleSearchRetrieval: {}, // Cú pháp cho Google Search trong Node.js/JS
      },
    ]

    const geminiModel = genAI.getGenerativeModel({ model })

    const result = await geminiModel.generateContent({
      contents: processedContents,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1000,
      },
    })

    const response = await result.response
    const text = response.text()

    console.log("Google Gemini SDK response received:", {
      textLength: text.length,
      textPreview: text.substring(0, 200) + "...",
    })

    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: response.usageMetadata,
      // Keep compatibility with frontend
      choices: [
        {
          message: {
            role: "assistant",
            content: text,
          },
        },
      ],
      prompts,
    })
  } catch (error: any) {
    console.error("Google GenAI SDK error:", error)

    if (error?.message?.includes("API key")) {
      return NextResponse.json(
        {
          error: "Invalid Google API key. Please check your configuration.",
          errorType: "invalid_key",
        },
        { status: 401 },
      )
    }

    if (error?.message?.includes("quota") || error?.message?.includes("limit")) {
      return NextResponse.json(
        {
          error: "Google API quota exceeded. Please try again later.",
          errorType: "quota_exceeded",
        },
        { status: 429 },
      )
    }

    return NextResponse.json(
      {
        error: error?.message || "Failed to get response from Google Gemini",
        errorType: "api_error",
      },
      { status: 500 },
    )
  }
}

async function uploadFilesToGeminiSDK(files: any[]) {
  if (!genAI) return []

  const uploadedFiles = []

  for (const file of files) {
    // Upload Office files and PDFs via Files API
    const needsUpload = [
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
      "application/msword", // .doc
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
      "application/vnd.ms-excel", // .xls
      "application/vnd.openxmlformats-officedocument.presentationml.presentation", // .pptx
      "application/pdf", // PDF files
    ].includes(file.type)

    if (needsUpload) {
      try {
        console.log(`Uploading ${file.name} to Gemini Files API using SDK...`)

        const buffer = Buffer.from(file.data, "base64")

        // @ts-ignore - files API may not be typed in all SDK versions
        const uploadResult = await (genAI as any).files.upload({
          file: buffer,
          mimeType: file.type,
          displayName: file.name,
        })

        console.log(`Successfully uploaded ${file.name}:`, {
          uri: uploadResult.file.uri,
          name: uploadResult.file.name,
          mimeType: uploadResult.file.mimeType,
        })

        uploadedFiles.push({
          originalFile: file,
          fileUri: uploadResult.file.uri,
          name: uploadResult.file.name,
          mimeType: uploadResult.file.mimeType,
        })
      } catch (error) {
        console.error(`Error uploading ${file.name}:`, error)
      }
    }
  }

  return uploadedFiles
}

// --- Reusable: chuẩn hoá + thêm/sửa message + upload & attach files ---
async function prepareMessagesForGemini(
  messages: any[],
  files: any[],
  {
    systemPrompt,
    replaceLastUser,
    attachFiles = true,
    inlineFallback = true,
  }: {
    systemPrompt?: string // sẽ trả ra qua field systemInstruction (KHÔNG thêm vào contents)
    replaceLastUser?: string
    attachFiles?: boolean
    inlineFallback?: boolean
  } = {},
) {
  const normalizeMessage = (msg: any) => {
    // Loại bỏ mọi message role=system trong contents để tránh 400
    if (msg?.role === "system") return null

    if (msg?.parts && Array.isArray(msg.parts)) return msg

    if (typeof msg?.content === "string") {
      return { role: msg.role, parts: [{ text: msg.content }] }
    }

    if (Array.isArray(msg?.content)) {
      const parts = msg.content
        .map((c: any) => {
          if (c?.type === "text" && typeof c.text === "string") return { text: c.text }
          if (c?.inlineData?.data && c?.inlineData?.mimeType)
            return {
              inlineData: {
                data: c.inlineData.data,
                mimeType: c.inlineData.mimeType,
              },
            }
          if (c?.fileData?.fileUri && c?.fileData?.mimeType)
            return {
              fileData: {
                fileUri: c.fileData.fileUri,
                mimeType: c.fileData.mimeType,
              },
            }
          return null
        })
        .filter(Boolean)
      return { role: msg.role, parts }
    }

    return { role: msg.role, parts: [] }
  }

  // 1) Normalize & loại system khỏi contents
  const arr = (messages || []).map(normalizeMessage).filter(Boolean)

  // 2) Optional: replace last user
  if (replaceLastUser) {
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i].role === "user") {
        arr[i] = { ...arr[i], parts: [{ text: replaceLastUser }] }
        break
      }
    }
  }

  // 3) Upload & attach files vào last user
  let uploadedFiles: any[] = []
  let fileParts: any[] = []
  if (attachFiles && files?.length) {
    try {
      uploadedFiles = await uploadFilesToGeminiSDK(files)
      if (uploadedFiles?.length) {
        fileParts = uploadedFiles
          .map((f: any) => {
            const uri = f?.fileUri || f?.uri || f?.name // "files/xxx" cũng OK
            const mime = f?.mimeType
            if (!uri || !mime) return null
            return { fileData: { fileUri: uri, mimeType: mime } }
          })
          .filter(Boolean)
      }
    } catch (e) {
      console.error("prepareMessagesForGemini: upload error, fallback inline if enabled", e)
    }

    if (!fileParts.length && inlineFallback) {
      const inlineParts = []
      for (const orig of files) {
        try {
          const mimeType = orig.type || orig.mimeType || "application/octet-stream"
          if (orig?.arrayBuffer) {
            const ab = await orig.arrayBuffer()
            inlineParts.push({
              inlineData: {
                mimeType,
                data: Buffer.from(ab as any).toString("base64"),
              },
            })
          } else if (orig?.buffer) {
            inlineParts.push({
              inlineData: {
                mimeType,
                data: Buffer.from(orig.buffer).toString("base64"),
              },
            })
          } else if (orig?.path) {
            const { readFileSync } = await import("node:fs")
            inlineParts.push({
              inlineData: {
                mimeType,
                data: readFileSync(orig.path).toString("base64"),
              },
            })
          }
        } catch (e) {
          console.error("prepareMessagesForGemini: inline fallback failed", e)
        }
      }
      fileParts = inlineParts
    }

    if (fileParts.length) {
      for (let i = arr.length - 1; i >= 0; i--) {
        if (arr[i].role === "user") {
          arr[i] = {
            ...arr[i],
            parts: [...(arr[i].parts || []), ...fileParts],
          }
          break
        }
      }
    }
  }

  // 4) Trả contents + systemInstruction (KHÔNG thêm system vào contents)
  const contents = arr.map((m) => ({ role: m.role, parts: m.parts }))

  return {
    contents,
    systemInstruction: systemPrompt || undefined, // dùng ở bước getGenerativeModel hoặc generateContent
    uploadedFiles,
    attachedPartsCount: fileParts.length,
  }
}

async function processMessagesForOpenAI(messages: any[], files: any[]) {
  const processedMessages = [...messages]

  // Add files to the last user message for OpenAI
  if (files.length > 0 && processedMessages.length > 0) {
    const lastMessage = processedMessages[processedMessages.length - 1]
    if (lastMessage.role === "user") {
      // Convert content to array format if it's a string
      if (typeof lastMessage.content === "string") {
        lastMessage.content = [
          {
            type: "text",
            text: lastMessage.content,
          },
        ]
      }

      // Add supported files (images only for OpenAI)
      files.forEach((file: any) => {
        if (file.type.startsWith("image/")) {
          lastMessage.content.push({
            type: "image_url",
            image_url: {
              url: `data:${file.type};base64,${file.data}`,
            },
          })
        } else if (file.type === "application/pdf") {
          // For PDFs, add a text note since OpenAI doesn't support direct PDF processing
          lastMessage.content.push({
            type: "text",
            text: `[PDF File: ${file.name} - Note: OpenAI cannot directly process PDF files. Please extract text content manually.]`,
          })
        }
      })
    }
  }

  return processedMessages
}

const CONTENT_SYSTEM: string = `
Bạn là Chatbot ORS. Nhiệm vụ: nhận câu hỏi của user, sinh ra prompt đơn giản cho Gemini.
LUỒNG XỬ LÝ:
User hỏi → Bạn tạo prompt → Prompt gửi cho Gemini → Gemini trả lời
QUY TẮC SINH PROMPT:

1. Nếu user chỉ chào hỏi (hi, hello, xin chào) → Trả lời: "Xin chào! Tôi có thể giúp gì cho bạn?"

2. Nếu user hỏi thông tin → Sinh prompt theo mẫu này:
"Trả lời câu hỏi: [câu hỏi user]

Yêu cầu:
- Không dùng ký hiệu đặc biệt, không markdown
- Có số liệu báo cáo nếu có
- Hiển thị ý chính đúng trọng tâm, ví dụ: đánh số 1. 2. 3.
- Mỗi ý: 1 câu tóm tắt + 1-2 câu giải thích
- Nếu có chỉ dẫn nguồn, số liệu thì Cuối mỗi ý ghi nguồn: <br /><strong>Nguồn:</strong> <a href='[link]' target='_blank'>[tên]</a>
Không có nguồn thì không ghi.
- Viết bằng tiếng Việt"

CHÚ Ý:
- [chủ đề] = thay bằng lĩnh vực phù hợp (tài chính, giáo dục, y tế, ngân hàng...)
- [câu hỏi user] = copy y nguyên câu hỏi của user
- Chỉ xuất prompt, không giải thích gì thêm
`

/**
 * @param contentText
 * @returns
 */
async function convertToPromptChatGPT(contentText: string): Promise<string> {
  const systemWithQuestion = CONTENT_SYSTEM.replace("{user_question}", contentText.trim())
  const promptGenerationMessages = [
    {
      role: "system",
      content: systemWithQuestion,
    },
    { role: "user", content: contentText },
  ]

  const chatgptResponse = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: promptGenerationMessages,
      temperature: 0, // ít “sáng tác”
      top_p: 0, // chặt chẽ hơn
      max_tokens: 400,
    }),
  })

  if (!chatgptResponse.ok) {
    const err = await chatgptResponse.json().catch(() => ({}))
    console.error("❌ ChatGPT API error:", err)
    return contentText
  }

  const chatgptData = await chatgptResponse.json()
  const optimizedPrompt = String(chatgptData?.choices?.[0]?.message?.content || contentText)

  console.log("Prompt from ChatGPT ✨:", optimizedPrompt)
  return optimizedPrompt
}

async function processMessagesForGoogleSDK(messages: any[], files: any[], uploadedFiles: any[] = []) {
  const processedContents = []
  const prompts = []

  for (const message of messages) {
    const role = message.role === "assistant" ? "model" : "user"
    const parts = []

    // Add text content
    if (typeof message.content === "string") {
      console.log("Processing message content (string):", {
        role,
        contentLength: message.content.length,
      })
      const contentText = await convertToPromptChatGPT(message.content)
      parts.push({ text: contentText })
      prompts.push({
        input: { system: CONTENT_SYSTEM, user: message.content },
        output: contentText,
      })
    } else if (Array.isArray(message.content)) {
      console.log("Processing message content (array):", {
        role,
        contentItems: message.content.length,
        contentTypes: (message.content as any[]).map((c: any) => c.type),
      })

      message.content.forEach(async (content: any) => {
        // prompts.push({ input: null, output: null });
        if (content.type === "text") {
          // Convert to prompt
          // const contentText = await convertToPromptChatGPT(content.text);
          parts.push({ text: content.text })
        } else if (content.type === "image_url" && content.image_url?.url) {
          // Extract mime type and base64 data from data URL
          const dataUrl = content.image_url.url
          const matches = (dataUrl as string).match(/^data:([^;]+);base64,(.+)$/)

          if (matches) {
            const [, mimeType, base64Data] = matches
            console.log("Adding image part:", { mimeType })
            ;(parts as any).push({
              inlineData: {
                mimeType: mimeType,
                data: base64Data,
              },
            })
          }
        }
      })
    }

    if (parts.length > 0) {
      processedContents.push({
        role,
        parts,
      })
    }
  }

  if (uploadedFiles.length > 0 && processedContents.length > 0) {
    const lastContent = processedContents[processedContents.length - 1]
    if (lastContent.role === "user") {
      uploadedFiles.forEach((uploadedFile: any) => {
        console.log("Adding uploaded file part:", {
          name: uploadedFile.name,
          uri: uploadedFile.fileUri,
          mimeType: uploadedFile.mimeType,
        })
        ;(lastContent.parts as any).push({
          fileData: {
            mimeType: uploadedFile.mimeType,
            fileUri: uploadedFile.fileUri,
          },
        })
      })
    }
  }

  // Add inline images (not uploaded files)
  if (files.length > 0 && processedContents.length > 0) {
    const lastContent = processedContents[processedContents.length - 1]
    if (lastContent.role === "user") {
      files.forEach((file: any) => {
        // Only use inlineData for images (Office files are uploaded)
        if (file.type.startsWith("image/")) {
          console.log("Adding inline image:", { type: file.type })
          ;(lastContent.parts as any).push({
            inlineData: {
              mimeType: file.type,
              data: file.data,
            },
          })
        }
      })
    }
  }

  return { processedContents, prompts }
}
