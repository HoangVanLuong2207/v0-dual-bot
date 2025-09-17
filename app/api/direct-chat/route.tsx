import { type NextRequest, NextResponse } from "next/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY
const TAVILY_API_KEY = process.env.TAVILY_API_KEY

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

// Function to get or create genAI instance with current GOOGLE_API_KEY
function getGenAI() {
  if (!GOOGLE_API_KEY) return null;
  return new GoogleGenerativeAI(GOOGLE_API_KEY);
}

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

    // Workflow: chatgpt-to-gemini (ChatGPT generates prompt, then ask Gemini)
    if (workflow === "chatgpt-to-gemini") {
      return await handleChatGPTToGemini(messages, actualModel, stream, files)
    }

    // Workflow: perplexity-to-gemini (Perplexity search, then ask Gemini)
    if (workflow === "perplexity-to-gemini") {
      return await handlePerplexityToGemini(messages, actualModel, stream, files)
    }
    
    // Workflow: perplexity-to-chatgpt-to-gemini (Perplexity search, then ChatGPT refines, then Gemini responds)
    if (workflow === "perplexity-chatgpt-gemini") {
      return await handlePerplexityChatGPTToGemini(messages, actualModel, stream, files)
    }

    // Default: direct Gemini processing
    return await handleGoogle(messages, actualModel, stream, files)
  } catch (error) {
    console.error("Direct Chat API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// Tavily-to-Gemini workflow handler
async function handleTavilyToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  const genAI = getGenAI();
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

    let conversationContext = ""
    if (messages.length > 1) {
      const previousMessages = messages.slice(0, -1)
      conversationContext = previousMessages
        .map((msg: any, index: number) => {
          const role = msg.role === "user" ? "Người dùng" : "Trợ lý"
          const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
          return `${role}: ${content}`
        })
        .join("\n")
    }

    // Step 2: Search with Tavily
    console.log("[Tavily-to-Gemini] Step 1: Searching with Tavily")
    const searchResults = await searchWithTavily(userQuestion)

    // Step 3: Build enhanced prompt with search results and conversation context
    let enhancedPrompt = userQuestion

    if (searchResults && searchResults.results && searchResults.results.length > 0) {
      const searchContext = searchResults.results
        .map((result: any, index: number) => `${index + 1}. ${result.title}\n${result.content}\nNguồn: ${result.url}`)
        .join("\n\n")

      enhancedPrompt = `Bạn là một AI assistant thông minh và hữu ích. Dựa trên thông tin tìm kiếm mới nhất và ngữ cảnh cuộc trò chuyện, hãy trả lời câu hỏi một cách chi tiết, chính xác và có cấu trúc:

${conversationContext ? `NGỮ CẢNH CUỘC TRÒ CHUYỆN:\n${conversationContext}\n\n` : ""}THÔNG TIN TÌM KIẾM:
${searchContext}

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

    const processedMessages = messages.map((msg, index) => {
      if (index === messages.length - 1) {
        // Replace last message with enhanced prompt
        return {
          ...msg,
          content: enhancedPrompt,
        }
      }
      return msg
    })

    // Step 4: Call Gemini
    console.log("[Tavily-to-Gemini] Step 2: Calling Gemini")
    const genAI = getGenAI();
    if (!genAI) {
      throw new Error("Google AI SDK initialization failed");
    }
    const geminiModel = genAI.getGenerativeModel({ model })

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
    
    // Clean up any remaining markdown characters
    const cleanText = text
      .replace(/\*\*(.*?)\*\*/g, '$1')  // Remove bold
      .replace(/\*(.*?)\*/g, '$1')       // Remove italic
      .replace(/`(.*?)`/g, '$1')          // Remove inline code
      .replace(/^#+\s+/gm, '')           // Remove headings
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')  // Remove links but keep text
      .replace(/^[-*+]\s+/gm, '')        // Remove list markers
      .replace(/\n{3,}/g, '\n\n')       // Normalize multiple newlines
      .trim();

    console.log("[Tavily-to-Gemini] Completed:", {
      originalLength: text.length,
      cleanedLength: cleanText.length,
      searchResultsCount: searchResults?.results?.length || 0,
    })

    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text: cleanText }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      choices: [
        {
          message: {
            role: "assistant",
            content: cleanText,
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
  const genAI = getGenAI();
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
      conversationLength: messages.length, // Log conversation length
    })

    // 1. Khai báo công cụ bạn muốn sử dụng
    const tools = [
      {
        googleSearchRetrieval: {}, // Cú pháp cho Google Search trong Node.js/JS
      },
    ]

    // Create system instruction with clear formatting rules
    const systemInstruction = {
      role: 'user',
      parts: [{
        text: 'Bạn là một trợ lý AI hữu ích. Khi trả lời câu hỏi, vui lòng tuân thủ các yêu cầu sau:\n' +
          '1. Không sử dụng bất kỳ định dạng markdown nào (không **, ##, ```, v.v.)\n' +
          '2. Trả lời bằng văn bản thuần, không cần xuống dòng thừa\n' +
          '3. Sử dụng các dấu gạch đầu dòng (-) thay vì đánh số nếu cần liệt kê\n' +
          '4. Trả lời bằng tiếng Việt\n' +
          '5. Giữ câu trả lời ngắn gọn, súc tích\n\n' +
          'LƯU Ý QUAN TRỌNG: Khi trả lời, TUYỆT ĐỐI KHÔNG sử dụng bất kỳ định dạng markdown nào. ' +
          'Chỉ trả lời bằng văn bản thuần. Nếu cần liệt kê, hãy dùng dấu gạch đầu dòng (-) thay vì đánh số.'
      }]
    };

    const geminiModel = genAI.getGenerativeModel({ 
      model,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1000,
      }
    });

    // Add system instruction as the first message
    const processedMessages = [systemInstruction, ...processedContents];

    const result = await geminiModel.generateContent({
      contents: processedMessages,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1000,
      },
    })

    const response = await result.response
    let text = response.text()
    
    // Clean up any remaining markdown characters
    const cleanText = text
      .replace(/\*\*(.*?)\*\*/g, '$1')  // Remove bold
      .replace(/\*(.*?)\*/g, '$1')       // Remove italic
      .replace(/`(.*?)`/g, '$1')          // Remove inline code
      .replace(/^#+\s+/gm, '')           // Remove headings
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')  // Remove links but keep text
      .replace(/^[-*+]\s+/gm, '')        // Remove list markers
      .replace(/\n{3,}/g, '\n\n')       // Normalize multiple newlines
      .trim();

    console.log("Google Gemini SDK response received:", {
      originalLength: text.length,
      cleanedLength: cleanText.length,
      textPreview: cleanText.substring(0, 200) + (cleanText.length > 200 ? "..." : ""),
    })

    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text: cleanText }],
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
            content: cleanText,
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

// ChatGPT-to-Gemini workflow handler
async function handleChatGPTToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  const genAI = getGenAI();
  if (!GOOGLE_API_KEY || !genAI) {
    return NextResponse.json({ error: "Google API key not configured" }, { status: 500 })
  }

  if (!OPENAI_API_KEY) {
    return NextResponse.json({ error: "OpenAI API key not configured for ChatGPT-to-Gemini workflow" }, { status: 500 })
  }

  try {
    // Step 1: Extract user question and conversation context
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
        .map((msg: any, index: number) => {
          const role = msg.role === "user" ? "Người dùng" : "Trợ lý"
          const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
          return `${role}: ${content}`
        })
        .join("\n")
    }

    // Step 2: Use ChatGPT to generate optimized prompt
    console.log("[ChatGPT-to-Gemini] Step 1: Generating prompt with ChatGPT")
    const optimizedPrompt = await convertToPromptChatGPT(userQuestion)

    // Step 3: Build enhanced prompt with conversation context
    let enhancedPrompt = optimizedPrompt

    if (conversationContext) {
      enhancedPrompt = `NGỮ CẢNH CUỘC TRÒ CHUYỆN:
${conversationContext}

CÂU HỎI HIỆN TẠI: ${userQuestion}

PROMPT ĐÃ TỐI ƯU:
${optimizedPrompt}`
    }

    const processedMessages = messages.map((msg, index) => {
      if (index === messages.length - 1) {
        // Replace last message with enhanced prompt
        return {
          ...msg,
          content: enhancedPrompt,
        }
      }
      return msg
    })

    // Step 4: Call Gemini with optimized prompt
    console.log("[ChatGPT-to-Gemini] Step 2: Calling Gemini with optimized prompt")
    const genAI = getGenAI();
    if (!genAI) {
      throw new Error("Google AI SDK initialization failed");
    }
    const geminiModel = genAI.getGenerativeModel({ model })

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
    const text = response.text()

    console.log("[ChatGPT-to-Gemini] Completed:", {
      textLength: text.length,
      optimizedPromptLength: optimizedPrompt.length,
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
      workflow: "chatgpt-to-gemini",
      step1: {
        type: "chatgpt_prompt_generation",
        originalQuery: userQuestion,
        optimizedPrompt: optimizedPrompt.substring(0, 200) + "...",
        promptLength: optimizedPrompt.length,
      },
      step2: {
        type: "gemini_response",
        promptLength: enhancedPrompt.length,
        content: text.substring(0, 200) + "...",
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

// Perplexity-to-Gemini workflow handler
async function handlePerplexityToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  const genAI = getGenAI();
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

    let conversationContext = ""
    if (messages.length > 1) {
      const previousMessages = messages.slice(0, -1)
      conversationContext = previousMessages
        .map((msg: any, index: number) => {
          const role = msg.role === "user" ? "Người dùng" : "Trợ lý"
          const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
          return `${role}: ${content}`
        })
        .join("\n")
    }

    // Step 2: Search with Perplexity API (simulated - using a comprehensive search approach)
    console.log("[Perplexity-to-Gemini] Step 1: Searching with Perplexity-style approach")
    const searchResults = await searchWithPerplexityStyle(userQuestion)

    // Step 3: Build enhanced prompt with search results and conversation context
    let enhancedPrompt = userQuestion

    if (searchResults && searchResults.results && searchResults.results.length > 0) {
      const searchContext = searchResults.results
        .map((result: any, index: number) => `${index + 1}. ${result.title}\n${result.content}\nNguồn: ${result.url}`)
        .join("\n\n")

      enhancedPrompt = `Bạn là một AI assistant thông minh và hữu ích với khả năng phân tích sâu. Dựa trên thông tin tìm kiếm toàn diện và ngữ cảnh cuộc trò chuyện, hãy trả lời câu hỏi một cách chi tiết, chính xác và có cấu trúc:

${conversationContext ? `NGỮ CẢNH CUỘC TRÒ CHUYỆN:\n${conversationContext}\n\n` : ""}THÔNG TIN TÌM KIẾM TOÀN DIỆN:
${searchContext}

CÂU HỎI HIỆN TẠI: ${userQuestion}

YÊU CẦU TRẢ LỜI:
- Phân tích và tổng hợp thông tin từ nhiều nguồn đáng tin cậy
- Kết hợp với ngữ cảnh cuộc trò chuyện trước đó nếu có liên quan
- Trình bày theo cấu trúc rõ ràng với các điểm chính được đánh số
- Đưa ra phân tích sâu và nhận xét chuyên môn
- Trích dẫn nguồn cụ thể cho từng thông tin quan trọng
- So sánh các quan điểm khác nhau nếu có
- Đưa ra kết luận hoặc khuyến nghị nếu phù hợp
- Trả lời bằng tiếng Việt với ngôn ngữ chuyên nghiệp
- Không sử dụng ký hiệu đặc biệt hoặc markdown formatting`
    }

    const processedMessages = messages.map((msg, index) => {
      if (index === messages.length - 1) {
        // Replace last message with enhanced prompt
        return {
          ...msg,
          content: enhancedPrompt,
        }
      }
      return msg
    })

    // Step 4: Call Gemini
    console.log("[Perplexity-to-Gemini] Step 2: Calling Gemini")
    const genAI = getGenAI();
    if (!genAI) {
      throw new Error("Google AI SDK initialization failed");
    }
    const geminiModel = genAI.getGenerativeModel({ model })

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
    
    // Clean up any remaining markdown characters
    const cleanText = text
      .replace(/\*\*(.*?)\*\*/g, '$1')  // Remove bold
      .replace(/\*(.*?)\*/g, '$1')       // Remove italic
      .replace(/`(.*?)`/g, '$1')          // Remove inline code
      .replace(/^#+\s+/gm, '')           // Remove headings
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')  // Remove links but keep text
      .replace(/^[-*+]\s+/gm, '')        // Remove list markers
      .replace(/\n{3,}/g, '\n\n')       // Normalize multiple newlines
      .trim();

    console.log("[Perplexity-to-Gemini] Completed:", {
      originalLength: text.length,
      cleanedLength: cleanText.length,
      searchResultsCount: searchResults?.results?.length || 0,
    })

    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text: cleanText }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      choices: [
        {
          message: {
            role: "assistant",
            content: cleanText,
          },
        },
      ],
      searchResults,
      workflow: "perplexity-to-gemini",
      step1: {
        type: "perplexity_search",
        query: userQuestion,
        resultsCount: searchResults?.results?.length || 0,
        prompt: `Tìm kiếm toàn diện về: "${userQuestion}"`,
      },
      step2: {
        type: "gemini_response",
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

// Perplexity-to-ChatGPT-to-Gemini workflow handler
async function handlePerplexityChatGPTToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  const genAI = getGenAI();
  if (!GOOGLE_API_KEY || !genAI || !OPENAI_API_KEY) {
    return NextResponse.json({ error: "API keys not configured" }, { status: 500 })
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

    // Step 2: Search with Perplexity-style search
    console.log("[Perplexity-ChatGPT-Gemini] Step 1: Searching with Perplexity-style approach")
    const searchResults = await searchWithPerplexityStyle(userQuestion)

    // Step 3: Prepare context for ChatGPT
    let searchContext = ""
    if (searchResults?.results?.length > 0) {
      searchContext = searchResults.results
        .map((result: any, index: number) => 
          `${index + 1}. ${result.title}\n${result.content}\nNguồn: ${result.url}`
        )
        .join("\n\n")
    }

    // Step 4: Call ChatGPT to refine the prompt
    console.log("[Perplexity-ChatGPT-Gemini] Step 2: Refining with ChatGPT")
    const chatGPTResponse = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-4",
        messages: [
          {
            role: "system",
            content: `Bạn là một trợ lý AI giúp cải thiện câu hỏi dựa trên ngữ cảnh tìm kiếm. 
            Hãy tạo một prompt rõ ràng, chi tiết cho Gemini dựa trên câu hỏi và thông tin tìm kiếm.`,
          },
          {
            role: "user",
            content: `THÔNG TIN TÌM KIẾM:
            ${searchContext}
            
            CÂU HỎI GỐC: ${userQuestion}
            
            YÊU CẦU:
            1. Tạo một prompt chi tiết cho Gemini dựa trên thông tin tìm kiếm
            2. Giữ nguyên ngữ cảnh quan trọng từ câu hỏi gốc
            3. Thêm hướng dẫn rõ ràng về định dạng câu trả lời
            4. Yêu cầu Gemini trả lời bằng tiếng Việt
            5. Không sử dụng bất kỳ định dạng markdown nào trong prompt
            
            Chỉ trả về prompt đã tối ưu, không cần giải thích thêm.`,
          },
        ],
        temperature: 0.7,
        max_tokens: 1000,
      }),
    })

    if (!chatGPTResponse.ok) {
      throw new Error("Failed to get response from ChatGPT")
    }

    const chatGPTData = await chatGPTResponse.json()
    const refinedPrompt = chatGPTData.choices[0]?.message?.content || userQuestion

    // Step 5: Call Gemini with the refined prompt
    console.log("[Perplexity-ChatGPT-Gemini] Step 3: Calling Gemini with refined prompt")
    const genAI = getGenAI();
    if (!genAI) {
      throw new Error("Google AI SDK initialization failed");
    }
    
    const geminiModel = genAI.getGenerativeModel({ 
      model,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 2000,
      },
    })
    const result = await geminiModel.generateContent({
      contents: [{
        role: "user",
        parts: [{ text: refinedPrompt }]
      }]
    })
    
    const response = await result.response
    let text = response.text()
    
    // Clean up any remaining markdown characters
    const cleanText = text
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/`(.*?)`/g, '$1')
      .replace(/^#+\s+/gm, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/^[-*+]\s+/gm, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim()

    console.log("[Perplexity-ChatGPT-Gemini] Completed:", {
      originalLength: text.length,
      cleanedLength: cleanText.length,
      searchResultsCount: searchResults?.results?.length || 0,
    })

    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text: cleanText }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      choices: [
        {
          message: {
            role: "assistant",
            content: cleanText,
          },
        },
      ],
      searchResults,
      workflow: "perplexity-chatgpt-gemini",
      steps: {
        step1: {
          type: "perplexity_search",
          query: userQuestion,
          resultsCount: searchResults?.results?.length || 0,
        },
        step2: {
          type: "chatgpt_refinement",
          prompt: refinedPrompt,
        },
        step3: {
          type: "gemini_response",
          content: cleanText.substring(0, 200) + (cleanText.length > 200 ? "..." : ""),
        },
      },
    })
  } catch (error: any) {
    console.error("Perplexity-ChatGPT-Gemini error:", error)
    return NextResponse.json(
      {
        error: error?.message || "Failed to process Perplexity-ChatGPT-Gemini workflow",
        errorType: "workflow_error",
      },
      { status: 500 },
    )
  }
}

// Perplexity-style search function (using Tavily with enhanced parameters)
async function searchWithPerplexityStyle(query: string) {
  if (!TAVILY_API_KEY) {
    console.warn("Tavily API key not configured, skipping Perplexity-style search")
    return null
  }

  try {
    console.log("[Perplexity-style] Searching for:", query.substring(0, 100) + "...")

    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        api_key: TAVILY_API_KEY,
        query: query,
        search_depth: "advanced", // More comprehensive search
        include_answer: true,
        include_raw_content: true, // Include more detailed content
        max_results: 8, // More results for comprehensive analysis
        include_domains: [],
        exclude_domains: [],
        include_images: false,
        topic: "general", // Can be adjusted based on query type
      }),
    })

    if (!response.ok) {
      console.error("Perplexity-style search API error:", response.status, response.statusText)
      return null
    }

    const data = await response.json()
    console.log("[Perplexity-style] Search completed:", {
      resultsCount: data.results?.length || 0,
      hasAnswer: !!data.answer,
      hasRawContent: data.results?.some((r: any) => r.raw_content) || false,
    })

    return data
  } catch (error) {
    console.error("Perplexity-style search error:", error)
    return null
  }
}

async function uploadFilesToGeminiSDK(files: any[]) {
  const genAI = getGenAI();
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

const CONTENT_SYSTEM = `
Bạn là Chatbot ORS. Nhiệm vụ: nhận câu hỏi của user, sinh ra prompt đơn giản cho Gemini.
LUỒNG XỬ LÝ:
User hỏi → Bạn tạo prompt → Prompt gửi cho Gemini → Gemini trả lời
QUY TẮC SINH PROMPT:

1. Nếu user chỉ chào hỏi (hi, hello, xin chào) → Trả lời: "Xin chào! Tôi có thể giúp gì cho bạn?"

2. Nếu user hỏi thông tin → Sinh prompt theo mẫu này:
"Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi sau một cách rõ ràng, dễ hiểu:\n\nCâu hỏi: [câu hỏi user]\n\nYêu cầu:\n- Không sử dụng bất kỳ ký hiệu markdown nào như **, ##, \`\`\`, v.v.\n- Trình bày thông tin rõ ràng, mạch lạc\n- Sử dụng các số thứ tự (1, 2, 3) để liệt kê các ý chính\n- Mỗi ý chính nên có phần tóm tắt ngắn gọn và giải thích chi tiết\n- Nếu có nguồn tham khảo, hãy ghi rõ ở cuối câu trả lời\n- Luôn trả lời bằng tiếng Việt\n- Tuyệt đối không sử dụng bất kỳ ký hiệu đặc biệt nào để định dạng văn bản"

CHÚ Ý:
- [câu hỏi user] = copy y nguyên câu hỏi của user
- Chỉ xuất prompt, không giải thích gì thêm
- Đảm bảo prompt yêu cầu Gemini không sử dụng bất kỳ định dạng markdown nào
`;

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
  const processedContents: Array<{
    role: string;
    parts: Array<{
      text?: string;
      inlineData?: { mimeType: string; data: string };
      fileData?: { mimeType: string; fileUri: string };
    }>;
  }> = [];
  
  const prompts: Array<{ input: any; output: string }> = [];

  for (const message of messages) {
    const role = message.role === "assistant" ? "model" : "user"
    const parts: Array<{
      text?: string;
      inlineData?: { mimeType: string; data: string };
      fileData?: { mimeType: string; fileUri: string };
    }> = [];

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
            parts.push({
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

  // Add uploaded files to the last user message
  if (uploadedFiles.length > 0 && processedContents.length > 0) {
    const lastContent = processedContents[processedContents.length - 1]
    if (lastContent.role === "user") {
      uploadedFiles.forEach((uploadedFile: any) => {
        console.log("Adding uploaded file part:", {
          name: uploadedFile.name,
          uri: uploadedFile.fileUri,
          mimeType: uploadedFile.mimeType,
        })
        lastContent.parts.push({
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
          lastContent.parts.push({
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
