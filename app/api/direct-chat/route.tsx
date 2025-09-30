import { type NextRequest, NextResponse } from "next/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY
const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY ?? process.env.NEXT_PUBLIC_PERPLEXITY_API_KEY
const PERPLEXITY_MODEL = process.env.PERPLEXITY_MODEL ?? process.env.NEXT_PUBLIC_PERPLEXITY_MODEL
const DEFAULT_PERPLEXITY_MODEL = "sonar-pro"
const OPENAI_BASE_URL = "https://api.openai.com/v1"
const GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

const MAX_REFERENCES = 5

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

function getPerplexityConfig() {
  const apiKey = PERPLEXITY_API_KEY?.trim() || undefined;
  const model = PERPLEXITY_MODEL?.trim() || DEFAULT_PERPLEXITY_MODEL;
  return { apiKey, model };
}

function stripReferenceSection(text: string) {
  if (!text) return text;
  const normalized = text.replace(/\r\n/g, '\n');
  const lines = normalized.split('\n');
  for (let i = lines.length - 1; i >= 0; i--) {
    const simplified = lines[i]
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase()
      .trim();
    if (simplified.startsWith('nguon tham khao')) {
      return lines.slice(0, i).join('\n').trim();
    }
  }
  return normalized.trim();
}

export async function POST(request: NextRequest) {
  try {
    const { messages, model, stream = false, files = [], workflow = "single" } = await request.json()
    const modelConfig = MODEL_MAPPING[model as keyof typeof MODEL_MAPPING]
    if (!modelConfig) {
      return NextResponse.json({ error: `Unsupported model: ${model}` }, { status: 400 })
    }
    const { provider, model: actualModel } = modelConfig
    if (workflow === "chatgpt-to-gemini") {
      return await handleChatGPTToGemini(messages, actualModel, stream, files)
    }
    if (workflow === "perplexity-to-gemini") {
      return await handlePerplexityToGemini(messages, actualModel, stream, files)
    }
    if (workflow === "perplexity-chatgpt-gemini") {
      return await handlePerplexityChatGPTToGemini(messages, actualModel, stream, files)
    }
    return await handleGoogle(messages, actualModel, stream, files)
  } catch (error) {
    console.error("Direct Chat API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

async function handleOpenAI(messages: any[], model: string, stream: boolean, files: any[]) {
  if (!OPENAI_API_KEY) {
    return NextResponse.json({ error: "OpenAI API key not configured" }, { status: 500 })
  }
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
      conversationLength: messages.length,
    })
    const tools = [
      {
        googleSearchRetrieval: {},
      },
    ]
    const systemInstruction = {
      role: 'user',
      parts: [{
        text: 'Bạn là một trợ lý AI hữu ích. Khi trả lời câu hỏi, vui lòng tuân thủ các yêu cầu sau:\n' +
          '1. Không sử dụng bất kỳ định dạng markdown nào (không **, ##, ```, v.v.)\n' +
          '2. Trả lời bằng văn bản thuần, không cần xuống dòng thừa\n' +
          '3. Sử dụng các dấu số thứ tự (1, 2, 3) để liệt kê nếu cần\n' +
          '4. Trả lời bằng tiếng Việt\n' +
          '5. Giữ câu trả lời ngắn gọn, súc tích\n' +
          '6. Không tự thêm phần "Nguồn tham khảo" trong câu trả lời; hệ thống sẽ hiển thị riêng nếu có dữ liệu kèm theo\n' +
          '7. Nếu không có nguồn, chỉ cần trả lời nội dung chính xác, không bổ sung ghi chú nào\n\n' +
          'LƯU Ý QUAN TRỌNG: TUYỆT ĐỐI KHÔNG sử dụng bất kỳ định dạng markdown nào. Chỉ trả lời bằng văn bản thuần.'
      }]
    };
    const geminiModel = genAI.getGenerativeModel({ 
      model,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1000,
      }
    });
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
    const responseText = stripReferenceSection(cleanText);
    const displayFiles = uploadedFiles.slice(0, MAX_REFERENCES);
    // Ensure sources section is formatted with clickable URLs
    const sourcesSection = displayFiles.length > 0
      ? `Nguồn tham khảo:
${displayFiles.map((file, index) => `${index + 1}) ${file.name} - ${file.fileUri}`).join('\n')}`
      : 'Nguồn tham khảo: (không có)';
    const responseParts = responseText ? [responseText] : [];
    responseParts.push(sourcesSection);
    const finalText = responseParts.join('\n\n');
    console.log("Google Gemini SDK response received:", {
      originalLength: text.length,
      cleanedLength: cleanText.length,
      textPreview: cleanText.substring(0, 200) + (cleanText.length > 200 ? "..." : ""),
    })
    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text: finalText }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: response.usageMetadata,
      choices: [
        {
          message: {
            role: "assistant",
            content: finalText,
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

async function handleChatGPTToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  const genAI = getGenAI();
  if (!GOOGLE_API_KEY || !genAI) {
    return NextResponse.json({ error: "Google API key not configured" }, { status: 500 })
  }
  if (!OPENAI_API_KEY) {
    return NextResponse.json({ error: "OpenAI API key not configured for ChatGPT-to-Gemini workflow" }, { status: 500 })
  }
  try {
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
    console.log("[ChatGPT-to-Gemini] Step 1: Generating prompt with ChatGPT")
    const optimizedPrompt = await convertToPromptChatGPT(userQuestion)
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
        return {
          ...msg,
          content: enhancedPrompt,
        }
      }
      return msg
    })
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

async function handlePerplexityToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  const genAI = getGenAI();
  if (!GOOGLE_API_KEY || !genAI) {
    return NextResponse.json({ error: "Google API key not configured" }, { status: 500 });
  }
  try {
    const lastMessage = messages[messages.length - 1];
    const userQuestion =
      typeof lastMessage.content === "string"
        ? lastMessage.content
        : lastMessage.content?.find((c: any) => c.type === "text")?.text || "";
    if (!userQuestion.trim()) {
      return NextResponse.json({ error: "No question provided" }, { status: 400 });
    }
    let conversationContext = "";
    if (messages.length > 1) {
      conversationContext = messages
        .slice(0, -1)
        .map((msg: any) => {
          const roleLabel = msg.role === "user" ? "Người dùng" : "Trợ lý";
          const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
          return `${roleLabel}: ${content}`;
        })
        .join("\n");
    }
    console.log("[Perplexity-to-Gemini] Step 1: Searching with Perplexity API");
    const searchResults = await searchWithPerplexityStyle(userQuestion);
    if (searchResults?.error) {
      console.error("[Perplexity-to-Gemini] Perplexity search failed:", searchResults.error);
      return NextResponse.json(
        {
          error: searchResults.error.message || "Perplexity search failed.",
          errorType: "perplexity_search_error",
          details: searchResults.error,
        },
        { status: 502 },
      );
    }
    const searchAnswer = typeof searchResults?.answer === "string" ? searchResults.answer.trim() : "";
    const researchResults: any[] = Array.isArray(searchResults?.results) ? searchResults.results : [];
    console.log("[Perplexity-to-Gemini] Perplexity response:", {
      hasAnswer: !!searchAnswer,
      answerPreview: searchAnswer ? searchAnswer.slice(0, 200) : null,
      resultsCount: researchResults.length,
      sampleResults: researchResults.slice(0, 2),
    });
    // Create search sections for the prompt
    const searchSections: string[] = [];
    if (searchAnswer) {
      searchSections.push(`TÓM TẮT TỪ NGUỒN TÌM KIẾM:\n${searchAnswer}`);
    }
    let referencesText = ''
    if (researchResults.length > 0) {
      const limitedResults = researchResults.slice(0, MAX_REFERENCES);
      referencesText = limitedResults
        .map((result: any, index: number) => {
          const rawUrl = typeof result.url === 'string' ? result.url.trim() : '';
          const url = rawUrl || '(khong co URL)';
          let title = (
            typeof result.title === 'string' && result.title.trim()
          ) || (
            typeof result.source === 'string' && result.source.trim()
          ) || '';
          if (!title && rawUrl) {
            try {
              const hostname = new URL(rawUrl).hostname.replace(/^www\./, '');
              title = hostname || title;
            } catch (error) {
              // ignore invalid URL
            }
          }
          if (!title) {
            title = `Nguon ${index + 1}`;
          }
          const content = result.content || '';
          return `${index + 1}. ${title} - ${url}\n${content}`;
        })
        .join('\n');
      searchSections.push(`THAM KHAO:
${referencesText}`);
    }
    // Construct enhanced prompt for Gemini
    const promptParts: string[] = [
      'Ban la mot tro ly AI huu ich. Hay tra loi cau hoi dua tren du lieu tim kiem duoc cung cap tu Perplexity, tuan thu cac yeu cau sau:\n' +
      '1. Khong su dung bat ky dinh dang markdown nao (khong **, ##, ``` , v.v.)\n' +
      '2. Tra loi bang van ban thuan, khong can xuong dong thua\n' +
      '3. Su dung cac dau so thu tu (1, 2, 3) de liet ke cac y chinh neu can\n' +
      '4. Tra loi bang tieng Viet\n' +
      '5. Chi tong hop thong tin duoc cung cap tu Perplexity, khong them nguon ben ngoai\n' +
      '6. Khong chen phan "Nguon tham khao" trong cau tra loi; he thong se hien thi phan nay tu du lieu dau vao\n' +
      '7. Neu can nhac den nguon, chi de cap ten hoac nguon goc trong noi dung, khong dinh kem URL trong cau tra loi'
    ];
    if (conversationContext) {
      promptParts.push(`Ngữ cảnh cuộc trò chuyện:\n${conversationContext}`);
    }
    if (searchSections.length > 0) {
      promptParts.push(`Dữ liệu tìm kiếm từ Perplexity:\n${searchSections.join("\n\n")}`);
    }
    promptParts.push(`Câu hỏi hiện tại: ${userQuestion}`);
    const enhancedPrompt = promptParts.join("\n\n");
    const processedMessages = messages.map((msg, index) => {
      if (index === messages.length - 1) {
        return {
          ...msg,
          content: enhancedPrompt,
        };
      }
      return msg;
    });
    console.log("[Perplexity-to-Gemini] Step 2: Calling Gemini with enhanced prompt");
    const geminiModel = genAI.getGenerativeModel({ model });
    const result = await geminiModel.generateContent({
      contents: processedMessages.map((msg) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content) }],
      })),
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 2000,
      },
    });
    const response = await result.response;
    let text = response.text();
    // Clean up any remaining markdown characters
    const cleanText = text
      .replace(/\*\*(.*?)\*\*/g, "$1")
      .replace(/\*(.*?)\*/g, "$1")
      .replace(/`(.*?)`/g, "$1")
      .replace(/^#+\s+/gm, "")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/^[-*+]\s+/gm, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    const responseText = stripReferenceSection(cleanText);
    const responseParts = responseText ? [responseText] : [];
    if (referencesText) {
      responseParts.push(`Nguồn tham khảo:
${referencesText}`);
    }
    const finalText = responseParts.length > 0 ? responseParts.join('\n\n') : responseText;
    console.log("[Perplexity-to-Gemini] Completed:", {
      originalLength: text.length,
      cleanedLength: cleanText.length,
      hasSummary: !!searchAnswer,
      searchResultsCount: researchResults.length,
    });
    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text: finalText }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      choices: [
        {
          message: {
            role: "assistant",
            content: finalText,
          },
        },
      ],
      searchResults,
      workflow: "perplexity-to-gemini",
      step1: {
        type: "perplexity_search",
        query: userQuestion,
        resultsCount: researchResults.length,
        prompt: searchAnswer || "Perplexity did not return a summary.",
        summary: searchAnswer || null,
        references: researchResults,
      },
      step2: {
        type: "gemini_response",
        promptLength: enhancedPrompt.length,
        content: finalText.substring(0, 200) + (finalText.length > 200 ? "..." : ""),
      },
    });
  } catch (error: any) {
    console.error("Perplexity-to-Gemini error:", error);
    return NextResponse.json(
      {
        error: error?.message || "Failed to process Perplexity-to-Gemini workflow",
        errorType: "workflow_error",
      },
      { status: 500 },
    );
  }
}

async function handlePerplexityChatGPTToGemini(messages: any[], model: string, stream: boolean, files: any[]) {
  const genAI = getGenAI();
  if (!GOOGLE_API_KEY || !genAI || !OPENAI_API_KEY) {
    return NextResponse.json({ error: "API keys not configured" }, { status: 500 });
  }
  try {
    const lastMessage = messages[messages.length - 1];
    const userQuestion =
      typeof lastMessage.content === "string"
        ? lastMessage.content
        : lastMessage.content?.find((c: any) => c.type === "text")?.text || "";
    if (!userQuestion.trim()) {
      return NextResponse.json({ error: "No question provided" }, { status: 400 });
    }
    console.log("[Perplexity-ChatGPT-Gemini] Step 1: Searching with Perplexity API");
    const searchResults = await searchWithPerplexityStyle(userQuestion);
    if (searchResults?.error) {
      console.error("[Perplexity-ChatGPT-Gemini] Perplexity search failed:", searchResults.error);
      return NextResponse.json(
        {
          error: searchResults.error.message || "Perplexity search failed.",
          errorType: "perplexity_search_error",
          details: searchResults.error,
        },
        { status: 502 },
      );
    }
    const searchAnswer = typeof searchResults?.answer === "string" ? searchResults.answer.trim() : "";
    const researchResults: any[] = Array.isArray(searchResults?.results) ? searchResults.results : [];
    console.log("[Perplexity-ChatGPT-Gemini] Perplexity response:", {
      hasAnswer: !!searchAnswer,
      answerPreview: searchAnswer ? searchAnswer.slice(0, 200) : null,
      resultsCount: researchResults.length,
      sampleResults: researchResults.slice(0, 2),
    });
    const searchContextSections: string[] = [];
    let chatReferencesText = '';
    if (searchAnswer) {
      searchContextSections.push(`Summary:\n${searchAnswer}`);
    }
    if (researchResults.length > 0) {
      const limitedResults = researchResults.slice(0, MAX_REFERENCES);
      chatReferencesText = limitedResults
        .map((result: any, index: number) => {
          const rawUrl = typeof result.url === 'string' ? result.url.trim() : '';
          const url = rawUrl || '(khong co URL)';
          let title = (
            typeof result.title === 'string' && result.title.trim()
          ) || (
            typeof result.source === 'string' && result.source.trim()
          ) || '';
          if (!title && rawUrl) {
            try {
              const hostname = new URL(rawUrl).hostname.replace(/^www\./, '');
              title = hostname || title;
            } catch (error) {
              // ignore invalid URL
            }
          }
          if (!title) {
            title = `Nguon ${index + 1}`;
          }
          const content = result.content || '';
          return `${index + 1}. ${title} - ${url}\n${content}`;
        })
        .join('\n');
      searchContextSections.push(`References:
${chatReferencesText}`);
    }
    const searchContext = searchContextSections.join("\n\n") || "No additional research data was returned.";
    console.log("[Perplexity-ChatGPT-Gemini] Step 2: Refining with ChatGPT");
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
            content:
              "You are an assistant that crafts detailed Vietnamese prompts for Gemini based on research data. Keep the prompt clear and avoid Markdown formatting.",
          },
          {
            role: "user",
            content: `RESEARCH DATA:
${searchContext}

ORIGINAL QUESTION: ${userQuestion}

TASK:
1. Create a detailed Vietnamese prompt for Gemini using the research summary and references.
2. Preserve any critical details from the original question.
3. Provide explicit guidance on how Gemini should structure the reply.
4. Require Gemini to answer in Vietnamese plain text without Markdown symbols.
5. Remind Gemini to use only the provided research information and not invent additional sources.
6. Do not instruct Gemini to add a "Nguon tham khao" section; the system will present references separately.

Return only the optimized prompt.`,
          },
        ],
        temperature: 0.7,
        max_tokens: 1000,
      }),
    });
    if (!chatGPTResponse.ok) {
      throw new Error("Failed to get response from ChatGPT");
    }
    const chatGPTData = await chatGPTResponse.json();
    const refinedPrompt = chatGPTData.choices?.[0]?.message?.content || userQuestion;
    console.log("[Perplexity-ChatGPT-Gemini] Step 3: Calling Gemini with refined prompt");
    const geminiModel = genAI.getGenerativeModel({
      model,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 2000,
      },
    });
    const result = await geminiModel.generateContent({
      contents: [
        {
          role: "user",
          parts: [{ text: refinedPrompt }],
        },
      ],
    });
    const response = await result.response;
    const text = response.text();
    const cleanText = text
      .replace(/\*\*(.*?)\*\*/g, "$1")
      .replace(/\*(.*?)\*/g, "$1")
      .replace(/`(.*?)`/g, "$1")
      .replace(/^#+\s+/gm, "")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/^[-*+]\s+/gm, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    const responseText = stripReferenceSection(cleanText);
    const responseParts = responseText ? [responseText] : [];
    if (chatReferencesText) {
      responseParts.push(`Nguồn tham khảo:
${chatReferencesText}`);
    }
    const finalText = responseParts.length > 0 ? responseParts.join('\n\n') : responseText;
    console.log("[Perplexity-ChatGPT-Gemini] Completed:", {
      originalLength: text.length,
      cleanedLength: cleanText.length,
      hasSummary: !!searchAnswer,
      searchResultsCount: researchResults.length,
    });
    return NextResponse.json({
      candidates: [
        {
          content: {
            parts: [{ text: finalText }],
            role: "model",
          },
          finishReason: "STOP",
        },
      ],
      choices: [
        {
          message: {
            role: "assistant",
            content: finalText,
          },
        },
      ],
      searchResults,
      workflow: "perplexity-chatgpt-gemini",
      referencesText: chatReferencesText,
      steps: {
        step1: {
          type: "perplexity_search",
          query: userQuestion,
          resultsCount: researchResults.length,
          prompt: searchAnswer || "Perplexity did not return a summary.",
          summary: searchAnswer || null,
          references: researchResults,
        },
        step2: {
          type: "chatgpt_refinement",
          prompt: refinedPrompt,
        },
        step3: {
          type: "gemini_response",
          content: finalText.substring(0, 200) + (finalText.length > 200 ? "..." : ""),
        },
      },
    });
  } catch (error: any) {
    console.error("Perplexity-ChatGPT-Gemini error:", error);
    return NextResponse.json(
      {
        error: error?.message || "Failed to process Perplexity-ChatGPT-Gemini workflow",
        errorType: "workflow_error",
      },
      { status: 500 },
    );
  }
}

async function searchWithPerplexityStyle(query: string) {
  const { apiKey, model } = getPerplexityConfig();
  if (!apiKey) {
    console.warn("Perplexity API key not configured, skipping search");
    return {
      error: {
        type: "missing_api_key",
        message: "Perplexity API key is not configured.",
      },
    };
  }
  const extractText = (value: any): string => {
    if (!value) return "";
    if (typeof value === "string") return value;
    if (Array.isArray(value)) {
      return value
        .map((item) => extractText(item?.text ?? item?.content ?? item?.value ?? item))
        .filter(Boolean)
        .join("\n");
    }
    if (typeof value === "object") {
      if (value.text) return extractText(value.text);
      if (value.content) return extractText(value.content);
      if (value.value) return extractText(value.value);
      if (Array.isArray(value.parts)) return extractText(value.parts);
    }
    return "";
  };
  try {
    console.log("[Perplexity] Searching for:", query.substring(0, 100) + "...");
    const response = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content:
              "You are a research assistant. Use the latest information you can find and cite clear sources.",
          },
          {
            role: "user",
            content: `Find the most recent information for the question below and return a concise summary with reliable sources.\n\nQUESTION: ${query}`,
          },
        ],
        temperature: 0.2,
        top_p: 0.8,
        top_k: 0,
        stream: false,
      }),
    });
    if (!response.ok) {
      const rawErrorText = await response.text().catch(() => "");
      let parsedError: any = null;
      if (rawErrorText) {
        try {
          parsedError = JSON.parse(rawErrorText);
        } catch {
          parsedError = null;
        }
      }
      const apiErrorMessage =
        parsedError?.error?.message ||
        parsedError?.message ||
        rawErrorText ||
        "Perplexity API error";
      console.error("[Perplexity] Search API error:", response.status, response.statusText, apiErrorMessage);
      return {
        error: {
          type: "perplexity_api_error",
          status: response.status,
          statusText: response.statusText,
          message: apiErrorMessage,
          raw: parsedError ?? (rawErrorText || null),
        },
      };
    }
    const data = await response.json();
    const choice = data.choices?.[0] || null;
    const message = choice?.message;
    let summary = extractText(message?.content);
    if (!summary && typeof choice?.text === "string") {
      summary = choice.text;
    }
    if (!summary && typeof data.answer === "string") {
      summary = data.answer;
    }
    if (!summary && Array.isArray(data.output_text)) {
      summary = data.output_text.join("\n");
    }
    summary = summary ? summary.toString().trim() : "";
    const citationCandidates: any[] = [];
    const pushGroup = (group: any) => {
      if (Array.isArray(group)) {
        citationCandidates.push(...group);
      }
    };
    pushGroup((message as any)?.citation_metadata?.citations);
    pushGroup(choice?.metadata?.citations);
    pushGroup(choice?.metadata?.sources);
    pushGroup(choice?.metadata?.web_results);
    pushGroup(data?.metadata?.citations);
    pushGroup(data?.citations);
    pushGroup(data?.sources);
    pushGroup(data?.web_results);
    pushGroup(data?.results);
    const seen = new Set<string>();
    const results: Array<{ title: string; url: string; content: string }> = [];
    citationCandidates.forEach((citation: any, index: number) => {
      let entry: { title: string; url: string; content: string } | null = null;
      if (typeof citation === "string") {
        entry = {
          title: `Source ${index + 1}`,
          url: citation,
          content: "",
        };
      } else if (citation && typeof citation === "object") {
        const title = citation.title || citation.source || citation.url || `Source ${index + 1}`;
        const url = citation.url || citation.source || citation.link || "";
        const content =
          citation.snippet ||
          citation.text ||
          citation.content ||
          citation.summary ||
          "";
        entry = {
          title,
          url,
          content,
        };
      }
      if (entry) {
        const key = `${entry.url || ""}|${entry.title}`;
        if (!seen.has(key)) {
          seen.add(key);
          results.push(entry);
        }
      }
    });
    console.log("[Perplexity] Search completed:", {
      hasSummary: !!summary,
      summaryPreview: summary ? summary.slice(0, 200) : null,
      resultsCount: results.length,
      sampleResults: results.slice(0, 3),
    });
    return {
      answer: summary,
      results,
      raw: data,
    };
  } catch (error) {
    console.error("[Perplexity] Search error:", error);
    return {
      error: {
        type: "perplexity_network_error",
        message: error instanceof Error ? error.message : String(error),
      },
    };
  }
}

async function uploadFilesToGeminiSDK(files: any[]) {
  const genAI = getGenAI();
  if (!genAI) return []
  const uploadedFiles = []
  for (const file of files) {
    const needsUpload = [
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.presentationml.presentation",
      "application/pdf",
    ].includes(file.type)
    if (needsUpload) {
      try {
        console.log(`Uploading ${file.name} to Gemini Files API using SDK...`)
        const buffer = Buffer.from(file.data, "base64")
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

async function prepareMessagesForGemini(
  messages: any[],
  files: any[],
  {
    systemPrompt,
    replaceLastUser,
    attachFiles = true,
    inlineFallback = true,
  }: {
    systemPrompt?: string
    replaceLastUser?: string
    attachFiles?: boolean
    inlineFallback?: boolean
  } = {},
) {
  const normalizeMessage = (msg: any) => {
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
  const arr = (messages || []).map(normalizeMessage).filter(Boolean)
  if (replaceLastUser) {
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i].role === "user") {
        arr[i] = { ...arr[i], parts: [{ text: replaceLastUser }] }
        break
      }
    }
  }
  let uploadedFiles: any[] = []
  let fileParts: any[] = []
  if (attachFiles && files?.length) {
    try {
      uploadedFiles = await uploadFilesToGeminiSDK(files)
      if (uploadedFiles?.length) {
        fileParts = uploadedFiles
          .map((f: any) => {
            const uri = f?.fileUri || f?.uri || f?.name
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
  const contents = arr.map((m) => ({ role: m.role, parts: m.parts }))
  return {
    contents,
    systemInstruction: systemPrompt || undefined,
    uploadedFiles,
    attachedPartsCount: fileParts.length,
  }
}

async function processMessagesForOpenAI(messages: any[], files: any[]) {
  const processedMessages = [...messages]
  if (files.length > 0 && processedMessages.length > 0) {
    const lastMessage = processedMessages[processedMessages.length - 1]
    if (lastMessage.role === "user") {
      if (typeof lastMessage.content === "string") {
        lastMessage.content = [
          {
            type: "text",
            text: lastMessage.content,
          },
        ]
      }
      files.forEach((file: any) => {
        if (file.type.startsWith("image/")) {
          lastMessage.content.push({
            type: "image_url",
            image_url: {
              url: `data:${file.type};base64,${file.data}`,
            },
          })
        } else if (file.type === "application/pdf") {
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
"Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi sau một cách rõ ràng, dễ hiểu:\n\nCâu hỏi: [câu hỏi user]\n\nYêu cầu:\n- Không sử dụng bất kỳ ký hiệu markdown nào như **, ##, \`\`\`, v.v.\n- Trình bày thông tin rõ ràng, mạch lạc\n- Sử dụng các số thứ tự (1, 2, 3) để liệt kê các ý chính\n- Mỗi ý chính nên có phần tóm tắt ngắn gọn và giải thích chi tiết\n- Nếu có nguồn tham khảo, chỉ liệt kê tối đa 5 nguồn ở cuối câu trả lời\n- Luôn trả lời bằng tiếng Việt\n- Tuyệt đối không sử dụng bất kỳ ký hiệu đặc biệt nào để định dạng văn bản"

CHÚ Ý:
- [câu hỏi user] = copy y nguyên câu hỏi của user
- Chỉ xuất prompt, không giải thích gì thêm
- Đảm bảo prompt yêu cầu Gemini không sử dụng bất kỳ định dạng markdown nào
`;

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
      temperature: 0,
      top_p: 0,
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
        if (content.type === "text") {
          parts.push({ text: content.text })
        } else if (content.type === "image_url" && content.image_url?.url) {
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
  if (files.length > 0 && processedContents.length > 0) {
    const lastContent = processedContents[processedContents.length - 1]
    if (lastContent.role === "user") {
      files.forEach((file: any) => {
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
