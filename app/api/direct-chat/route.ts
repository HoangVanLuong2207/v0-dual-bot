import { type NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

const OPENAI_BASE_URL = "https://api.openai.com/v1";
const GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

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
  "gemini-2.5-flash-preview-image-generation": { provider: "google", model: "gemini-2.5-flash-preview-image-generation" },
  "gemini-2.5-flash-live-preview-04-09": { provider: "google", model: "gemini-2.5-flash-live-preview-04-09" },
};

const genAI = GOOGLE_API_KEY ? new GoogleGenerativeAI(GOOGLE_API_KEY) : null;

export async function POST(request: NextRequest) {
  try {
    const {
      messages,
      model,
      stream = false,
      files = [],
      workflow = "single",
    } = await request.json();

    const modelConfig = MODEL_MAPPING[model as keyof typeof MODEL_MAPPING];
    if (!modelConfig) {
      return NextResponse.json(
        { error: `Unsupported model: ${model}` },
        { status: 400 }
      );
    }

    const { provider, model: actualModel } = modelConfig;
    return await handleGoogle(messages, actualModel, stream, files);

    // if (workflow === "chatgpt-to-gemini") {
    //   return await handleChatGPTToGeminiWorkflow(messages, files);
    // }

    // if (provider === "openai") {
    //   return await handleOpenAI(messages, actualModel, stream, files);
    // } else if (provider === "google") {
    //   return await handleGoogle(messages, actualModel, stream, files);
    // }

    // return NextResponse.json({ error: "Invalid provider" }, { status: 400 });
  } catch (error) {
    console.error("Direct Chat API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

async function handleOpenAI(
  messages: any[],
  model: string,
  stream: boolean,
  files: any[]
) {
  if (!OPENAI_API_KEY) {
    return NextResponse.json(
      { error: "OpenAI API key not configured" },
      { status: 500 }
    );
  }

  // Process files for OpenAI (supports images, not PDFs directly)
  const processedMessages = await processMessagesForOpenAI(messages, files);

  const requestBody = {
    model,
    messages: processedMessages,
    stream,
    temperature: 0.7,
    max_tokens: 2000,
  };

  console.log("Sending request to OpenAI:", {
    model,
    messagesCount: processedMessages.length,
    filesCount: files.length,
  });

  try {
    const response = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("OpenAI API error:", errorData);

      if (errorData.error?.code === "insufficient_quota") {
        return NextResponse.json(
          {
            error:
              "OpenAI quota exceeded. Please check your billing or try using Gemini instead.",
            errorType: "quota_exceeded",
            fallbackSuggestion: "gemini",
          },
          { status: 402 }
        );
      }

      if (errorData.error?.code === "invalid_api_key") {
        return NextResponse.json(
          {
            error: "Invalid OpenAI API key. Please check your configuration.",
            errorType: "invalid_key",
          },
          { status: 401 }
        );
      }

      return NextResponse.json(
        {
          error:
            errorData.error?.message || "Failed to get response from OpenAI",
          errorType: "api_error",
        },
        { status: response.status }
      );
    }

    if (stream) {
      return new Response(response.body, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else {
      const data = await response.json();
      return NextResponse.json(data);
    }
  } catch (error) {
    console.error("OpenAI request failed:", error);
    return NextResponse.json(
      {
        error: "Network error when calling OpenAI API",
        errorType: "network_error",
      },
      { status: 500 }
    );
  }
}

async function handleGoogle(
  messages: any[],
  model: string,
  stream: boolean,
  files: any[]
) {
  if (!GOOGLE_API_KEY || !genAI) {
    return NextResponse.json(
      { error: "Google API key not configured" },
      { status: 500 }
    );
  }

  try {
    console.log("messages 👉", messages, files);
    const uploadedFiles = await uploadFilesToGeminiSDK(files);

    const { processedContents, prompts } = await processMessagesForGoogleSDK(
      messages,
      files,
      uploadedFiles
    );

    console.log("Sending request to Google Gemini SDK:", {
      model,
      contentsCount: processedContents.length,
      filesCount: files.length,
      uploadedFilesCount: uploadedFiles.length,
    });

    // 1. Khai báo công cụ bạn muốn sử dụng
    const tools = [
      {
        googleSearchRetrieval: {}, // Cú pháp cho Google Search trong Node.js/JS
      },
    ];

    const geminiModel = genAI.getGenerativeModel({ model });

    const result = await geminiModel.generateContent({
      contents: processedContents,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1000,
      },
    });

    const response = await result.response;
    const text = response.text();

    console.log("Google Gemini SDK response received:", {
      textLength: text.length,
      textPreview: text.substring(0, 200) + "...",
    });

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
    });
  } catch (error) {
    console.error("Google GenAI SDK error:", error);

    if (error.message?.includes("API key")) {
      return NextResponse.json(
        {
          error: "Invalid Google API key. Please check your configuration.",
          errorType: "invalid_key",
        },
        { status: 401 }
      );
    }

    if (error.message?.includes("quota") || error.message?.includes("limit")) {
      return NextResponse.json(
        {
          error: "Google API quota exceeded. Please try again later.",
          errorType: "quota_exceeded",
        },
        { status: 429 }
      );
    }

    return NextResponse.json(
      {
        error: error.message || "Failed to get response from Google Gemini",
        errorType: "api_error",
      },
      { status: 500 }
    );
  }
}

async function uploadFilesToGeminiSDK(files: any[]) {
  if (!genAI) return [];

  const uploadedFiles = [];

  for (const file of files) {
    // Upload Office files and PDFs via Files API
    const needsUpload = [
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
      "application/msword", // .doc
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
      "application/vnd.ms-excel", // .xls
      "application/vnd.openxmlformats-officedocument.presentationml.presentation", // .pptx
      "application/pdf", // PDF files
    ].includes(file.type);

    if (needsUpload) {
      try {
        console.log(`Uploading ${file.name} to Gemini Files API using SDK...`);

        const buffer = Buffer.from(file.data, "base64");

        const uploadResult = await genAI.files.upload({
          file: buffer,
          mimeType: file.type,
          displayName: file.name,
        });

        console.log(`Successfully uploaded ${file.name}:`, {
          uri: uploadResult.file.uri,
          name: uploadResult.file.name,
          mimeType: uploadResult.file.mimeType,
        });

        uploadedFiles.push({
          originalFile: file,
          fileUri: uploadResult.file.uri,
          name: uploadResult.file.name,
          mimeType: uploadResult.file.mimeType,
        });
      } catch (error) {
        console.error(`Error uploading ${file.name}:`, error);
      }
    }
  }

  return uploadedFiles;
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
    systemPrompt?: string; // sẽ trả ra qua field systemInstruction (KHÔNG thêm vào contents)
    replaceLastUser?: string;
    attachFiles?: boolean;
    inlineFallback?: boolean;
  } = {}
) {
  const normalizeMessage = (msg: any) => {
    // Loại bỏ mọi message role=system trong contents để tránh 400
    if (msg?.role === "system") return null;

    if (msg?.parts && Array.isArray(msg.parts)) return msg;

    if (typeof msg?.content === "string") {
      return { role: msg.role, parts: [{ text: msg.content }] };
    }

    if (Array.isArray(msg?.content)) {
      const parts = msg.content
        .map((c: any) => {
          if (c?.type === "text" && typeof c.text === "string")
            return { text: c.text };
          if (c?.inlineData?.data && c?.inlineData?.mimeType)
            return {
              inlineData: {
                data: c.inlineData.data,
                mimeType: c.inlineData.mimeType,
              },
            };
          if (c?.fileData?.fileUri && c?.fileData?.mimeType)
            return {
              fileData: {
                fileUri: c.fileData.fileUri,
                mimeType: c.fileData.mimeType,
              },
            };
          return null;
        })
        .filter(Boolean);
      return { role: msg.role, parts };
    }

    return { role: msg.role, parts: [] };
  };

  // 1) Normalize & loại system khỏi contents
  let arr = (messages || []).map(normalizeMessage).filter(Boolean);

  // 2) Optional: replace last user
  if (replaceLastUser) {
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i].role === "user") {
        arr[i] = { ...arr[i], parts: [{ text: replaceLastUser }] };
        break;
      }
    }
  }

  // 3) Upload & attach files vào last user
  let uploadedFiles: any[] = [];
  let fileParts: any[] = [];
  if (attachFiles && files?.length) {
    try {
      uploadedFiles = await uploadFilesToGeminiSDK(files);
      if (uploadedFiles?.length) {
        fileParts = uploadedFiles
          .map((f: any) => {
            const uri = f?.fileUri || f?.uri || f?.name; // "files/xxx" cũng OK
            const mime = f?.mimeType;
            if (!uri || !mime) return null;
            return { fileData: { fileUri: uri, mimeType: mime } };
          })
          .filter(Boolean);
      }
    } catch (e) {
      console.error(
        "prepareMessagesForGemini: upload error, fallback inline if enabled",
        e
      );
    }

    if (!fileParts.length && inlineFallback) {
      const inlineParts = [];
      for (const orig of files) {
        try {
          const mimeType =
            orig.type || orig.mimeType || "application/octet-stream";
          if (orig?.arrayBuffer) {
            const ab = await orig.arrayBuffer();
            inlineParts.push({
              inlineData: {
                mimeType,
                data: Buffer.from(ab as any).toString("base64"),
              },
            });
          } else if (orig?.buffer) {
            inlineParts.push({
              inlineData: {
                mimeType,
                data: Buffer.from(orig.buffer).toString("base64"),
              },
            });
          } else if (orig?.path) {
            const { readFileSync } = await import("node:fs");
            inlineParts.push({
              inlineData: {
                mimeType,
                data: readFileSync(orig.path).toString("base64"),
              },
            });
          }
        } catch (e) {
          console.error("prepareMessagesForGemini: inline fallback failed", e);
        }
      }
      fileParts = inlineParts;
    }

    if (fileParts.length) {
      for (let i = arr.length - 1; i >= 0; i--) {
        if (arr[i].role === "user") {
          arr[i] = {
            ...arr[i],
            parts: [...(arr[i].parts || []), ...fileParts],
          };
          break;
        }
      }
    }
  }

  // 4) Trả contents + systemInstruction (KHÔNG thêm system vào contents)
  const contents = arr.map((m) => ({ role: m.role, parts: m.parts }));

  return {
    contents,
    systemInstruction: systemPrompt || undefined, // dùng ở bước getGenerativeModel hoặc generateContent
    uploadedFiles,
    attachedPartsCount: fileParts.length,
  };
}

async function processMessagesForOpenAI(messages: any[], files: any[]) {
  const processedMessages = [...messages];

  // Add files to the last user message for OpenAI
  if (files.length > 0 && processedMessages.length > 0) {
    const lastMessage = processedMessages[processedMessages.length - 1];
    if (lastMessage.role === "user") {
      // Convert content to array format if it's a string
      if (typeof lastMessage.content === "string") {
        lastMessage.content = [
          {
            type: "text",
            text: lastMessage.content,
          },
        ];
      }

      // Add supported files (images only for OpenAI)
      files.forEach((file: any) => {
        if (file.type.startsWith("image/")) {
          lastMessage.content.push({
            type: "image_url",
            image_url: {
              url: `data:${file.type};base64,${file.data}`,
            },
          });
        } else if (file.type === "application/pdf") {
          // For PDFs, add a text note since OpenAI doesn't support direct PDF processing
          lastMessage.content.push({
            type: "text",
            text: `[PDF File: ${file.name} - Note: OpenAI cannot directly process PDF files. Please extract text content manually.]`,
          });
        }
      });
    }
  }

  return processedMessages;
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
- Không dùng ký hiệu **, không markdown
- Có số liệu báo cáo nếu có
- Hiển thị ý chính đúng trọng tâm, ví dụ: đánh số 1. 2. 3.
- Mỗi ý: 1 câu tóm tắt + 1-2 câu giải thích
- Nếu có chỉ dẫn nguồn, số liệu thì Cuối mỗi ý ghi nguồn: <br /><strong>Nguồn:</strong> <a href='[link]' target='_blank'>[tên]</a>
Không có nguồn thì không ghi.
- Viết bằng tiếng Việt"

CHÚ Ý:
- [chủ đề] = thay bằng lĩnh vực phù hợp (tài chính, giáo dục, y tế, ngân hàng...)
- [câu hỏi user] = copy y nguyên câu hỏi của user
- Chỉ xuất prompt, không giải thích gì thêm
`;

/**
 * @param contentText
 * @returns
 */
async function convertToPromptChatGPT(contentText: string): Promise<string> {
  const systemWithQuestion = CONTENT_SYSTEM.replace(
    "{user_question}",
    contentText.trim()
  );
  const promptGenerationMessages = [
    {
      role: "system",
      content: systemWithQuestion,
    },
    { role: "user", content: contentText },
  ];

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
  });

  if (!chatgptResponse.ok) {
    const err = await chatgptResponse.json().catch(() => ({}));
    console.error("❌ ChatGPT API error:", err);
    return contentText;
  }

  const chatgptData = await chatgptResponse.json();
  const optimizedPrompt = String(
    chatgptData?.choices?.[0]?.message?.content || contentText
  );

  console.log("Prompt from ChatGPT ✨:", optimizedPrompt);
  return optimizedPrompt;
}

async function processMessagesForGoogleSDK(
  messages: any[],
  files: any[],
  uploadedFiles: any[] = []
) {
  const processedContents = [];
  const prompts = [];

  for (const message of messages) {
    const role = message.role === "assistant" ? "model" : "user";
    const parts = [];

    // Add text content
    if (typeof message.content === "string") {
      console.log("Processing message content (string):", {
        role,
        contentLength: message.content.length,
      });
      const contentText = await convertToPromptChatGPT(message.content);
      parts.push({ text: contentText });
      prompts.push({
        input: { system: CONTENT_SYSTEM, user: message.content },
        output: contentText,
      });
    } else if (Array.isArray(message.content)) {
      console.log("Processing message content (array):", {
        role,
        contentItems: message.content.length,
        contentTypes: message.content.map((c) => c.type),
      });

      message.content.forEach(async (content: any) => {
        // prompts.push({ input: null, output: null });
        if (content.type === "text") {
          // Convert to prompt
          // const contentText = await convertToPromptChatGPT(content.text);
          parts.push({ text: content.text });
        } else if (content.type === "image_url" && content.image_url?.url) {
          // Extract mime type and base64 data from data URL
          const dataUrl = content.image_url.url;
          const matches = dataUrl.match(/^data:([^;]+);base64,(.+)$/);

          if (matches) {
            const [, mimeType, base64Data] = matches;
            console.log("Adding image part:", { mimeType });
            parts.push({
              inlineData: {
                mimeType: mimeType,
                data: base64Data,
              },
            });
          }
        }
      });
    }

    if (parts.length > 0) {
      processedContents.push({
        role,
        parts,
      });
    }
  }

  if (uploadedFiles.length > 0 && processedContents.length > 0) {
    const lastContent = processedContents[processedContents.length - 1];
    if (lastContent.role === "user") {
      uploadedFiles.forEach((uploadedFile: any) => {
        console.log("Adding uploaded file part:", {
          name: uploadedFile.name,
          uri: uploadedFile.fileUri,
          mimeType: uploadedFile.mimeType,
        });

        lastContent.parts.push({
          fileData: {
            mimeType: uploadedFile.mimeType,
            fileUri: uploadedFile.fileUri,
          },
        });
      });
    }
  }

  // Add inline images (not uploaded files)
  if (files.length > 0 && processedContents.length > 0) {
    const lastContent = processedContents[processedContents.length - 1];
    if (lastContent.role === "user") {
      files.forEach((file: any) => {
        // Only use inlineData for images (Office files are uploaded)
        if (file.type.startsWith("image/")) {
          console.log("Adding inline image:", { type: file.type });
          lastContent.parts.push({
            inlineData: {
              mimeType: file.type,
              data: file.data,
            },
          });
        }
      });
    }
  }

  return { processedContents, prompts };
}
